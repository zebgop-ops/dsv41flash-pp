// Native CPU MoE for DeepSeek-V4.1 MXFP4 routed experts (AVX2 + FMA, OpenMP).
//
// Why: kt-kernel's AVX2 MXFP4 path runs one thread per selected expert and is
// compute-bound at ~3 GB/s of packed weights, so a decode step costs ~5.7 ms per
// layer no matter how many threads it gets. This kernel splits every expert's
// rows across all threads and streams the FP4 weights at memory bandwidth.
//
// Weights stay in the checkpoint's native MXFP4 (E2M1 nibbles, one UE8M0 scale
// per 32 elements along K); values are dequantized to fp32 exactly (the ×2 LUT
// is folded into the scale) and accumulated in fp32. Output is bf16 (RNE).
//
// Layout per expert (contiguous): w13 rows interleaved [I][w1 row K/2 bytes | w3
// row K/2 bytes], s13 [I][2][K/32], w2 [H][I/2], s2 [H][I/32].
//
// Activation vectors are pre-permuted per 32-group so that the 16 low nibbles of
// a group's 16 bytes pair with xp[g*32+0..15] and the 16 high nibbles with
// xp[g*32+16..31] (element 2j = low nibble of byte j, 2j+1 = high nibble).
#include <immintrin.h>
#include <omp.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>

namespace {

struct Layer {
    int E, H, I, threads;
    float lim;              // swiglu clamp limit (0 = off)
    uint8_t* w13 = nullptr; // [E][I][K] bytes (w1 row | w3 row), K = H
    uint8_t* s13 = nullptr; // [E][I][2][H/32]
    uint8_t* w2 = nullptr;  // [E][H][I/2]
    uint8_t* s2 = nullptr;  // [E][H][I/32]
    size_t w13_stride, s13_stride, w2_stride, s2_stride; // per expert, bytes
    // scratch
    std::vector<float> xp;     // [M][H] permuted fp32 activations
    std::vector<float> hp;     // [M*K][I] permuted fp32 intermediate
    std::vector<float> yp;     // [M*K][H] fp32 per-(token,slot) expert outputs
    std::vector<int> tok_of_e; // token/slot lists grouped by expert
    std::vector<int> e_start;  // [E+1]
};

static void* alloc_big(size_t n) {
    void* p = nullptr;
    if (posix_memalign(&p, 2u << 20, n) != 0) return nullptr;
    madvise(p, n, MADV_HUGEPAGE);
    return p;
}

static inline float e8m0_half(uint8_t s) {
    // 2^(s-127) * 0.5 (the LUT holds 2x the E2M1 magnitudes); s==0 -> 0 like the reference
    if (s == 0) return 0.0f;
    uint32_t bits = (uint32_t)(s - 1) << 23;
    float f; memcpy(&f, &bits, 4); return f;
}

// permuted position of element i (0..31) inside its group
static inline int perm_pos(int i) { return (i & 1) ? 16 + (i >> 1) : (i >> 1); }

static inline void permute_row(const float* src, float* dst, int K) {
    for (int g = 0; g < K / 32; ++g)
        for (int i = 0; i < 32; ++i) dst[g * 32 + perm_pos(i)] = src[g * 32 + i];
}

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

static const int8_t kLut[32] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
                                0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// 32 packed nibbles -> 4 fp32 vectors (x2 magnitudes) in permuted order
static inline void unpack32(const uint8_t* wp, __m256& f0, __m256& f1, __m256& f2, __m256& f3) {
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m256i lut = _mm256_loadu_si256((const __m256i*)kLut);
    __m128i b = _mm_loadu_si128((const __m128i*)wp);
    __m128i lo = _mm_and_si128(b, m4);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), m4);
    __m256i codes = _mm256_set_m128i(hi, lo);
    __m256i vals = _mm256_shuffle_epi8(lut, codes);
    __m128i v0 = _mm256_castsi256_si128(vals), v1 = _mm256_extracti128_si256(vals, 1);
    f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v0));
    f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v0, 8)));
    f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v1));
    f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v1, 8)));
}

// fused: dot(dequant(row), xp) for one token
static inline float dot_row(const uint8_t* wp, const uint8_t* sc, const float* xp, int K) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    const int G = K / 32;
    for (int g = 0; g < G; ++g) {
        __m256 f0, f1, f2, f3;
        _mm_prefetch((const char*)(wp + g * 16 + 512), _MM_HINT_T0);
        unpack32(wp + g * 16, f0, f1, f2, f3);
        const float* x = xp + g * 32;
        __m256 pa = _mm256_mul_ps(f0, _mm256_loadu_ps(x));
        __m256 pb = _mm256_mul_ps(f1, _mm256_loadu_ps(x + 8));
        pa = _mm256_fmadd_ps(f2, _mm256_loadu_ps(x + 16), pa);
        pb = _mm256_fmadd_ps(f3, _mm256_loadu_ps(x + 24), pb);
        __m256 s = _mm256_set1_ps(e8m0_half(sc[g]));
        acc0 = _mm256_fmadd_ps(pa, s, acc0);
        acc1 = _mm256_fmadd_ps(pb, s, acc1);
    }
    return hsum256(_mm256_add_ps(acc0, acc1));
}

// dequantize one row (scale applied) into permuted fp32 order
static inline void dequant_row(const uint8_t* wp, const uint8_t* sc, float* out, int K) {
    const int G = K / 32;
    for (int g = 0; g < G; ++g) {
        __m256 f0, f1, f2, f3;
        unpack32(wp + g * 16, f0, f1, f2, f3);
        __m256 s = _mm256_set1_ps(e8m0_half(sc[g]));
        float* o = out + g * 32;
        _mm256_storeu_ps(o, _mm256_mul_ps(f0, s));
        _mm256_storeu_ps(o + 8, _mm256_mul_ps(f1, s));
        _mm256_storeu_ps(o + 16, _mm256_mul_ps(f2, s));
        _mm256_storeu_ps(o + 24, _mm256_mul_ps(f3, s));
    }
}

static inline float dot_f32(const float* a, const float* b, int K) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (int i = 0; i < K; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    return hsum256(_mm256_add_ps(acc0, acc1));
}

static inline float silu(float g) { return g / (1.0f + expf(-g)); }

static inline uint16_t f32_to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)(u >> 16); // inf/nan
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}

static inline float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f;
}

static bool pread_all(int fd, void* dst, size_t n, int64_t off) {
    uint8_t* p = (uint8_t*)dst;
    while (n) {
        ssize_t r = pread(fd, p, n, off);
        if (r <= 0) return false;
        p += r; n -= (size_t)r; off += r;
    }
    return true;
}

} // namespace

namespace {
static void gateup_task(Layer* L, int task, const std::vector<int>& active, int nb13, const float* xp, float* hp,
                        float* tmp, int K, float lim) {
    const int H = L->H, I = L->I; const int RB = 32;
    const int e = active[task / nb13], rb = task % nb13;
    const int t0 = L->e_start[e], t1 = L->e_start[e + 1], nt = t1 - t0;
    const uint8_t* w = L->w13 + (size_t)e * L->w13_stride;
    const uint8_t* s = L->s13 + (size_t)e * L->s13_stride;
    float* dg = tmp; float* du = tmp + H;
    for (int r = rb * RB; r < (rb + 1) * RB; ++r) {
        const uint8_t* w1 = w + (size_t)r * H; const uint8_t* w3 = w1 + H / 2;
        const uint8_t* s1 = s + (size_t)r * 2 * (H / 32); const uint8_t* s3 = s1 + H / 32;
        const int pos = (r / 32) * 32 + perm_pos(r % 32);
        if (nt == 1) {
            const int i = L->tok_of_e[t0]; const int t = i / K;
            float g = dot_row(w1, s1, xp + (size_t)t * H, H);
            float u = dot_row(w3, s3, xp + (size_t)t * H, H);
            if (lim > 0) { u = std::min(std::max(u, -lim), lim); g = std::min(g, lim); }
            hp[(size_t)i * I + pos] = silu(g) * u;
        } else {
            dequant_row(w1, s1, dg, H); dequant_row(w3, s3, du, H);
            for (int q = t0; q < t1; ++q) {
                const int i = L->tok_of_e[q]; const int t = i / K;
                float g = dot_f32(dg, xp + (size_t)t * H, H);
                float u = dot_f32(du, xp + (size_t)t * H, H);
                if (lim > 0) { u = std::min(std::max(u, -lim), lim); g = std::min(g, lim); }
                hp[(size_t)i * I + pos] = silu(g) * u;
            }
        }
    }
}

static void down_task(Layer* L, int task, const std::vector<int>& active, int nb2, const float* hp, float* yp,
                      float* tmp, int K) {
    (void)K;
    const int H = L->H, I = L->I; const int RB = 32;
    const int e = active[task / nb2], rb = task % nb2;
    const int t0 = L->e_start[e], t1 = L->e_start[e + 1], nt = t1 - t0;
    const uint8_t* w = L->w2 + (size_t)e * L->w2_stride;
    const uint8_t* s = L->s2 + (size_t)e * L->s2_stride;
    float* dr = tmp;
    for (int r = rb * RB; r < (rb + 1) * RB; ++r) {
        const uint8_t* wr = w + (size_t)r * (I / 2); const uint8_t* sr = s + (size_t)r * (I / 32);
        if (nt == 1) {
            const int i = L->tok_of_e[t0];
            yp[(size_t)i * H + r] = dot_row(wr, sr, hp + (size_t)i * I, I);
        } else {
            dequant_row(wr, sr, dr, I);
            for (int q = t0; q < t1; ++q) {
                const int i = L->tok_of_e[q];
                yp[(size_t)i * H + r] = dot_f32(dr, hp + (size_t)i * I, I);
            }
        }
    }
}
} // namespace

extern "C" {

void* cpu_moe_create(int E, int H, int I, float swiglu_limit, int threads) {
    Layer* L = new Layer();
    L->E = E; L->H = H; L->I = I; L->lim = swiglu_limit; L->threads = threads;
    L->w13_stride = (size_t)I * H;             // I rows x (H/2 + H/2) bytes
    L->s13_stride = (size_t)I * 2 * (H / 32);
    L->w2_stride = (size_t)H * (I / 2);
    L->s2_stride = (size_t)H * (I / 32);
    L->w13 = (uint8_t*)alloc_big(L->w13_stride * E);
    L->s13 = (uint8_t*)alloc_big(L->s13_stride * E);
    L->w2 = (uint8_t*)alloc_big(L->w2_stride * E);
    L->s2 = (uint8_t*)alloc_big(L->s2_stride * E);
    if (!L->w13 || !L->s13 || !L->w2 || !L->s2) { fprintf(stderr, "cpu_moe: alloc failed\n"); return nullptr; }
    return L;
}

// offs: [E][6] absolute byte offsets in `path` for w1, s1, w3, s3, w2, s2
int cpu_moe_load(void* h, const char* path, const int64_t* offs) {
    Layer* L = (Layer*)h;
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("cpu_moe: open"); return -1; }
    const int H = L->H, I = L->I;
    const size_t rowb = (size_t)H / 2, srow = (size_t)H / 32;
    int fail = 0;
    #pragma omp parallel num_threads(std::min(L->threads, 8))
    {
        std::vector<uint8_t> t1(rowb * I), t3(rowb * I), q1(srow * I), q3(srow * I);
        #pragma omp for schedule(dynamic, 1)
        for (int e = 0; e < L->E; ++e) {
            const int64_t* o = offs + (size_t)e * 6;
            bool ok = pread_all(fd, t1.data(), t1.size(), o[0]) && pread_all(fd, q1.data(), q1.size(), o[1]) &&
                      pread_all(fd, t3.data(), t3.size(), o[2]) && pread_all(fd, q3.data(), q3.size(), o[3]) &&
                      pread_all(fd, L->w2 + (size_t)e * L->w2_stride, L->w2_stride, o[4]) &&
                      pread_all(fd, L->s2 + (size_t)e * L->s2_stride, L->s2_stride, o[5]);
            if (!ok) { fail = 1; continue; }
            uint8_t* w = L->w13 + (size_t)e * L->w13_stride;
            uint8_t* s = L->s13 + (size_t)e * L->s13_stride;
            for (int r = 0; r < I; ++r) {
                memcpy(w + (size_t)r * H, t1.data() + (size_t)r * rowb, rowb);
                memcpy(w + (size_t)r * H + rowb, t3.data() + (size_t)r * rowb, rowb);
                memcpy(s + (size_t)r * 2 * srow, q1.data() + (size_t)r * srow, srow);
                memcpy(s + (size_t)r * 2 * srow + srow, q3.data() + (size_t)r * srow, srow);
            }
        }
    }
    close(fd);
    return fail ? -2 : 0;
}

// x: bf16 [M][H]; ids: int32 [M][K]; wts: fp32 [M][K]; out: bf16 [M][H]
void cpu_moe_forward(void* h, int M, const uint16_t* x, const int32_t* ids, const float* wts, int K,
                     uint16_t* out) {
    Layer* L = (Layer*)h;
    const int H = L->H, I = L->I, E = L->E;
    const float lim = L->lim;
    const size_t MK = (size_t)M * K;
    if (L->xp.size() < (size_t)M * H) L->xp.resize((size_t)M * H);
    if (L->hp.size() < MK * I) L->hp.resize(MK * I);
    if (L->yp.size() < MK * H) L->yp.resize(MK * H);
    L->tok_of_e.resize(MK); L->e_start.assign(E + 1, 0);
    // group (token, slot) pairs by expert
    for (size_t i = 0; i < MK; ++i) { int e = ids[i]; if (e >= 0 && e < E) L->e_start[e + 1]++; }
    for (int e = 0; e < E; ++e) L->e_start[e + 1] += L->e_start[e];
    {
        std::vector<int> fill(L->e_start.begin(), L->e_start.end() - 1);
        for (size_t i = 0; i < MK; ++i) { int e = ids[i]; if (e >= 0 && e < E) L->tok_of_e[fill[e]++] = (int)i; }
    }
    std::vector<int> active; for (int e = 0; e < E; ++e) if (L->e_start[e + 1] > L->e_start[e]) active.push_back(e);
    const int RB = 32; // rows per task
    const int nb13 = I / RB, nb2 = H / RB;
    const int n13 = (int)active.size() * nb13, n2 = (int)active.size() * nb2;
    float* xp = L->xp.data(); float* hp = L->hp.data(); float* yp = L->yp.data();
    static const bool trace = getenv("CPU_MOE_TRACE") != nullptr;
    double tt[5] = {0, 0, 0, 0, 0};
    auto now = []() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    if (trace) tt[0] = now();

    #pragma omp parallel num_threads(L->threads)
    {
        std::vector<float> tmp((size_t)H * 2);
        // 1. permute inputs
        #pragma omp for schedule(static)
        for (int t = 0; t < M; ++t) {
            float* row = tmp.data();
            for (int j = 0; j < H; ++j) row[j] = bf16_to_f32(x[(size_t)t * H + j]);
            permute_row(row, xp + (size_t)t * H, H);
        }
        if (trace) { 
            #pragma omp master
            tt[1] = now();
        }
        // 2. gate/up + activation -> hp (permuted for the down projection)
        // decode-sized batches: static (each thread streams contiguous rows);
        // prefill: dynamic (per-expert token counts differ).
        #pragma omp for schedule(dynamic, 2) nowait
        for (int task = (M <= 16 ? n13 : 0); task < n13; ++task) gateup_task(L, task, active, nb13, xp, hp, tmp.data(), K, lim);
        #pragma omp for schedule(static)
        for (int task = 0; task < (M <= 16 ? n13 : 0); ++task) gateup_task(L, task, active, nb13, xp, hp, tmp.data(), K, lim);

        if (trace) {
            #pragma omp master
            tt[2] = now();
        }
        // 3. down projection -> yp [i][H]
        #pragma omp for schedule(dynamic, 2) nowait
        for (int task = (M <= 16 ? n2 : 0); task < n2; ++task) down_task(L, task, active, nb2, hp, yp, tmp.data(), K);
        #pragma omp for schedule(static)
        for (int task = 0; task < (M <= 16 ? n2 : 0); ++task) down_task(L, task, active, nb2, hp, yp, tmp.data(), K);

        if (trace) {
            #pragma omp master
            tt[3] = now();
        }
        // 4. weighted combine -> bf16
        #pragma omp for schedule(static)
        for (int t = 0; t < M; ++t) {
            float* acc = tmp.data();
            for (int j = 0; j < H; ++j) acc[j] = 0.0f;
            for (int k = 0; k < K; ++k) {
                const size_t i = (size_t)t * K + k; const int e = ids[i];
                if (e < 0 || e >= E) continue;
                const float wgt = wts[i]; const float* y = yp + i * H;
                for (int j = 0; j < H; ++j) acc[j] += wgt * y[j];
            }
            for (int j = 0; j < H; ++j) out[(size_t)t * H + j] = f32_to_bf16(acc[j]);
        }
    }
    if (trace) {
        tt[4] = now();
        fprintf(stderr, "cpu_moe M=%d: permute %.3f  gateup %.3f  down %.3f  combine %.3f  total %.3f ms\n",
                M, tt[1] - tt[0], tt[2] - tt[1], tt[3] - tt[2], tt[4] - tt[3], tt[4] - tt[0]);
    }
}

void cpu_moe_trace_flush(void*) {}

void cpu_moe_free(void* h) {
    Layer* L = (Layer*)h; if (!L) return;
    free(L->w13); free(L->s13); free(L->w2); free(L->s2); delete L;
}

// cudaLaunchHostFunc-compatible entry: arg points to a Work struct
struct Work { void* layer; int M; int K; const uint16_t* x; const int32_t* ids; const float* wts; uint16_t* out; };
void cpu_moe_host_fn(void* arg) {
    Work* w = (Work*)arg;
    cpu_moe_forward(w->layer, w->M, w->x, w->ids, w->wts, w->K, w->out);
}

} // extern "C"
