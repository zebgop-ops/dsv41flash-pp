// Per-instruction throughput on this GPU: which arithmetic pipes are throttled on the CMP 170HX.
// Each kernel runs a dependent-free chain of 8 independent accumulators per thread, many blocks.
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>
#define ITERS 4096
#define CHK(x) do { cudaError_t e = (x); if (e) { printf("CUDA %s\n", cudaGetErrorString(e)); return 1; } } while (0)

__global__ void k_ffma(float* out, float a, float b) {
  float x[8]; for (int i = 0; i < 8; i++) x[i] = threadIdx.x + i;
  for (int it = 0; it < ITERS; it++) {
#pragma unroll
    for (int i = 0; i < 8; i++) x[i] = fmaf(x[i], a, b);
  }
  float s = 0; for (int i = 0; i < 8; i++) s += x[i]; if (s == 1.234f) out[0] = s;
}
__global__ void k_fmul_fadd(float* out, float a, float b) {
  float x[8]; for (int i = 0; i < 8; i++) x[i] = threadIdx.x + i;
  for (int it = 0; it < ITERS; it++) {
#pragma unroll
    for (int i = 0; i < 8; i++) x[i] = __fadd_rn(__fmul_rn(x[i], a), b);   // 2 ops
  }
  float s = 0; for (int i = 0; i < 8; i++) s += x[i]; if (s == 1.234f) out[0] = s;
}
__global__ void k_fadd(float* out, float a, float b) {
  float x[8]; for (int i = 0; i < 8; i++) x[i] = threadIdx.x + i;
  for (int it = 0; it < ITERS; it++) {
#pragma unroll
    for (int i = 0; i < 8; i++) x[i] = __fadd_rn(x[i], b);
  }
  float s = 0; for (int i = 0; i < 8; i++) s += x[i]; if (s == 1.234f) out[0] = s;
}
__global__ void k_imad(uint32_t* out, uint32_t a, uint32_t b) {
  uint32_t x[8]; for (int i = 0; i < 8; i++) x[i] = threadIdx.x + i;
  for (int it = 0; it < ITERS; it++) {
#pragma unroll
    for (int i = 0; i < 8; i++) x[i] = x[i] * a + b;
  }
  uint32_t s = 0; for (int i = 0; i < 8; i++) s ^= x[i]; if (s == 12345u) out[0] = s;
}
__global__ void k_lop(uint32_t* out, uint32_t a, uint32_t b) {
  uint32_t x[8]; for (int i = 0; i < 8; i++) x[i] = threadIdx.x + i;
  for (int it = 0; it < ITERS; it++) {
#pragma unroll
    for (int i = 0; i < 8; i++) x[i] = (x[i] ^ a) + b;   // LOP3 + IADD3
  }
  uint32_t s = 0; for (int i = 0; i < 8; i++) s ^= x[i]; if (s == 12345u) out[0] = s;
}
__global__ void k_hfma2(__half2* out, __half2 a, __half2 b) {
  __half2 x[8]; for (int i = 0; i < 8; i++) x[i] = __floats2half2_rn(threadIdx.x * 1e-3f + i, 1.f);
  for (int it = 0; it < ITERS; it++) {
#pragma unroll
    for (int i = 0; i < 8; i++) x[i] = __hfma2(x[i], a, b);
  }
  __half2 s = x[0]; for (int i = 1; i < 8; i++) s = __hadd2(s, x[i]); if (__low2float(s) == 1.234f) out[0] = s;
}
__global__ void k_hadd2(__half2* out, __half2 a, __half2 b) {
  __half2 x[8]; for (int i = 0; i < 8; i++) x[i] = __floats2half2_rn(threadIdx.x * 1e-3f + i, 1.f);
  for (int it = 0; it < ITERS; it++) {
#pragma unroll
    for (int i = 0; i < 8; i++) x[i] = __hadd2(x[i], b);
  }
  __half2 s = x[0]; for (int i = 1; i < 8; i++) s = __hadd2(s, x[i]); if (__low2float(s) == 1.234f) out[0] = s;
}
__global__ void k_dp4a(uint32_t* out, uint32_t a, uint32_t b) {
  uint32_t x[8]; for (int i = 0; i < 8; i++) x[i] = threadIdx.x + i;
  for (int it = 0; it < ITERS; it++) {
#pragma unroll
    for (int i = 0; i < 8; i++) x[i] = __dp4a(x[i], a, b);
  }
  uint32_t s = 0; for (int i = 0; i < 8; i++) s ^= x[i]; if (s == 12345u) out[0] = s;
}
__global__ void k_dfma(double* out, double a, double b) {
  double x[8]; for (int i = 0; i < 8; i++) x[i] = threadIdx.x + i;
  for (int it = 0; it < ITERS; it++) {
#pragma unroll
    for (int i = 0; i < 8; i++) x[i] = fma(x[i], a, b);
  }
  double s = 0; for (int i = 0; i < 8; i++) s += x[i]; if (s == 1.234) out[0] = s;
}

template <typename F> float run(F f, const char* name, double ops_per_iter_elem) {
  int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  dim3 grid(sms * 8), block(256);
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  f(grid, block); cudaDeviceSynchronize();
  cudaEventRecord(s); for (int r = 0; r < 5; r++) f(grid, block); cudaEventRecord(e); cudaEventSynchronize(e);
  float ms; cudaEventElapsedTime(&ms, s, e); ms /= 5;
  double ops = (double) grid.x * block.x * ITERS * 8 * ops_per_iter_elem;
  printf("%-12s %8.3f ms   %8.1f Gop/s total   %6.1f op/clk/SM (at 1.41 GHz)\n", name, ms, ops / ms / 1e6,
         ops / (ms * 1e-3) / sms / 1.41e9);
  return ms;
}
int main() {
  float* f; uint32_t* u; __half2* h; double* d;
  cudaMalloc(&f, 64); cudaMalloc(&u, 64); cudaMalloc(&h, 64); cudaMalloc(&d, 64);
  __half2 ha = __floats2half2_rn(0.999f, 0.999f), hb = __floats2half2_rn(1e-3f, 1e-3f);
  run([&](dim3 g, dim3 b) { k_ffma<<<g, b>>>(f, 0.999f, 1e-3f); }, "FFMA", 1);
  run([&](dim3 g, dim3 b) { k_fmul_fadd<<<g, b>>>(f, 0.999f, 1e-3f); }, "FMUL+FADD", 2);
  run([&](dim3 g, dim3 b) { k_fadd<<<g, b>>>(f, 0.999f, 1e-3f); }, "FADD", 1);
  run([&](dim3 g, dim3 b) { k_imad<<<g, b>>>(u, 0x83DCD12Du, 7u); }, "IMAD", 1);
  run([&](dim3 g, dim3 b) { k_lop<<<g, b>>>(u, 0x83DCD12Du, 7u); }, "LOP3+IADD3", 2);
  run([&](dim3 g, dim3 b) { k_hfma2<<<g, b>>>(h, ha, hb); }, "HFMA2(x2)", 2);
  run([&](dim3 g, dim3 b) { k_hadd2<<<g, b>>>(h, ha, hb); }, "HADD2(x2)", 2);
  run([&](dim3 g, dim3 b) { k_dp4a<<<g, b>>>(u, 0x01010101u, 7u); }, "DP4A", 1);
  run([&](dim3 g, dim3 b) { k_dfma<<<g, b>>>(d, 0.999, 1e-3); }, "DFMA", 1);
  return 0;
}
