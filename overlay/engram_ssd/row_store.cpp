// Engram row store: exact FP8 row + UE8M0 scale retrieval straight from the
// original safetensors shard on the NVMe, served through the kernel page cache.
//
// Design adapted from 0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000 adapter/row_store.cpp
// (MIT, Copyright (c) 2026 0xSero); rewritten here with a thread pool so that a
// prefill chunk (thousands of tokens x 24 rows) does not serialize on one pread.
// No CUDA calls in the callback: it is launched via cudaLaunchHostFunc and is
// therefore legal inside CUDA graph capture/replay.
//
// Row layout in the checkpoint (per Engram table):
//   weight: [rows, 256] F8_E4M3   at byte offset woff (contiguous 256 B rows)
//   scale : [rows,   8] F8_E8M0   at byte offset soff (contiguous   8 B rows)
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <mutex>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

struct Pool {
  std::vector<std::thread> threads;
  std::mutex m;
  std::condition_variable cv, done_cv;
  std::function<void(int)> job;   // job(index)
  uint64_t njobs = 0, next = 0, finished = 0;
  bool stop = false;
  explicit Pool(int n) {
    for (int i = 0; i < n; ++i) threads.emplace_back([this] { loop(); });
  }
  ~Pool() {
    { std::lock_guard<std::mutex> g(m); stop = true; }
    cv.notify_all();
    for (auto &t : threads) t.join();
  }
  void loop() {
    for (;;) {
      uint64_t idx;
      {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return stop || next < njobs; });
        if (stop) return;
        idx = next++;
      }
      job(idx);
      {
        std::lock_guard<std::mutex> g(m);
        if (++finished == njobs) done_cv.notify_all();
      }
    }
  }
  void run(uint64_t n, std::function<void(int)> f) {
    {
      std::lock_guard<std::mutex> g(m);
      job = std::move(f); njobs = n; next = 0; finished = 0;
    }
    cv.notify_all();
    std::unique_lock<std::mutex> lk(m);
    done_cv.wait(lk, [&] { return finished == njobs; });
  }
};

struct Store {
  int fd = -1;
  uint64_t rows = 0, woff = 0, soff = 0, row_lo = 0, row_hi = 0;
  Pool *pool = nullptr;
  std::atomic<uint64_t> lookups{0}, batches{0};
};

struct Work {
  Store *store;
  const int64_t *ids;   // [count] host (pinned) row ids, -1 = skip
  uint8_t *weights;     // [count, 256] host (pinned)
  uint8_t *scales;      // [count, 8]   host (pinned)
  uint64_t count;
};

void fail(const char *reason) {
  std::fprintf(stderr, "engram row_store: %s (errno=%d %s)\n", reason, errno, std::strerror(errno));
  std::abort();  // never let a generation continue with missing/stale rows
}

void pread_all(int fd, void *out, size_t len, uint64_t off) {
  uint8_t *p = static_cast<uint8_t *>(out);
  while (len) {
    ssize_t got = pread(fd, p, len, off);
    if (got < 0) { if (errno == EINTR) continue; fail("pread failed"); }
    if (got == 0) fail("short read (EOF)");
    p += got; len -= got; off += got;
  }
}

void fetch_one(Store *s, int64_t id, uint8_t *w, uint8_t *sc) {
  if (id < 0 || uint64_t(id) < s->row_lo || uint64_t(id) >= s->row_hi) {
    std::memset(w, 0, 256); std::memset(sc, 0, 8); return;
  }
  if (uint64_t(id) >= s->rows) fail("row id out of bounds");
  pread_all(s->fd, w, 256, s->woff + uint64_t(id) * 256);
  pread_all(s->fd, sc, 8, s->soff + uint64_t(id) * 8);
}

}  // namespace

extern "C" Store *row_store_open(const char *path, uint64_t rows, uint64_t woff,
                                 uint64_t soff, int nthreads) {
  auto *s = new Store;
  s->fd = open(path, O_RDONLY | O_CLOEXEC);
  if (s->fd < 0) { delete s; return nullptr; }
  struct stat st;
  if (fstat(s->fd, &st) || woff + rows * 256 > uint64_t(st.st_size) ||
      soff + rows * 8 > uint64_t(st.st_size)) { close(s->fd); delete s; return nullptr; }
  s->rows = rows; s->woff = woff; s->soff = soff; s->row_lo = 0; s->row_hi = rows;
  if (nthreads < 1) nthreads = 1;
  s->pool = new Pool(nthreads);
  return s;
}

extern "C" void row_store_range(Store *s, uint64_t lo, uint64_t hi) {
  if (lo > hi || hi > s->rows) fail("invalid row ownership range");
  s->row_lo = lo; s->row_hi = hi;
}

// Host callback (cudaLaunchHostFunc signature: void(*)(void*)).
extern "C" void row_store_lookup(void *opaque) {
  auto *work = static_cast<Work *>(opaque);
  Store *s = work->store;
  const uint64_t n = work->count;
  s->lookups.fetch_add(n, std::memory_order_relaxed);
  s->batches.fetch_add(1, std::memory_order_relaxed);
  if (n == 0) return;
  const uint64_t chunk = 64;  // rows per job
  const uint64_t njobs = (n + chunk - 1) / chunk;
  if (njobs == 1) {
    for (uint64_t i = 0; i < n; ++i)
      fetch_one(s, work->ids[i], work->weights + i * 256, work->scales + i * 8);
    return;
  }
  s->pool->run(njobs, [&](int j) {
    const uint64_t lo = uint64_t(j) * chunk, hi = std::min<uint64_t>(n, lo + chunk);
    for (uint64_t i = lo; i < hi; ++i)
      fetch_one(s, work->ids[i], work->weights + i * 256, work->scales + i * 8);
  });
}

// Synchronous variant for tests / warmup (same semantics).
extern "C" void row_store_lookup_sync(Store *s, const int64_t *ids, uint8_t *weights,
                                      uint8_t *scales, uint64_t count) {
  Work w{s, ids, weights, scales, count};
  row_store_lookup(&w);
}

extern "C" void row_store_stats(Store *s, uint64_t *out) {
  out[0] = s->lookups.load(); out[1] = s->batches.load();
}

extern "C" void row_store_close(Store *s) {
  if (!s) return;
  delete s->pool;
  close(s->fd);
  delete s;
}
