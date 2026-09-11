"""CPU-only check of librow_store.so against a synthetic shard-like file."""
import ctypes as C, os, time, numpy as np, tempfile, sys
lib = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "librow_store.so"))
P, U = C.c_void_p, C.c_uint64
lib.row_store_open.argtypes = [C.c_char_p, U, U, U, C.c_int]; lib.row_store_open.restype = P
lib.row_store_lookup_sync.argtypes = [P, P, P, P, U]
lib.row_store_range.argtypes = [P, U, U]
lib.row_store_close.argtypes = [P]
rows = 200_000
rng = np.random.default_rng(0)
w = rng.integers(0, 256, size=(rows, 256), dtype=np.uint8)
s = rng.integers(0, 256, size=(rows, 8), dtype=np.uint8)
hdr = b"X" * 1234
with tempfile.NamedTemporaryFile(delete=False) as f:
    f.write(hdr); f.write(w.tobytes()); f.write(s.tobytes()); path = f.name
woff, soff = len(hdr), len(hdr) + rows * 256
st = lib.row_store_open(path.encode(), rows, woff, soff, 16)
assert st, "open failed"
for n in (1, 24, 4096, 24 * 4096):
    ids = rng.integers(-1, rows, size=n, dtype=np.int64)
    ow = np.zeros((n, 256), np.uint8); osc = np.zeros((n, 8), np.uint8)
    t = time.perf_counter()
    lib.row_store_lookup_sync(st, ids.ctypes.data, ow.ctypes.data, osc.ctypes.data, n)
    dt = time.perf_counter() - t
    valid = ids >= 0
    assert np.array_equal(ow[valid], w[ids[valid]]) and np.array_equal(osc[valid], s[ids[valid]])
    assert not ow[~valid].any() and not osc[~valid].any()
    print(f"n={n:7d} ok  {dt*1e3:8.2f} ms  {n/dt/1e6:6.2f} M rows/s")
lib.row_store_range(st, 1000, 2000)
ids = np.array([999, 1000, 1999, 2000], np.int64); ow = np.zeros((4, 256), np.uint8); osc = np.zeros((4, 8), np.uint8)
lib.row_store_lookup_sync(st, ids.ctypes.data, ow.ctypes.data, osc.ctypes.data, 4)
assert not ow[0].any() and ow[1].tobytes() == w[1000].tobytes() and ow[2].tobytes() == w[1999].tobytes() and not ow[3].any()
print("range ownership ok")
lib.row_store_close(st); os.unlink(path); print("PASS")
