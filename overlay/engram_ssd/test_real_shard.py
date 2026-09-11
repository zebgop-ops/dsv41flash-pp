"""Validate librow_store against the real Engram shard (CPU only, numpy reference).
usage: test_real_shard.py <model_dir> [layer_id] [n_rows]"""
import ctypes as C, json, os, struct, sys, time, numpy as np
model_dir = sys.argv[1]; layer = int(sys.argv[2]) if len(sys.argv) > 2 else 1
n = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000
index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
wkey, skey = f"layers.{layer}.engram.embed.weight", f"layers.{layer}.engram.embed.scale"
path = os.path.join(model_dir, index[wkey])
with open(path, "rb") as f:
    hl = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(hl))
w, s = hdr[wkey], hdr[skey]; rows = w["shape"][0]; base = 8 + hl
woff, soff = base + w["data_offsets"][0], base + s["data_offsets"][0]
print(f"shard {os.path.basename(path)} rows={rows} w={w['dtype']}{w['shape']} s={s['dtype']}{s['shape']} woff={woff} soff={soff}")
lib = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "librow_store.so"))
P, U = C.c_void_p, C.c_uint64
lib.row_store_open.argtypes = [C.c_char_p, U, U, U, C.c_int]; lib.row_store_open.restype = P
lib.row_store_lookup_sync.argtypes = [P, P, P, P, U]; lib.row_store_close.argtypes = [P]
st = lib.row_store_open(path.encode(), rows, woff, soff, 16); assert st
rng = np.random.default_rng(1)
ids = rng.integers(0, rows, size=n, dtype=np.int64)
ow = np.zeros((n, 256), np.uint8); osc = np.zeros((n, 8), np.uint8)
t = time.perf_counter(); lib.row_store_lookup_sync(st, ids.ctypes.data, ow.ctypes.data, osc.ctypes.data, n); cold = time.perf_counter() - t
t = time.perf_counter(); lib.row_store_lookup_sync(st, ids.ctypes.data, ow.ctypes.data, osc.ctypes.data, n); warm = time.perf_counter() - t
mm = np.memmap(path, dtype=np.uint8, mode="r")
chk = rng.choice(n, size=min(n, 2000), replace=False)
for i in chk:
    r = int(ids[i])
    assert ow[i].tobytes() == mm[woff + r*256: woff + (r+1)*256].tobytes(), f"weight mismatch row {r}"
    assert osc[i].tobytes() == mm[soff + r*8: soff + (r+1)*8].tobytes(), f"scale mismatch row {r}"
nz = (ow != 0).any(1).mean()
print(f"n={n}: cold {cold*1e3:.0f} ms ({n/cold:.0f} rows/s), warm {warm*1e3:.0f} ms ({n/warm:.0f} rows/s); {len(chk)} rows verified; nonzero rows {nz:.3f}")
lib.row_store_close(st); print("PASS")
