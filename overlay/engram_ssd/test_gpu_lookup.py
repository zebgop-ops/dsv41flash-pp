"""GPU check of SsdEngramTable against the CPU reference on the real shard.
Run inside the image with a GPU: python3 test_gpu_lookup.py <model_dir> [layer] [tokens]"""
import os, sys, time, torch
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from engram_ssd.engram_ssd import SsdEngramTable, find_engram_shard, reference_lookup_cpu
model_dir = sys.argv[1]; layer = int(sys.argv[2]) if len(sys.argv) > 2 else 1
T = int(sys.argv[3]) if len(sys.argv) > 3 else 512
heads = 24
_, rows, _, _ = find_engram_shard(model_dir, layer)
dev = torch.device("cuda", 0)
tab = SsdEngramTable(model_dir, layer, 0, rows, 0, heads, heads, max_tokens=max(T, 64))
g = torch.Generator().manual_seed(0)
ids = torch.randint(0, rows, (T, heads), generator=g, dtype=torch.int32).to(dev)
out = torch.empty((T, heads, 256), dtype=torch.bfloat16, device=dev)
for i in range(2):
    torch.cuda.synchronize(); t = time.perf_counter()
    tab.lookup(ids, out); torch.cuda.synchronize()
    print(f"lookup {i}: {T} tokens x {heads} heads in {(time.perf_counter()-t)*1e3:.1f} ms")
ref = reference_lookup_cpu(model_dir, layer, ids.reshape(-1).cpu()[:4096])
got = out.view(-1, 256)[:4096].cpu()
assert torch.equal(got, ref), (got[:2], ref[:2])
print("exact match on 4096 rows; stats", tab.stats(), "PASS")
# graph capture smoke: lookup inside a CUDA graph must replay correctly
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    tab.lookup(ids, out)
torch.cuda.current_stream().wait_stream(s)
gph = torch.cuda.CUDAGraph()
with torch.cuda.graph(gph, stream=s):
    tab.lookup(ids, out)
ids2 = torch.randint(0, rows, (T, heads), generator=g, dtype=torch.int32).to(dev)
ids.copy_(ids2); out.zero_(); gph.replay(); torch.cuda.synchronize()
ref2 = reference_lookup_cpu(model_dir, layer, ids2.reshape(-1).cpu()[:2048])
assert torch.equal(out.view(-1, 256)[:2048].cpu(), ref2)
print("graph replay exact; PASS")
