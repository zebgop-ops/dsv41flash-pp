"""libcpu_moe vs torch reference on a real V4.1 layer (CPU only).
usage: test_cpu_moe.py <model_dir> <layer> [threads]"""
import sys, time, torch, torch.nn.functional as F
sys.path.insert(0, "/opt/dsv41")
from hybrid.cpu_moe import NativeCpuMoe, expert_offsets
from safetensors import safe_open
model_dir, layer = sys.argv[1], int(sys.argv[2]); threads = int(sys.argv[3]) if len(sys.argv) > 3 else 16
E, K, H, I, LIM = 384, 6, 5120, 2304, 10.0
moe = NativeCpuMoe(model_dir, layer, E, H, I, LIM, threads=threads)
t = time.perf_counter(); moe.load(); print(f"loaded layer {layer} in {time.perf_counter()-t:.1f}s", flush=True)
path, _ = expert_offsets(model_dir, layer, E)
E2M1 = torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6])
def deq(name):
    with safe_open(path, "pt") as f:
        p = f.get_tensor(name + ".weight"); s = f.get_tensor(name + ".scale")
    p = p.view(torch.uint8); nib = torch.stack([p & 0xF, p >> 4], -1).reshape(p.shape[0], -1).long()
    sc = (s.view(torch.uint8).to(torch.int32) << 23).view(torch.float32).repeat_interleave(32, 1)
    return E2M1[nib] * sc
def ref(x, ids, wts):
    x = x.float(); out = torch.zeros(x.shape[0], H)
    for e in sorted(set(ids.flatten().tolist())):
        w1, w3, w2 = (deq(f"layers.{layer}.ffn.experts.{e}.{n}") for n in ("w1", "w3", "w2"))
        tsel, k = torch.where(ids == e)
        g = x[tsel] @ w1.t(); u = x[tsel] @ w3.t(); u = u.clamp(-LIM, LIM); g = g.clamp(max=LIM)
        out[tsel] += wts[tsel, k].unsqueeze(-1) * ((F.silu(g) * u) @ w2.t())
    return out
g = torch.Generator().manual_seed(0)
for M in (1, 4, 40):
    x = (torch.randn(M, H, generator=g) / 10).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(M)]).to(torch.int32)
    wts = (torch.rand(M, K, generator=g) / K)
    y = moe.forward_cpu(x, ids, wts).float(); r = ref(x, ids, wts)
    err = (y - r).abs(); cos = F.cosine_similarity(y.flatten(), r.flatten(), dim=0).item()
    print(f"M={M}: max abs {err.max():.4f} mean abs {err.mean():.6f} ref mean|.| {r.abs().mean():.5f} cos {cos:.6f}", flush=True)
    assert cos > 0.9999 and err.max() < 0.05 * max(1.0, r.abs().max().item())
for M in (1, 2, 4, 8, 64, 512, 2048):
    x = (torch.randn(M, H, generator=g) / 10).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(M)]).to(torch.int32)
    wts = (torch.rand(M, K, generator=g) / K)
    for _ in range(2): moe.forward_cpu(x, ids, wts)
    n = 10 if M <= 8 else 2; t = time.perf_counter()
    for _ in range(n): moe.forward_cpu(x, ids, wts)
    dt = (time.perf_counter() - t) / n
    print(f"M={M:5d}: {dt*1e3:8.2f} ms/step  ({K*3*H*I*0.5*M/dt/1e9 if M<=8 else 0:5.1f} GB/s of FP4)", flush=True)
print("PASS")
