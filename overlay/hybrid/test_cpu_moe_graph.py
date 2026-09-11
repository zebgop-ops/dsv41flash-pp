"""NativeCpuMoe under CUDA graph capture/replay must equal its eager result for changing inputs.
usage: test_cpu_moe_graph.py <model_dir> <layer>"""
import sys, torch
sys.path.insert(0, "/opt/dsv41")
from hybrid.cpu_moe import NativeCpuMoe
model_dir, layer = sys.argv[1], int(sys.argv[2])
E, K, H, I, LIM = 384, 6, 5120, 2304, 10.0
dev = torch.device("cuda", 0)
moe = NativeCpuMoe(model_dir, layer, E, H, I, LIM, threads=16, max_tokens=64); moe.load(); moe.warmup(dev, K)
g = torch.Generator().manual_seed(1)
def inputs(M):
    x = (torch.randn(M, H, generator=g) / 10).to(torch.bfloat16).to(dev)
    ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(M)]).to(torch.int32).to(dev)
    w = (torch.rand(M, K, generator=g) / K).to(dev)
    return x, ids, w
for M in (1, 4, 8):
    sx, sids, sw = inputs(M)                       # static buffers
    # warm up the stream/allocator like vLLM does, then capture
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2): out = moe.forward(sx, sids, sw)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        gout = moe.forward(sx, sids, sw)
    torch.cuda.synchronize()
    for trial in range(3):
        x, ids, w = inputs(M)
        sx.copy_(x); sids.copy_(ids); sw.copy_(w); torch.cuda.synchronize()
        graph.replay(); torch.cuda.synchronize()
        got = gout.clone()
        ref = moe.forward_cpu(x.cpu(), ids.cpu(), w.cpu()).to(dev)
        err = (got.float() - ref.float()).abs().max().item()
        print(f"M={M} trial {trial}: replay vs eager-cpu max abs {err:.3g} (ref absmax {ref.float().abs().max().item():.3g}) {'OK' if err == 0 else 'MISMATCH'}", flush=True)
print("DONE")
