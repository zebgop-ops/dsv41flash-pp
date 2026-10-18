#!/usr/bin/env python3
"""Dense EXL3 linears of one layer: exllamav3 kernel vs the same matrices re-stored as Marlin W8A16, CUDA-graph
replay, T tokens, interleaved A/B so clock drift hits both. usage: dense.py <snapshot> <layer> [T list]"""
import json, os, sys, torch
sys.path.insert(0, "/opt/dsv41/exl3")
os.environ["EXL3_DENSE_W8"] = "1"
import dsv41_exl3 as m
from safetensors import safe_open
snap, L = sys.argv[1].rstrip("/") + "/", int(sys.argv[2])
Ts = [int(t) for t in (sys.argv[3] if len(sys.argv) > 3 else "1,4").split(",")]
idx = json.load(open(snap + "model.safetensors.index.json"))["weight_map"]
files = {}
def get(n):
    f = idx[n]
    if f not in files: files[f] = safe_open(snap + f, "pt", device="cuda")
    return files[f].get_tensor(n)
def lin(name): return m.make_linear_exl3(*(get(f"{name}.{s}") for s in ("trellis", "suh", "svh", "mul1")))
groups = {
    "wq_a+wkv": [f"layers.{L}.attn.wq_a", f"layers.{L}.attn.wkv"],
    "wq_b": [f"layers.{L}.attn.wq_b"],
    "wo_b": [f"layers.{L}.attn.wo_b"],
    "shared w1+w3": [f"layers.{L}.ffn.shared_experts.w1", f"layers.{L}.ffn.shared_experts.w3"],
    "shared w2": [f"layers.{L}.ffn.shared_experts.w2"],
}
class Lay: pass
def timeit(fn, n=200):
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s): fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n): g.replay()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / n * 1000
tot = {}
for name, parts in groups.items():
    inners = [lin(p) for p in parts]
    K = inners[0].in_features; N = sum(h.out_features for h in inners)
    lay = Lay(); w = torch.cat([h.get_weight_tensor() for h in inners], dim=1)
    lay._w8_q, lay._w8_s, err = m.to_marlin_w8(w); lay._w8_k, lay._w8_n = K, N
    lay._w8_empty = torch.empty(0, dtype=torch.int32, device="cuda")
    for T in Ts:
        x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
        def exl3():   # what Exl3LinearMethod.apply does for a merged / single EXL3 linear
            xh = x.contiguous().half()
            out = torch.cat([h.forward(xh, {}, out_dtype=torch.float32) for h in inners], dim=-1) if len(inners) > 1 else inners[0].forward(xh, {}, out_dtype=torch.float32)
            return out.to(torch.bfloat16)
        def w8(): return m.apply_marlin_w8(lay, x)
        r = [(timeit(exl3), timeit(w8)) for _ in range(3)]
        te, tw = min(a for a, _ in r), min(b for _, b in r)
        y1, y2 = exl3().float(), w8().float()
        tot.setdefault(T, [0, 0]); tot[T][0] += te; tot[T][1] += tw
        print(f"{name:13s} K={K:5d} N={N:5d} T={T}: EXL3 {te:6.1f} us   Marlin W8 {tw:6.1f} us   ({te/tw:4.2f}x)   out rel diff {((y1-y2).norm()/y1.norm()).item():.1e}, w8 re-round {err:.1e}", flush=True)
for T, (a, b) in tot.items():
    print(f"layer total T={T}: EXL3 {a:.0f} us, Marlin W8 {b:.0f} us ({a/b:.2f}x)")
