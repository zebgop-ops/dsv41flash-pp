#!/usr/bin/env python3
"""One DeepSeek-V4.1 EXL3 MoE layer in isolation: load layer L's 384 experts from the pack, build the fused
kernel state exactly as the server does, and time exl3_moe for T tokens (CUDA-graph replay, so launch
overhead is excluded). usage: moe_layer.py <snapshot> <layer> [T list] [num_active override]"""
import json, os, sys, time, torch
sys.path.insert(0, "/opt/dsv41/exl3")
os.environ.setdefault("EXL3_TEMP_ROWS_FUSED", "256")
import dsv41_exl3 as m
from safetensors import safe_open

snap, L = sys.argv[1].rstrip("/") + "/", int(sys.argv[2])
Ts = [int(t) for t in (sys.argv[3] if len(sys.argv) > 3 else "1,2,4,8,16").split(",")]
dev = torch.device("cuda")
idx = json.load(open(snap + "model.safetensors.index.json"))["weight_map"]
E, TOPK, H, I = 384, 6, 5120, 2304
files = {}
def get(name):
    f = idx[name]
    if f not in files: files[f] = safe_open(snap + f, "pt", device="cuda")
    return files[f].get_tensor(name)
class Layer(torch.nn.Module): pass
layer = Layer()
inners = []
t0 = time.time()
for e in range(E):
    pk = {}
    for w in ("w1", "w3", "w2"):
        pk[w] = m.make_linear_exl3(*(get(f"layers.{L}.ffn.experts.{e}.{w}.{s}") for s in ("trellis", "suh", "svh", "mul1")))
    inners.append({"gate": pk["w1"], "up": pk["w3"], "down": pk["w2"]})
layer.w13_trellis = inners[0]["gate"].trellis
layer._exl3_hidden_size, layer._exl3_intermediate_local = H, I
m.build_exl3_fused_state(layer, inners)
layer._exl3_inners = inners
print(f"layer {L}: K={inners[0]['gate'].K} loaded in {time.time()-t0:.0f}s, concurrency {layer._exl3_fused_concurrency}, "
      f"GPU mem {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)
g = torch.Generator(device="cpu").manual_seed(0)
for T in Ts:
    x = (torch.randn(T, H, generator=g) * 0.05).to(dev, torch.bfloat16)
    ids = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(T)]).to(dev)
    w = torch.rand(T, TOPK, generator=g).to(dev)
    w = w / w.sum(1, keepdim=True)
    for _ in range(3): m.apply_exl3_experts(x, ids, w, layer, limit=10.0)
    if os.environ.get("CMP_REF") == "1":   # compare against the fused kernel and the per-expert reference loop
        os.environ["EXL3_MOE_COOP"] = "0"; y_fused = m.apply_exl3_experts(x, ids, w, layer, limit=10.0).float()
        os.environ["EXL3_MOE_COOP"] = "1"; y_coop = m.apply_exl3_experts(x, ids, w, layer, limit=10.0).float()
        y_ref = m.apply_exl3_python_loop(x.reshape(T, H), ids, w, inners, 10.0).float()
        rel = lambda a, b: ((a - b).norm() / b.norm()).item()
        print(f"T={T}: coop vs loop {rel(y_coop, y_ref):.2e}, fused vs loop {rel(y_fused, y_ref):.2e}, coop vs fused {rel(y_coop, y_fused):.2e}", flush=True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        m.apply_exl3_experts(x, ids, w, layer, limit=10.0)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=s):
            y = m.apply_exl3_experts(x, ids, w, layer, limit=10.0)
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(50): graph.replay()
    en.record(); torch.cuda.synchronize()
    us = st.elapsed_time(en) / 50 * 1000
    nexp = int(torch.unique(ids).numel())
    gb = nexp * 3 * H * I * inners[0]["gate"].K / 8 / 1e9
    print(f"T={T:3d}: {nexp:3d} distinct experts, {us:7.1f} us per layer (whole MoE incl. routing glue), "
          f"{gb/us*1e6:6.1f} GB/s of trellis, {nexp*3*H*I/us/1e3:6.1f} Gweights/s", flush=True)
