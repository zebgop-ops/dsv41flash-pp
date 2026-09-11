"""Interleave Engram-SSD lookups (host callback + thread pool) with kt-kernel CPU expert
forwards (host callbacks) on one stream, like rank 2 does: usage <model_dir>"""
import os, sys, time, torch
sys.path.insert(0, "/opt/dsv41"); sys.path.insert(0, "/opt/dsv41/kt-site")
from engram_ssd.engram_ssd import SsdEngramTable, find_engram_shard
from kt_kernel import KTMoEWrapper
model_dir = sys.argv[1]
E, K, H, I = 384, 6, 5120, 2304
dev = torch.device("cuda", 0)
_, rows, _, _ = find_engram_shard(model_dir, 14)
tab = SsdEngramTable(model_dir, 14, 0, rows, 0, 24, 24, max_tokens=2048)
wr = []
for L in (21, 22, 23):
    w = KTMoEWrapper(layer_idx=L, num_experts=E, num_experts_per_tok=K, hidden_size=H, moe_intermediate_size=I,
                     num_gpu_experts=0, gpu_experts_mask=None, cpuinfer_threads=16, threadpool_count=1,
                     weight_path=model_dir, chunked_prefill_size=2048, method="MXFP4", cpu_save=False,
                     max_deferred_experts_per_token=0, swiglu_limit=10.0)
    w.load_weights(torch.arange(E, dtype=torch.int64))
    for a in ("gate_weights","up_weights","down_weights","gate_scales","up_scales","down_scales"): setattr(w, a, None)
    wr.append(w)
print("loaded 3 layers", flush=True)
stream = torch.cuda.current_stream(dev).cuda_stream
g = torch.Generator().manual_seed(0)
for step, M in enumerate((7, 1, 1, 64, 1, 2048, 1, 1)):
    ids = torch.randint(0, rows, (M, 24), generator=g, dtype=torch.int32).to(dev)
    out = torch.empty((M, 24, 256), dtype=torch.bfloat16, device=dev)
    tab.lookup(ids, out)                       # engram (layer 14) before the MoE layers
    x = (torch.randn(M, H, generator=g) / 10).to(torch.bfloat16).to(dev)
    for w in wr:
        tid = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(M)]).to(torch.int32).to(dev)
        tw = (torch.rand(M, K, generator=g) / K).to(dev)
        x = w.forward(x, tid, tw, stream).view(M, H) * 0.5 + x
    torch.cuda.synchronize()
    print(f"step {step} M={M}: ok finite={bool(torch.isfinite(x.float()).all())} engram_absmax={out.float().abs().max().item():.2f}", flush=True)
print("PASS")
