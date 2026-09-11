"""Isolated kt-kernel MXFP4 forward on real DeepSeek-V4.1 expert weights (one layer), GPU-resident
activations like vLLM: usage python3 test_kt_real_layer.py <model_dir> <layer> [threads]"""
import os, sys, time, torch
sys.path.insert(0, "/opt/dsv41/kt-site")
from kt_kernel import KTMoEWrapper
model_dir, layer = sys.argv[1], int(sys.argv[2]); threads = int(sys.argv[3]) if len(sys.argv) > 3 else 16
E, K, H, I = 384, 6, 5120, 2304
w = KTMoEWrapper(layer_idx=layer, num_experts=E, num_experts_per_tok=K, hidden_size=H, moe_intermediate_size=I,
                 num_gpu_experts=0, gpu_experts_mask=None, cpuinfer_threads=threads, threadpool_count=1,
                 weight_path=model_dir, chunked_prefill_size=2048, method="MXFP4", cpu_save=False,
                 max_deferred_experts_per_token=0, swiglu_limit=10.0)
t = time.perf_counter(); w.load_weights(torch.arange(E, dtype=torch.int64)); print(f"loaded layer {layer} in {time.perf_counter()-t:.1f}s", flush=True)
dev = torch.device("cuda", 0); stream = torch.cuda.current_stream(dev).cuda_stream
g = torch.Generator().manual_seed(0)
for M in (1, 7, 64, 512, 2048):
    x = (torch.randn(M, H, generator=g) / 10).to(torch.bfloat16).to(dev)
    ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(M)]).to(torch.int32).to(dev)
    wts = (torch.rand(M, K, generator=g) / K).to(dev)
    torch.cuda.synchronize(); t = time.perf_counter()
    y = w.forward(x, ids, wts, stream); torch.cuda.synchronize()
    dt = time.perf_counter() - t
    print(f"M={M:5d}: {dt*1e3:8.1f} ms  out {tuple(y.shape)} {y.dtype} finite={bool(torch.isfinite(y.float()).all())} absmax={y.float().abs().max().item():.3f}", flush=True)
print("PASS")
