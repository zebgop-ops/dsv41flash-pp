"""kt-kernel MXFP4 decode-latency sweep over thread counts on one real layer.
usage: sweep_kt_threads.py <model_dir> <layer> <threads,...> [threadpool_count=1]"""
import sys, time, torch
sys.path.insert(0, "/opt/dsv41/kt-site")
from kt_kernel import KTMoEWrapper
model_dir, layer = sys.argv[1], int(sys.argv[2]); thread_list = [int(t) for t in sys.argv[3].split(",")]
tp = int(sys.argv[4]) if len(sys.argv) > 4 else 1
E, K, H, I = 384, 6, 5120, 2304
dev = torch.device("cuda", 0); stream = torch.cuda.current_stream(dev).cuda_stream
g = torch.Generator().manual_seed(0)
bytes_per_layer_step = K * 3 * H * I * 0.5  # fp4 payload touched per token (excl. scales)
for threads in thread_list:
    w = KTMoEWrapper(layer_idx=layer, num_experts=E, num_experts_per_tok=K, hidden_size=H, moe_intermediate_size=I,
                     num_gpu_experts=0, gpu_experts_mask=None, cpuinfer_threads=threads, threadpool_count=tp,
                     weight_path=model_dir, chunked_prefill_size=2048, method="MXFP4", cpu_save=False,
                     max_deferred_experts_per_token=0, swiglu_limit=10.0)
    t = time.perf_counter(); w.load_weights(torch.arange(E, dtype=torch.int64)); load = time.perf_counter() - t
    for M in (1, 2, 4, 8):
        x = (torch.randn(M, H, generator=g) / 10).to(torch.bfloat16).to(dev)
        ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(M)]).to(torch.int32).to(dev)
        wts = (torch.rand(M, K, generator=g) / K).to(dev)
        for _ in range(3): w.forward(x, ids, wts, stream); torch.cuda.synchronize()
        n = 20; torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n): w.forward(x, ids, wts, stream)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t) / n
        gbs = bytes_per_layer_step * M / dt / 1e9 if M <= 8 else 0
        print(f"threads={threads:2d} tp={tp} M={M}: {dt*1e3:7.2f} ms/step  (~{gbs:5.1f} GB/s of FP4 weights)  load {load:.0f}s", flush=True)
    del w
    import gc; gc.collect()
print("DONE")
