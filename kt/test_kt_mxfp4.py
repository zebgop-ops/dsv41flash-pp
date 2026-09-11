"""CPU-only test of kt-kernel's AVX2 MXFP4 MoE: correctness vs torch (plain SwiGLU),
then a decode benchmark at DeepSeek-V4.1-Flash shape."""
import os, sys, time, torch
from kt_kernel import kt_kernel_ext
E2M1 = torch.tensor([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0])

def quant(w, gs=32):
    e, n, k = w.shape
    r = w.float().view(e, n, k // gs, gs)
    amax = r.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    exp = torch.ceil(torch.log2(amax / 6.0))            # ue8m0-style power-of-two scale
    scale = torch.exp2(exp)
    nib = (r / scale).unsqueeze(-1).sub(E2M1).abs().argmin(-1).to(torch.uint8).view(e, n, k // 2, 2)
    packed = ((nib[..., 1] << 4) | nib[..., 0]).contiguous()
    return packed, scale.squeeze(-1).to(torch.bfloat16).contiguous()

def dequant(packed, scale, gs=32):
    e, n, k2 = packed.shape
    lo, hi = packed & 0xF, packed >> 4
    nib = torch.stack([lo, hi], -1).view(e, n, k2 * 2).long()
    return E2M1[nib] * scale.float().repeat_interleave(gs, -1)

def make_pool(threads):
    wp = kt_kernel_ext.WorkerPoolConfig(); wp.subpool_count = 1
    wp.subpool_numa_map = [0]; wp.subpool_thread_count = [threads]
    return kt_kernel_ext.CPUInfer(wp)

def build(cpu_infer, E, K, H, I, w, max_len, limit=0.0):
    cfg = kt_kernel_ext.moe.MOEConfig(E, K, H, I, 0)
    cfg.max_len = max_len; cfg.pool = cpu_infer.backend_
    cfg.quant_config.bits = 4; cfg.quant_config.group_size = 32; cfg.quant_config.zero_point = False
    if hasattr(cfg, "swiglu_limit"): cfg.swiglu_limit = limit
    cfg.gate_projs = [[t.data_ptr() for t in w["g"]]]; cfg.up_projs = [[t.data_ptr() for t in w["u"]]]
    cfg.down_projs = [[t.data_ptr() for t in w["d"]]]
    cfg.gate_scales = [[t.data_ptr() for t in w["gs"]]]; cfg.up_scales = [[t.data_ptr() for t in w["us"]]]
    cfg.down_scales = [[t.data_ptr() for t in w["ds"]]]
    moe = kt_kernel_ext.moe.AVX2MXFP4_MOE(cfg)
    p2l = torch.arange(E, dtype=torch.int64).contiguous()
    cpu_infer.submit(moe.load_weights_task(p2l.data_ptr())); cpu_infer.sync()
    return moe

def run(moe, cpu_infer, x, ids, wts, K):
    M = x.shape[0]; bsz = torch.tensor([M], dtype=torch.int32); y = torch.empty_like(x)
    cpu_infer.submit(moe.forward_task(bsz.data_ptr(), K, ids.data_ptr(), wts.data_ptr(), x.data_ptr(), y.data_ptr(), False))
    cpu_infer.sync(); return y

threads = int(os.environ.get("THREADS", "16"))
cpu_infer = make_pool(threads)
# ---- correctness (small) ----
torch.manual_seed(0)
E, K, H, I = 16, 6, 512, 256
g = torch.randn(E, I, H) / 20; u = torch.randn(E, I, H) / 20; d = torch.randn(E, H, I) / 20
gq, gs = quant(g); uq, us = quant(u); dq, ds = quant(d)
w = {"g": list(gq), "u": list(uq), "d": list(dq), "gs": list(gs), "us": list(us), "ds": list(ds)}
moe = build(cpu_infer, E, K, H, I, w, 64)
M = 8
x = (torch.randn(M, H) / 4).to(torch.bfloat16).contiguous()
ids = torch.stack([torch.randperm(E)[:K] for _ in range(M)]).to(torch.int64).contiguous()
wts = torch.rand(M, K).contiguous()
y = run(moe, cpu_infer, x, ids, wts, K)
gd, ud, dd = dequant(gq, gs), dequant(uq, us), dequant(dq, ds)
ref = torch.zeros(M, H)
xf = x.float()
for t in range(M):
    for j in range(K):
        e = ids[t, j].item()
        a = xf[t] @ gd[e].T; b = xf[t] @ ud[e].T
        ref[t] += wts[t, j] * ((torch.nn.functional.silu(a) * b) @ dd[e].T)
err = (y.float() - ref).abs().max().item(); rel = err / ref.abs().max().item()
print(f"correctness: max abs err {err:.4f}, rel {rel:.4f}, ref max {ref.abs().max():.3f}")
assert rel < 0.05, "mismatch"
# ---- V4.1 shape decode bench ----
E, K, H, I = int(os.environ.get("E", "384")), 6, 5120, 2304
torch.manual_seed(1)
def rnd_q(n, k):
    packed = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8)
    scale = torch.full((E, n, k // 32), 2.0 ** -6).to(torch.bfloat16)
    return packed.contiguous(), scale.contiguous()
gq, gs = rnd_q(I, H); uq, us = rnd_q(I, H); dq, ds = rnd_q(H, I)
w = {"g": list(gq), "u": list(uq), "d": list(dq), "gs": list(gs), "us": list(us), "ds": list(ds)}
print(f"bench weights: {(gq.numel()+uq.numel()+dq.numel())/2**30:.2f} GiB packed, E={E}")
moe = build(cpu_infer, E, K, H, I, w, 256, 10.0)
for M in (1, 2, 4, 8, 32, 128, 256):
    x = (torch.randn(M, H) / 10).to(torch.bfloat16).contiguous()
    ids = torch.stack([torch.randperm(E)[:K] for _ in range(M)]).to(torch.int64).contiguous()
    wts = (torch.rand(M, K) / K).contiguous()
    for _ in range(3): run(moe, cpu_infer, x, ids, wts, K)
    n = 20 if M <= 8 else 5
    t = time.perf_counter()
    for _ in range(n): run(moe, cpu_infer, x, ids, wts, K)
    dt = (time.perf_counter() - t) / n
    print(f"M={M:4d}: {dt*1e3:8.2f} ms/layer-step  ({M/dt:8.1f} tok/s per layer, {min(M*K,E)*3*5120*2304/2/dt/2**30:6.1f} GiB/s weight traffic)")
