"""kt-kernel MXFP4 with swiglu_limit vs the DeepSeek reference (clamp up to +-L, gate to max L)."""
import sys, torch
sys.path.insert(0, "/opt/dsv41/kt-site")
from kt_kernel import KTMoEWrapper
E2M1 = torch.tensor([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0])
def quant(w, gs=32):
    e, n, k = w.shape; r = w.float().view(e, n, k//gs, gs)
    amax = r.abs().amax(-1, keepdim=True).clamp(min=1e-8); scale = torch.exp2(torch.ceil(torch.log2(amax/6.0)))
    nib = (r/scale).unsqueeze(-1).sub(E2M1).abs().argmin(-1).to(torch.uint8).view(e, n, k//2, 2)
    return ((nib[...,1]<<4)|nib[...,0]).contiguous(), scale.squeeze(-1).to(torch.bfloat16).contiguous()
def dequant(p, s, gs=32):
    e, n, k2 = p.shape; nib = torch.stack([p & 0xF, p >> 4], -1).view(e, n, k2*2).long()
    return E2M1[nib] * s.float().repeat_interleave(gs, -1)
torch.manual_seed(0); E, K, H, I = 16, 6, 512, 256
g = torch.randn(E, I, H) * 2; u = torch.randn(E, I, H) * 2; d = torch.randn(E, H, I) / 20   # big -> clamps active
import tempfile, os, json
from safetensors.torch import save_file
tmp = tempfile.mkdtemp(); tensors = {}
gq, gs = quant(g); uq, us = quant(u); dq, ds = quant(d)
def e8m0(s): return (s.float().log2().round().clamp(-127, 127) + 127).to(torch.uint8)
for i in range(E):
    tensors[f"layers.0.ffn.experts.{i}.w1.weight"] = gq[i].view(torch.int8); tensors[f"layers.0.ffn.experts.{i}.w1.scale"] = e8m0(gs[i]).view(torch.float8_e8m0fnu)
    tensors[f"layers.0.ffn.experts.{i}.w3.weight"] = uq[i].view(torch.int8); tensors[f"layers.0.ffn.experts.{i}.w3.scale"] = e8m0(us[i]).view(torch.float8_e8m0fnu)
    tensors[f"layers.0.ffn.experts.{i}.w2.weight"] = dq[i].view(torch.int8); tensors[f"layers.0.ffn.experts.{i}.w2.scale"] = e8m0(ds[i]).view(torch.float8_e8m0fnu)
save_file(tensors, os.path.join(tmp, "model.safetensors"))
json.dump({"weight_map": {k: "model.safetensors" for k in tensors}}, open(os.path.join(tmp, "model.safetensors.index.json"), "w"))
dev = torch.device("cuda", 0); stream = torch.cuda.current_stream(dev).cuda_stream
for limit in (0.0, 10.0):
    w = KTMoEWrapper(layer_idx=0, num_experts=E, num_experts_per_tok=K, hidden_size=H, moe_intermediate_size=I, num_gpu_experts=0, gpu_experts_mask=None,
                     cpuinfer_threads=8, threadpool_count=1, weight_path=tmp, chunked_prefill_size=64, method="MXFP4", cpu_save=False, max_deferred_experts_per_token=0, swiglu_limit=limit)
    w.load_weights(torch.arange(E, dtype=torch.int64))
    M = 8; x = torch.randn(M, H).to(torch.bfloat16); ids = torch.stack([torch.randperm(E)[:K] for _ in range(M)]).to(torch.int32); wts = torch.rand(M, K)
    y = w.forward(x.to(dev), ids.to(dev), wts.to(dev), stream).float().cpu()
    gd, ud, dd = dequant(gq, gs), dequant(uq, us), dequant(dq, ds); xf = x.float(); ref = torch.zeros(M, H)
    for t in range(M):
        for j in range(K):
            e = ids[t, j].item(); a = xf[t] @ gd[e].T; b = xf[t] @ ud[e].T
            if limit > 0: b = b.clamp(-limit, limit); a = a.clamp(max=limit)
            ref[t] += wts[t, j] * ((torch.nn.functional.silu(a) * b) @ dd[e].T)
    err = (y - ref).abs().max().item(); print(f"limit={limit}: max abs err {err:.4f} / ref max {ref.abs().max():.3f} -> rel {err/ref.abs().max():.4f}")
