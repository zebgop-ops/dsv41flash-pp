"""Torch oracle for DeepSeek-V4.1 block 0 (SWA-only layer) vs vLLM's dumped activations.
usage: ref_layer0.py <dump_dir> [layer=0]"""
import json, math, sys, torch, torch.nn.functional as F
from safetensors import safe_open
S = "/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277"
cfg = json.load(open(S + "/config.json"))["text_config"]; idx = json.load(open(S + "/model.safetensors.index.json"))["weight_map"]
dump_dir = sys.argv[1]; L = int(sys.argv[2]) if len(sys.argv) > 2 else 0
dev = torch.device("cuda"); torch.set_grad_enabled(False)
def get(n):
    with safe_open(S + "/" + idx[n], "pt") as f: return f.get_tensor(n).to(dev)
def deq_fp8(w, s):  # [N,K] e4m3, [N/32,K/32] e8m0 -> f32
    sc = (s.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
    return w.to(torch.float32) * sc.repeat_interleave(32, 0).repeat_interleave(32, 1)
E2M1 = torch.tensor([0,.5,1,1.5,2,3,4,6,-0,-.5,-1,-1.5,-2,-3,-4,-6], device=dev)
def deq_fp4(p, s):  # [N,K/2] u8, [N,K/32] e8m0 -> f32 [N,K]
    p = p.view(torch.uint8); nib = torch.stack([p & 0xF, p >> 4], -1).reshape(p.shape[0], -1).long()
    sc = (s.view(torch.uint8).to(torch.int32) << 23).view(torch.float32).repeat_interleave(32, 1)
    return E2M1[nib] * sc
def lin(name):
    w = get(name + ".weight")
    if w.dtype == torch.float8_e4m3fn: return deq_fp8(w, get(name + ".scale"))
    if w.dtype == torch.int8: return deq_fp4(w, get(name + ".scale"))
    return w.float()
def rms(x, w, eps): x = x.float(); return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)) * w.float()
eps = cfg["rms_norm_eps"]; hc = cfg["hc_mult"]; hc_eps = cfg["hc_eps"]; iters = cfg["hc_sinkhorn_iters"]
d = torch.load(f"{dump_dir}/L{L:02d}_r0.pt"); ids = d["input_ids"].to(dev); pos = d["positions"].to(dev); T = ids.shape[0]
print("tokens", ids.tolist(), "positions", pos.tolist())
# ---- inputs: embedding broadcast, identity pre-mix (layer 0) ----
emb = get("embed.weight")[ids].float()                       # [T, D]
D = emb.shape[1]
residual = emb.unsqueeze(1).repeat(1, hc, 1)                  # [T, hc, D]
pre_mix = torch.zeros(T, hc, device=dev); pre_mix[:, 0] = 1.0
def hc_mixes(x, fn, scale, base):
    xf = x.flatten(1).float(); r = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    m = (xf @ fn.float().t()) * r
    pre = torch.sigmoid(m[:, :hc] * scale[0] + base[:hc]) + hc_eps
    post = 2 * torch.sigmoid(m[:, hc:2*hc] * scale[1] + base[hc:2*hc])
    comb = (m[:, 2*hc:] * scale[2] + base[2*hc:]).view(T, hc, hc)
    comb = torch.softmax(comb, -1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps); comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    return pre, post, comb
def hc_pre(x, pm): return (pm.unsqueeze(-1) * x).sum(1)
def hc_post(x, res, post, comb): return post.unsqueeze(-1) * x.unsqueeze(1) + torch.einsum("tjh,tjd->thd", comb, res)
p = f"layers.{L}."
attn_pre, attn_post, attn_comb = hc_mixes(residual, get(p+"hc_attn_fn"), get(p+"hc_attn_scale"), get(p+"hc_attn_base"))
x = rms(hc_pre(residual, pre_mix), get(p+"attn_norm.weight"), eps)
# ---- attention (SWA only for layer 0) ----
H, HD, RD = cfg["num_attention_heads"], cfg["head_dim"], cfg["qk_rope_head_dim"]; G, OR = cfg["o_groups"], cfg["o_lora_rank"]
qr = rms(x @ lin(p+"attn.wq_a").t(), get(p+"attn.q_norm.weight"), eps)
q = (qr @ lin(p+"attn.wq_b").t()).view(T, H, HD)
kv = rms(x @ lin(p+"attn.wkv").t(), get(p+"attn.kv_norm.weight"), eps)     # [T, 512]
freqs = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, RD, 2, device=dev).float() / RD))
ang = pos.float().unsqueeze(1) * freqs.unsqueeze(0); cis = torch.polar(torch.ones_like(ang), ang)  # [T, RD/2]
def rope(v, inverse=False):  # rotate last RD dims, adjacent pairs
    c = cis.conj() if inverse else cis
    xr = torch.view_as_complex(v[..., -RD:].float().contiguous().unflatten(-1, (-1, 2)))
    c = c.view(T, *([1] * (v.ndim - 2)), RD // 2)
    out = v.float().clone(); out[..., -RD:] = torch.view_as_real(xr * c).flatten(-2); return out
q = rope(q); kvr = rope(kv)
sink = get(p+"attn.attn_sink").float()                                    # [H]
scale = HD ** -0.5
logits = torch.einsum("thd,sd->ths", q, kvr) * scale                      # [T, H, S]
causal = torch.arange(T, device=dev).unsqueeze(1) >= torch.arange(T, device=dev).unsqueeze(0)  # t >= s
win = (torch.arange(T, device=dev).unsqueeze(1) - torch.arange(T, device=dev).unsqueeze(0)) < cfg["sliding_window"]
mask = (causal & win).unsqueeze(1)
logits = logits.masked_fill(~mask, float("-inf"))
lg = torch.cat([logits, sink.view(1, H, 1).expand(T, H, 1)], -1)
pr = torch.softmax(lg, -1)[..., :T]
o = torch.einsum("ths,sd->thd", pr, kvr)
o = rope(o, inverse=True).view(T, G, -1)
wo_a = lin(p+"attn.wo_a").view(G, OR, -1)
o = torch.einsum("tgd,grd->tgr", o, wo_a).flatten(1)
attn_out = o @ lin(p+"attn.wo_b").t()
res_after_attn = hc_post(attn_out, residual, attn_post, attn_comb)
# ---- ffn ----
ffn_pre, ffn_post, ffn_comb = hc_mixes(res_after_attn, get(p+"hc_ffn_fn"), get(p+"hc_ffn_scale"), get(p+"hc_ffn_base"))
xf = rms(hc_pre(res_after_attn, attn_pre), get(p+"ffn_norm.weight"), eps)
gw = get(p+"ffn.gate.weight").float(); gb = get(p+"ffn.gate.bias").float() if (p+"ffn.gate.bias") in idx else get(p+"ffn.gate.e_score_correction_bias").float()
scores = F.softplus(xf @ gw.t()).sqrt()
topi = (scores + gb).topk(cfg["num_experts_per_tok"], -1)[1]; w = scores.gather(1, topi); w = w / (w.sum(-1, keepdim=True) + 1e-20) * cfg["routed_scaling_factor"]
lim = cfg["swiglu_limit"]
def expert(prefix, xin):
    g = xin @ lin(prefix+".w1").t(); u = xin @ lin(prefix+".w3").t()
    if lim > 0: u = u.clamp(-lim, lim); g = g.clamp(max=lim)
    return (F.silu(g) * u) @ lin(prefix+".w2").t()
y = expert(p+"ffn.shared_experts", xf)
for e in sorted(set(topi.flatten().tolist())):
    tsel, k = torch.where(topi == e)
    y[tsel] += w[tsel, k].unsqueeze(-1) * expert(p+f"ffn.experts.{e}", xf[tsel])
res_after_block = hc_post(y, res_after_attn, ffn_post, ffn_comb)
# ---- compare with vLLM dump ----
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch
v_res_attn = d["residual"].to(dev).float(); v_hidden = d["hidden_states"].to(dev).float()
v_res_block = mhc_post_torch(d["hidden_states"].to(dev), d["residual"].to(dev), d["post_mix"].to(dev), d["res_mix"].to(dev)).float()
def cmp(name, a, b):
    err = (a - b).abs(); cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    print(f"{name}: max abs {err.max():.4f} mean abs {err.mean():.5f} | ref mean|.| {b.abs().mean():.5f} | cosine {cos:.5f}")
cmp("residual after attention (vLLM vs ref)", v_res_attn, res_after_attn)
cmp("FFN output x (vLLM hidden_states vs ref y)", v_hidden, y)
cmp("residual after block (vLLM vs ref)", v_res_block, res_after_block)
cmp("ffn pre_mix (vLLM vs ref)", d["pre_mix"].to(dev).float(), ffn_pre)
print("routing ref:", topi.tolist()[:2], "weights", [round(v, 3) for v in w[0].tolist()])
