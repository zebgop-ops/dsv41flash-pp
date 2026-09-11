"""Sub-block oracle: feed vLLM's own attention/MoE inputs into torch references of block L's
attention and MoE and compare outputs. usage: ref_layer0_v2.py <dump_dir> [layer=0]"""
import json, sys, torch, torch.nn.functional as F
from safetensors import safe_open
S = "/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277"
cfg = json.load(open(S + "/config.json"))["text_config"]; idx = json.load(open(S + "/model.safetensors.index.json"))["weight_map"]
dump_dir = sys.argv[1]; L = int(sys.argv[2]) if len(sys.argv) > 2 else 0
dev = torch.device("cuda"); torch.set_grad_enabled(False)
def get(n):
    with safe_open(S + "/" + idx[n], "pt") as f: return f.get_tensor(n).to(dev)
def deq_fp8(w, s):
    sc = (s.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
    return w.to(torch.float32) * sc.repeat_interleave(32, 0).repeat_interleave(32, 1)
E2M1 = torch.tensor([0,.5,1,1.5,2,3,4,6,-0,-.5,-1,-1.5,-2,-3,-4,-6], device=dev)
def deq_fp4(p, s):
    p = p.view(torch.uint8); nib = torch.stack([p & 0xF, p >> 4], -1).reshape(p.shape[0], -1).long()
    sc = (s.view(torch.uint8).to(torch.int32) << 23).view(torch.float32).repeat_interleave(32, 1)
    return E2M1[nib] * sc
def lin(name):
    w = get(name + ".weight")
    if w.dtype == torch.float8_e4m3fn: return deq_fp8(w, get(name + ".scale"))
    if w.dtype == torch.int8: return deq_fp4(w, get(name + ".scale"))
    return w.float()
def rms(x, w, eps): x = x.float(); return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)) * w.float()
def cmp(name, a, b):
    a, b = a.float(), b.float(); err = (a - b).abs(); cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    print(f"{name}: max abs {err.max():.4f} mean abs {err.mean():.5f} | ref mean|.| {b.abs().mean():.5f} | cosine {cos:.5f}")
eps = cfg["rms_norm_eps"]; p = f"layers.{L}."
H, HD, RD = cfg["num_attention_heads"], cfg["head_dim"], cfg["qk_rope_head_dim"]; G, OR = cfg["o_groups"], cfg["o_lora_rank"]
# ---------------- attention ----------------
a = torch.load(f"{dump_dir}/L{L:02d}_r0_attn.pt"); x = a["x"].to(dev).float(); pos = a["positions"].to(dev); T = x.shape[0]
print("attn input", tuple(x.shape), "positions", pos.tolist())
qr = rms(x @ lin(p+"attn.wq_a").t(), get(p+"attn.q_norm.weight"), eps)
q = (qr @ lin(p+"attn.wq_b").t()).view(T, H, HD)
kv = rms(x @ lin(p+"attn.wkv").t(), get(p+"attn.kv_norm.weight"), eps)
freqs = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, RD, 2, device=dev).float() / RD))
ang = pos.float().unsqueeze(1) * freqs.unsqueeze(0); cis = torch.polar(torch.ones_like(ang), ang)
def rope(v, inverse=False):
    c = cis.conj() if inverse else cis
    xr = torch.view_as_complex(v[..., -RD:].float().contiguous().unflatten(-1, (-1, 2)))
    c = c.view(T, *([1] * (v.ndim - 2)), RD // 2)
    out = v.float().clone(); out[..., -RD:] = torch.view_as_real(xr * c).flatten(-2); return out
qR = rope(q); kR = rope(kv)
sink = get(p+"attn.attn_sink").float(); scale = HD ** -0.5
def attend(qR, kR, use_sink=True):
    logits = torch.einsum("thd,sd->ths", qR, kR) * scale
    t = torch.arange(T, device=dev); mask = ((t.unsqueeze(1) >= t.unsqueeze(0)) & ((t.unsqueeze(1) - t.unsqueeze(0)) < cfg["sliding_window"])).unsqueeze(1)
    logits = logits.masked_fill(~mask, float("-inf"))
    if use_sink: logits = torch.cat([logits, sink.view(1, H, 1).expand(T, H, 1)], -1)
    pr = torch.softmax(logits, -1)[..., :T]
    return torch.einsum("ths,sd->thd", pr, kR)
def finish(o):
    o = rope(o, inverse=True).view(T, G, -1)
    wo_a = lin(p+"attn.wo_a").view(G, OR, -1)
    return (torch.einsum("tgd,grd->tgr", o, wo_a).flatten(1)) @ lin(p+"attn.wo_b").t()
v_out = a["out"].to(dev).float()
cmp("ATTN out (vLLM vs ref, sink)", v_out, finish(attend(qR, kR)))
cmp("ATTN out (vLLM vs ref, NO sink)", v_out, finish(attend(qR, kR, use_sink=False)))
# variants to diagnose convention issues
cmp("ATTN variant: no rope anywhere", v_out, finish(attend(q, kv)))
o_nr = attend(qR, kR).view(T, G, -1); wo_a = lin(p+"attn.wo_a").view(G, OR, -1)
cmp("ATTN variant: no inverse rope", v_out, (torch.einsum("tgd,grd->tgr", o_nr, wo_a).flatten(1)) @ lin(p+"attn.wo_b").t())
# ---------------- MoE ----------------
f = torch.load(f"{dump_dir}/L{L:02d}_r0_ffn.pt"); xf = f["x"].to(dev).float(); v_ffn = f["out"].to(dev).float()
gw = get(p+"ffn.gate.weight").float(); gb = get(p+"ffn.gate.bias").float() if (p+"ffn.gate.bias") in idx else get(p+"ffn.gate.e_score_correction_bias").float()
scores = F.softplus(xf @ gw.t()).sqrt()
topi = (scores + gb).topk(cfg["num_experts_per_tok"], -1)[1]; w = scores.gather(1, topi); w = w / (w.sum(-1, keepdim=True) + 1e-20) * cfg["routed_scaling_factor"]
lim = cfg["swiglu_limit"]
def expert(prefix, xin):
    g = xin @ lin(prefix+".w1").t(); u = xin @ lin(prefix+".w3").t()
    if lim > 0: u = u.clamp(-lim, lim); g = g.clamp(max=lim)
    return (F.silu(g) * u) @ lin(prefix+".w2").t()
shared = expert(p+"ffn.shared_experts", xf); routed = torch.zeros_like(xf)
for e in sorted(set(topi.flatten().tolist())):
    tsel, k = torch.where(topi == e); routed[tsel] += w[tsel, k].unsqueeze(-1) * expert(p+f"ffn.experts.{e}", xf[tsel])
cmp("MOE out (vLLM vs ref shared+routed)", v_ffn, shared + routed)
cmp("MOE variant: routed only", v_ffn, routed); cmp("MOE variant: shared only", v_ffn, shared)
cmp("MOE variant: no route_scale", v_ffn, shared + routed / cfg["routed_scaling_factor"])
print("ref routing:", topi[0].tolist(), [round(v,3) for v in w[0].tolist()])
