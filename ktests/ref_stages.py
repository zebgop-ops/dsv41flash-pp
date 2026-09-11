"""Stage oracle: compare each attention stage dumped by ampere_sparse.py (DSV41_DEBUG_DUMP)
against the torch reference of DeepSeek's inference/model.py. usage: ref_stages.py <dump_dir> [layer=0]"""
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
def lin(name):
    w = get(name + ".weight")
    if w.dtype == torch.float8_e4m3fn: return deq_fp8(w, get(name + ".scale"))
    return w.float()
def rms(x, w, eps): x = x.float(); return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)) * w.float()
def cmp(name, a, b):
    a, b = a.float(), b.float(); err = (a - b).abs(); cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    print(f"{name:52s} max abs {err.max():8.4f} mean abs {err.mean():8.5f} | ref mean|.| {b.abs().mean():8.5f} | cos {cos:.5f}")
eps = cfg["rms_norm_eps"]; p = f"layers.{L}."
H, HD, RD = cfg["num_attention_heads"], cfg["head_dim"], cfg["qk_rope_head_dim"]; G, OR = cfg["o_groups"], cfg["o_lora_rank"]
a = torch.load(f"{dump_dir}/L{L:02d}_r0_attn.pt"); x = a["x"].to(dev).float(); pos = a["positions"].to(dev); T = x.shape[0]
st = torch.load(f"{dump_dir}/L{L:02d}_r0_stages.pt")
print("T", T, "positions", pos.tolist()[:8], "... wo_a dtype", st.get("wo_a_dtype"), "scale", st.get("scale"), "ref scale", HD ** -0.5)
assert torch.equal(st["positions"], a["positions"]), "stage dump positions != attn dump positions"
# ---- reference stages ----
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
    qR, kR = qR.float(), kR.float()
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
# ---- stage 1: projections + norms (pre-RoPE) ----
q_pre = st["q_pre"].to(dev)[:, :H]; kv_pre = st["kv_pre"].to(dev)
cmp("S1 q pre-RoPE (wq_a/q_norm/wq_b)", q_pre, q)
cmp("S1 kv pre-RoPE (wkv/kv_norm)", kv_pre, kv)
cmp("S1 q nope lanes", q_pre[..., :HD-RD], q[..., :HD-RD]); cmp("S1 q rope lanes", q_pre[..., HD-RD:], q[..., HD-RD:])
# ---- stage 2: q after RoPE ----
q_rope = st["q_rope"].to(dev)[:, :H]
cmp("S2 q post-RoPE (fused insert op)", q_rope, qR)
cmp("S2 q post-RoPE rope lanes", q_rope[..., HD-RD:], qR[..., HD-RD:])
cmp("S2 q post-RoPE vs rope(vLLM q_pre)", q_rope, rope(q_pre))
# ---- stage 3: gathered K rows ----
pq = st["prefill_q"].to(dev); ind = st["prefill_indices"].to(dev); lens = st["prefill_lens"]
rows = st["prefill_kv_rows"].to(dev); kvg = st["prefill_kv"].to(dev)
print("prefill q", tuple(pq.shape), "indices", tuple(ind.shape), "lens", None if lens is None else lens.tolist()[:8], "rows used", rows.numel(), "min/max", rows.min().item(), rows.max().item())
cmp("S3 prefill_q vs q_rope", pq[:, :H], q_rope)
# map: row id -> position. Ref: each query t attends rows for positions <= t. Assume rows are contiguous per position in order.
if rows.numel() == T:
    rowpos = torch.arange(T, device=dev)
    cmp("S3 gathered K rows vs ref kR (assume row order = pos)", kvg, kR)
    cmp("S3 gathered K nope (fp8 ue8m0)", kvg[:, :HD-RD], kR[:, :HD-RD]); cmp("S3 gathered K rope (bf16)", kvg[:, HD-RD:], kR[:, HD-RD:])
    cmp("S3 gathered K vs rope(vLLM kv_pre)", kvg, rope(kv_pre))
    # per-query index sets
    bad = 0
    row_of = {int(r): i for i, r in enumerate(rows.tolist())}
    for t in range(T):
        sel = ind[t]; sel = sel[sel >= 0].tolist()
        want = set(rows[:t+1].tolist())
        if set(sel) != want: bad += 1
    print("S3 index sets matching causal prefix:", T - bad, "/", T)
else:
    print("S3: rows used != T; indices need explicit row->pos mapping")
# ---- stage 4: sparse attention kernel ----
o_v = st["attn_o"].to(dev)[:, :H]
cmp("S4 attn kernel out vs ref attend(qR,kR)", o_v, attend(qR, kR))
# kernel with vLLM's own inputs (q_rope, gathered K)
if rows.numel() == T:
    o_own = attend(q_rope, kvg); cmp("S4 attn kernel out vs attend(vLLM q, vLLM K)", o_v, o_own)
    cmp("S4 ... same, no sink", o_v, attend(q_rope, kvg, use_sink=False))
    # try alt scale conventions
    for nm, sc in [("scale*sqrt2", scale * 2 ** 0.5), ("scale/sqrt2", scale / 2 ** 0.5), ("1/sqrt(576)", 576 ** -0.5)]:
        lg = torch.einsum("thd,sd->ths", q_rope.float(), kvg.float()) * sc
        t_ = torch.arange(T, device=dev); m = (t_.unsqueeze(1) >= t_.unsqueeze(0)).unsqueeze(1)
        lg = lg.masked_fill(~m, float("-inf")); lg = torch.cat([lg, sink.view(1, H, 1).expand(T, H, 1)], -1)
        cmp(f"S4 variant {nm}", o_v, torch.einsum("ths,sd->thd", torch.softmax(lg, -1)[..., :T], kvg.float()))
# ---- stage 5: o_proj ----
oi = st["o_proj_in"].to(dev); oo = st["o_proj_out"].to(dev); v_out = a["out"].to(dev)
cmp("S5 o_proj in vs attn_o", oi, o_v)
cmp("S5 o_proj out vs finish(vLLM o)", oo, finish(oi))
cmp("S5 o_proj out vs attn hook out", oo, v_out)
cmp("S5 o_proj out vs ref end-to-end", oo, finish(attend(qR, kR)))
