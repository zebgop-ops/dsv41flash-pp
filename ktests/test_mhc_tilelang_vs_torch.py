"""V4.1 mHC pre (delayed) / post: tilelang kernels on SM8x vs torch reference, real layer-0 params."""
import json, torch
from safetensors import safe_open
from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_delayed_tilelang, mhc_post_tilelang
from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch, mhc_post_torch
S = "/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277"
idx = json.load(open(S + "/model.safetensors.index.json"))["weight_map"]
dev = torch.device("cuda")
def get(n):
    with safe_open(S + "/" + idx[n], "pt") as f: return f.get_tensor(n).to(dev)
L = 3
fn = get(f"layers.{L}.hc_attn_fn").float(); base = get(f"layers.{L}.hc_attn_base").float(); scale = get(f"layers.{L}.hc_attn_scale").float()
nw = get(f"layers.{L}.attn_norm.weight")
print("fn", tuple(fn.shape), "base", tuple(base.shape), "scale", scale.tolist(), "norm", tuple(nw.shape), nw.dtype)
torch.manual_seed(0); T, HC, H = 37, 4, 5120
residual = (torch.randn(T, HC, H, device=dev) * 0.5).to(torch.bfloat16).contiguous()
pre_mix = torch.softmax(torch.randn(T, HC, device=dev), -1).contiguous()
args = (residual, fn, scale, base, 1e-20, 1e-6, 1e-6, 2.0, 20)
post_t, comb_t, x_t, pre_t = mhc_pre_delayed_tilelang(*args, pre_mix=pre_mix, norm_weight=nw, norm_eps=1e-20)
post_r, comb_r, x_r, pre_r = mhc_pre_delayed_torch(*args, pre_mix=pre_mix)
h = x_r.float(); x_rn = (h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-20) * nw.float()).to(torch.bfloat16)
def cmp(name, a, b):
    a, b = a.float(), b.float(); err = (a - b).abs(); print(f"{name}: shape {tuple(a.shape)} max abs {err.max():.4f} mean rel {(err/(b.abs()+1e-3)).mean():.4f} ref absmax {b.abs().max():.3f}")
cmp("post", post_t, post_r); cmp("comb", comb_t, comb_r); cmp("x(normed)", x_t, x_rn); cmp("pre", pre_t, pre_r)
x = (torch.randn(T, H, device=dev) * 0.3).to(torch.bfloat16)
out_t = mhc_post_tilelang(x, residual, post_t, comb_t); out_r = mhc_post_torch(x, residual, post_r, comb_r)
cmp("post-out", out_t, out_r)
# first layer variant (broadcast): x given, fn_broadcast shape (mix, H)
fnb = fn.view(-1, HC, H).sum(1).contiguous() if fn.shape[1] == HC * H else fn
x0 = (torch.randn(T, H, device=dev) * 0.5).to(torch.bfloat16); res0 = x0.unsqueeze(1).expand(-1, HC, -1).contiguous()
a = (res0, fnb, scale, base, 1e-20, 1e-6, 1e-6, 2.0, 20)
pt, ct, xt, prt = mhc_pre_delayed_tilelang(*a, x=x0, norm_weight=nw, norm_eps=1e-20)
pr, cr, xr, prr = mhc_pre_delayed_torch(*a, x=x0)
h = xr.float(); xrn = (h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-20) * nw.float()).to(torch.bfloat16)
cmp("bcast post", pt, pr); cmp("bcast comb", ct, cr); cmp("bcast x", xt, xrn); cmp("bcast pre", prt, prr)
