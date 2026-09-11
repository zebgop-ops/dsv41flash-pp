#!/usr/bin/env python3
"""DSV41_MHC_TORCH=1: run the mHC pre/post steps with the torch reference instead of
the tilelang kernels (bisecting numerics on SM8x). usage: <model.py>"""
import sys
path = sys.argv[1]; src = open(path).read()
if "_dsv41_mhc_pre_torch" in src:
    print("already patched"); sys.exit(0)
old = "from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton\n"
new = old + '''

def _dsv41_mhc_pre_torch(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                         hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                         pre_mix=None, x=None, norm_weight=None, norm_eps=1e-6):
    from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch

    post, comb, layer_input, pre = mhc_pre_delayed_torch(
        residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
        hc_post_mult_value, sinkhorn_repeat, pre_mix=pre_mix, x=x,
    )
    if norm_weight is not None:
        h = layer_input.float()
        h = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + norm_eps)
        layer_input = (h * norm_weight.float()).to(residual.dtype)
    return post.contiguous(), comb.contiguous(), layer_input.contiguous(), pre.contiguous()


if __import__("os").environ.get("DSV41_MHC_TORCH") == "1":
    from vllm.model_executor.kernels.mhc.torch import mhc_post_torch as _dsv41_mhc_post_torch

    mhc_pre_delayed_tilelang = _dsv41_mhc_pre_torch  # noqa: F811
    mhc_post_tilelang = _dsv41_mhc_post_torch  # noqa: F811
'''
assert old in src; src = src.replace(old, new, 1)
open(path, "w").write(src); print("patched", path)
