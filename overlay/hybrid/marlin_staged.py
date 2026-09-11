"""Marlin MXFP4 MoE weight preparation staged through host RAM.

vLLM's prepare keeps the raw [E, N, K/2] expert tensor on the GPU while the packed copy is
built (and the caller frames pin both raw tensors), so every MoE layer costs ~6.7 GiB of
transient HBM at load. That is what limits a CMP 170HX rank to 7 expert layers. Here the
raw tensor is moved to the host first and the packed tensor is filled expert by expert, so
the GPU peak is just the packed weight: 8 expert layers per rank fit.
"""
from __future__ import annotations

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
    marlin_permute_bias,
    marlin_permute_scales,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    mxfp4_marlin_process_scales,
)

try:
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import get_marlin_input_dtype
except ImportError:  # lives next to the other marlin helpers in some versions
    from vllm.model_executor.layers.quantization.utils.marlin_utils import get_marlin_input_dtype
from vllm.model_executor.utils import replace_parameter


def _repack_staged(layer, name: str, size_n: int, size_k: int, perm, is_a_8bit: bool):
    raw = getattr(layer, name)
    device = raw.device
    e = raw.shape[0]
    assert raw.shape == (e, size_n, size_k // 2), raw.shape
    # pageable on purpose: torch's pinned-host allocator caches freed blocks for the life
    # of the process, which would keep ~6.7 GiB of locked host RAM per worker after load.
    host = torch.empty(raw.shape, dtype=raw.dtype, device="cpu")
    host.copy_(raw.data)
    torch.cuda.synchronize(device)
    # drop the GPU copy before allocating the packed one
    replace_parameter(layer, name, torch.empty(0, dtype=raw.dtype, device=device))
    del raw
    torch.cuda.empty_cache()
    out = None
    stage = torch.empty(host.shape[1:], dtype=host.dtype, device=device)
    for i in range(e):
        stage.copy_(host[i])
        qweight = stage.view(torch.int32).T.contiguous()
        packed = ops.gptq_marlin_repack(
            b_q_weight=qweight, perm=perm, size_k=size_k, size_n=size_n, num_bits=4, is_a_8bit=is_a_8bit
        )
        if out is None:
            out = torch.empty((e, *packed.shape), dtype=packed.dtype, device=device)
        out[i] = packed
    del stage, host
    return out


def prepare_marlin_mxfp4_moe_staged(layer) -> None:
    """In-place equivalent of vLLM's prepare_moe_fp4_layer_for_marlin (MXFP4 flavour)."""
    input_dtype = get_marlin_input_dtype()
    if input_dtype is not None and input_dtype.itemsize == 1 and input_dtype != torch.float8_e4m3fn:
        raise RuntimeError("MXFP4 weight + INT8 activation is not supported.")
    group_size = 32
    w13 = layer.w13_weight
    e, n2, khalf = w13.shape
    n, k = n2 // 2, khalf * 2
    device = w13.device
    param_dtype = layer.params_dtype
    is_a_8bit = input_dtype is not None and input_dtype.itemsize == 1
    del w13
    layer.workspace = marlin_make_workspace_new(device, 4, existing=getattr(layer, "workspace", None))
    perm = torch.empty(0, dtype=torch.int, device=device)

    for name, size_n, size_k in (("w13_weight", n * 2, k), ("w2_weight", k, n)):
        packed = _repack_staged(layer, name, size_n, size_k, perm, is_a_8bit)
        replace_parameter(layer, name, packed)
        del packed
        torch.cuda.empty_cache()

    for name, size_n, size_k in (("w13", n * 2, k), ("w2", k, n)):
        scales = getattr(layer, name + "_weight_scale").view(torch.float8_e8m0fnu).to(param_dtype)
        tensor_list = []
        for i in range(e):
            ms = marlin_permute_scales(s=scales[i].T, size_k=size_k, size_n=size_n, group_size=group_size, is_a_8bit=is_a_8bit)
            tensor_list.append(mxfp4_marlin_process_scales(ms, input_dtype=input_dtype))
        replace_parameter(layer, name + "_weight_scale", torch.cat([x.unsqueeze(0) for x in tensor_list], 0))
        del scales, tensor_list

    for name in ("w13_bias", "w2_bias"):
        if getattr(layer, name, None) is None:
            continue
        bias = getattr(layer, name).to(param_dtype)
        replace_parameter(layer, name, torch.cat([marlin_permute_bias(bias[i]).unsqueeze(0) for i in range(e)], 0))
