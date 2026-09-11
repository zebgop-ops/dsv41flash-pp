#!/usr/bin/env python3
"""Route DeepSeek V4.1 attention to the Ampere subclass on SM8x.
usage: patch_select_attn.py <vllm/models/deepseek_v4_1/nvidia/model.py>"""
import sys
path = sys.argv[1]; src = open(path).read()
if "DeepseekV41AmpereMLAAttention" in src:
    print("already patched"); sys.exit(0)
old = '''    backend = vllm_config.attention_config.backend
    device_capability = current_platform.get_device_capability()
    if backend in (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE,
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
    ):
        raise ValueError(
            f"{backend.name} is not a DeepSeek V4.1 attention backend. "
'''
new = '''    backend = vllm_config.attention_config.backend
    device_capability = current_platform.get_device_capability()
    if device_capability is not None and device_capability.major == 8:
        if backend is not None and backend.name != "TRITON_MLA_SPARSE_DSV41":
            raise ValueError(
                f"{backend.name} is not supported for DeepSeek V4.1 on SM8x; "
                "use TRITON_MLA_SPARSE_DSV41 (default)."
            )
        from vllm.models.deepseek_v4_1.ampere.ampere_sparse import (
            DeepseekV41AmpereMLAAttention,
        )

        return DeepseekV41AmpereMLAAttention
    if backend in (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE,
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
    ):
        raise ValueError(
            f"{backend.name} is not a DeepSeek V4.1 attention backend. "
'''
assert old in src, "anchor not found"
src = src.replace(old, new, 1)
open(path, "w").write(src); print("patched", path)
