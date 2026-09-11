#!/usr/bin/env python3
"""Register the SM8x Triton backends for DeepSeek V4.1 in the attention backend enum.
usage: patch_registry.py <vllm/v1/attention/backends/registry.py>"""
import sys
path = sys.argv[1]; src = open(path).read()
if "TRITON_MLA_SPARSE_DSV41" in src:
    print("already patched"); sys.exit(0)
old = '''    FLASHINFER_MLA_SPARSE_DSV41 = (
        "vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse."
        "DeepseekV4FlashInferMLASparseBackend"
    )
'''
new = old + '''    # SM8x (Ampere) Triton sparse MLA for DeepSeek V4.1 (overlay).
    TRITON_MLA_SPARSE_DSV41 = (
        "vllm.models.deepseek_v4_1.ampere.ampere_sparse."
        "DeepseekV41AmpereMLASparseBackend"
    )
    TRITON_SPARSE_SWA_DSV41 = (
        "vllm.models.deepseek_v4_1.ampere.ampere_sparse."
        "DeepseekV41AmpereSparseSWABackend"
    )
'''
assert old in src, "anchor not found"
open(path, "w").write(src.replace(old, new, 1)); print("patched", path)
