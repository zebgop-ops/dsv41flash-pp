#!/usr/bin/env python3
"""SM8x: the DSA indexer metadata builder must gate DeepGEMM calls on architecture
support (is_deep_gemm_supported), not on package presence (has_deep_gemm), or it
calls get_paged_mqa_logits_metadata on Ampere -> 'Unsupported architecture'.
usage: patch_mla_indexer.py <vllm/v1/attention/backends/mla/indexer.py>"""
import sys
path = sys.argv[1]; src = open(path).read()
if "is_deep_gemm_supported" in src:
    print("already patched"); sys.exit(0)
assert "    has_deep_gemm,\n" in src
src = src.replace("    has_deep_gemm,\n", "    has_deep_gemm,\n    is_deep_gemm_supported,\n", 1)
n = src.count("has_deep_gemm()")
src = src.replace("has_deep_gemm()", "is_deep_gemm_supported()")
open(path, "w").write(src); print(f"patched {path}: {n} call sites")
