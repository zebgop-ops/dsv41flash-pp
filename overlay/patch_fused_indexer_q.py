#!/usr/bin/env python3
"""SM8x: fused indexer-Q RoPE+quant kernel -- software fp8 encode, uint8 output
pointer, and a CuTe DSL gate on SM90+. Handles both the plain-@triton.jit variant
(image 0909) and the VllmTritonJitKernel variant (main). usage: <file>"""
import re, sys
path = sys.argv[1]; src = open(path).read()
if "_encode_fp8_u8" in src:
    print("already patched"); sys.exit(0)
def rep(old, new, count=1, optional=False):
    global src
    if old not in src:
        if optional:
            return
        raise AssertionError(f"anchor missing: {old[:70]!r}")
    src = src.replace(old, new, count)
rep("from vllm.utils.import_utils import has_cutedsl\n",
    "from vllm.utils.import_utils import has_cutedsl\n"
    "from vllm.v1.attention.ops.fp8_sm80 import _encode_fp8_u8\n"
    "\n\ndef is_cutedsl_supported() -> bool:\n"
    "    # CuTe DSL kernels target SM90+; compiling them for SM8x aborts the process.\n"
    "    return has_cutedsl() and current_platform.has_device_capability(90)\n")
n = len(re.findall(r"^[ \t]+fp8_dtype = tl\.float8e4b8 if USE_FNUZ else tl\.float8e4nv\n", src, re.M))
assert n == 1, n
src = re.sub(r"^[ \t]+fp8_dtype = tl\.float8e4b8 if USE_FNUZ else tl\.float8e4nv\n", "", src, flags=re.M)
src, k = re.subn(r"tl\.div_rn\((x_nope|r_even|r_odd), index_q_scale\)\.to\(fp8_dtype\)",
                 r"_encode_fp8_u8(tl.div_rn(\1, index_q_scale), USE_FNUZ)", src)
assert k == 3, k
# warmup declaration (main variant only)
rep('''            index_q_fp8=TritonWarmupTensor(
                current_platform.fp8_dtype(),
''', '''            index_q_fp8=TritonWarmupTensor(
                torch.uint8,
''', optional=True)
# launch: hand the Triton kernel a uint8 pointer (both variants)
rep('''            index_weights_head_scale,
            index_q_fp8,
            index_weights_out,
            fp8_max=fp8_max,
''', '''            index_weights_head_scale,
            index_q_fp8.view(torch.uint8),
            index_weights_out,
            fp8_max=fp8_max,
''', optional=True)
rep('''            index_q_fp8,
            index_q_fp8.stride(0),
            index_q_fp8.stride(1),
''', '''            index_q_fp8.view(torch.uint8),
            index_q_fp8.stride(0),
            index_q_fp8.stride(1),
''', optional=True)
assert "index_q_fp8.view(torch.uint8)" in src, "no launch site patched"
c = src.count("    if has_cutedsl():\n")
assert c == 2, c
src = src.replace("    if has_cutedsl():\n", "    if is_cutedsl_supported():\n")
open(path, "w").write(src); print("patched", path)
