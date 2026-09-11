#!/usr/bin/env python3
"""SM8x: route the fp8 converts of the DeepSeek V4.1 Triton cache kernels through
vllm.v1.attention.ops.fp8_sm80 (software RNE encode / ALU-LUT decode), and gate the
CuTe DSL dispatch on SM90+. Anchor-based, idempotent.
usage: patch_v41_ops_fp8.py <vllm site dir>"""
import sys, os
site = sys.argv[1]
def edit(rel, repls, needle_done):
    p = os.path.join(site, rel); src = open(p).read()
    if needle_done in src:
        print("already patched", rel); return
    for old, new in repls:
        assert old in src, f"anchor missing in {rel}: {old[:60]!r}"
        src = src.replace(old, new, 1)
    open(p, "w").write(src); print("patched", rel)

# --- cache_utils.py (quantize/dequantize main K cache) ------------------------
edit("models/deepseek_v4_1/common/ops/cache_utils.py", [
    ("from vllm.utils.import_utils import has_cutedsl\n",
     "from vllm.utils.import_utils import has_cutedsl\n"
     "from vllm.v1.attention.ops.fp8_sm80 import _decode_fp8_f32, _encode_fp8_u8\n"
     "\n\ndef is_cutedsl_supported() -> bool:\n"
     "    # CuTe DSL kernels target SM90+; compiling them for SM8x aborts the process.\n"
     "    return has_cutedsl() and current_platform.has_device_capability(90)\n"),
    ("    if has_cutedsl():\n", "    if is_cutedsl_supported():\n"),
    ("""            if use_fnuz:
                x_fp8 = x_clamped.to(tl.float8e4b8)
            else:
                x_fp8 = x_clamped.to(tl.float8e4nv)
            x_uint8 = x_fp8.to(tl.uint8, bitcast=True)
""", """            x_uint8 = _encode_fp8_u8(x_clamped, use_fnuz)
"""),
    ("""                if use_fnuz:
                    x_fp8 = x_uint8.to(tl.float8e4b8, bitcast=True)
                else:
                    x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)

                # Convert fp8 to float32 for computation
                x_float = x_fp8.to(tl.float32)
""", """                x_float = _decode_fp8_f32(x_uint8, use_fnuz)
"""),
], "_decode_fp8_f32")

# --- fused_compress_quant_cache.py (compressor latent -> fp8 rows) ------------
edit("models/deepseek_v4_1/common/ops/fused_compress_quant_cache.py", [
    ("from vllm.triton_utils import tl, triton\n",
     "from vllm.triton_utils import tl, triton\nfrom vllm.v1.attention.ops.fp8_sm80 import _encode_e4m3fn_u8\n"),
    ("    fp8 = tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv)\n",
     "    fp8 = _encode_e4m3fn_u8(tl.clamp(scaled, -448.0, 448.0))\n"),
    ("        tl.store(dst + d, tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv))\n",
     "        tl.store(dst + d, _encode_e4m3fn_u8(tl.clamp(scaled, -448.0, 448.0)))\n"),
], "_encode_e4m3fn_u8")

# --- indexer_k_store.py (index keys -> fp8 K cache) ----------------------------
edit("models/deepseek_v4_1/common/ops/indexer_k_store.py", [
    ("from vllm.triton_utils import tl, triton\n",
     "from vllm.triton_utils import tl, triton\nfrom vllm.v1.attention.ops.fp8_sm80 import _encode_e4m3fn_u8\n"),
    ("        x_uint8 = x_clamped.to(tl.float8e4nv).to(tl.uint8, bitcast=True)\n",
     "        x_uint8 = _encode_e4m3fn_u8(x_clamped)\n"),
], "_encode_e4m3fn_u8")
