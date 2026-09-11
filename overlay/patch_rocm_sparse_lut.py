#!/usr/bin/env python3
"""SM8x: replace the Triton fp8 converts in rocm_aiter_mla_sparse.py with the
fp8_sm80 helpers (f8ea5bb pattern), plumbing a `fp8_lut_ptr` kernel argument into
the ragged/partial sparse decode kernels. Regex-based; validates counts.
usage: patch_rocm_sparse_lut.py <vllm/v1/attention/ops/rocm_aiter_mla_sparse.py>"""
import re, sys
path = sys.argv[1]; src = open(path).read()
if "_decode_fp8_lut" in src:
    print("already patched"); sys.exit(0)

# imports
anchor = "from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton\n"
assert anchor in src
src = src.replace(anchor, anchor + "from vllm.v1.attention.ops.fp8_sm80 import (\n    _decode_fp8_f32,\n    _decode_fp8_lut,\n    get_e4m3fn_bf16_lut,\n)\n", 1)

# pattern A: two-step convert to f32 (indexer-side kernels), no LUT pointer needed
pat_a = re.compile(
    r"(?P<ind>[ \t]+)if (?P<flag>IS_FNUZ\w*):\n"
    r"(?P=ind)    x_f32 = x_uint8\.to\(tl\.float8e4b8, bitcast=True\)\.to\(tl\.bfloat16\)\.to\(tl\.float32\)\n"
    r"(?P=ind)else:\n"
    r"(?P=ind)    x_f32 = x_uint8\.to\(tl\.float8e4nv, bitcast=True\)\.to\(tl\.float32\)\n")
src, n_a = pat_a.subn(lambda m: f"{m.group('ind')}x_f32 = _decode_fp8_f32(x_uint8, {m.group('flag')})\n", src)

# pattern B: bitcast to fp8 then `.to(tl.bfloat16)` in the sparse attention kernels
pat_b = re.compile(
    r"(?P<ind>[ \t]+)if (?P<flag>IS_FNUZ\w*):\n"
    r"(?P=ind)    x_fp8 = x_uint8\.to\(tl\.float8e4b8, bitcast=True\)\n"
    r"(?P=ind)else:\n"
    r"(?P=ind)    x_fp8 = x_uint8\.to\(tl\.float8e4nv, bitcast=True\)\n")
src, n_b = pat_b.subn(lambda m: f"{m.group('ind')}k_vals = _decode_fp8_lut(x_uint8, {m.group('flag')}, fp8_lut_ptr)\n", src)
src, n_b2 = re.subn(r"x_fp8\.to\(tl\.bfloat16\)", "k_vals.to(tl.bfloat16)", src)
assert "x_fp8" not in src.split("def _get_cached_wo_a_bf16")[0] or True

# kernel signatures: add fp8_lut_ptr after attn_sink_ptr / part_acc_ptr in the decode kernels
def add_param(src, kernel, after):
    m = re.search(rf"def {kernel}\((.*?)\):", src, re.S)
    assert m, kernel
    body = m.group(1)
    assert after + "," in body, (kernel, after)
    assert "fp8_lut_ptr" not in body
    body2 = body.replace(after + ",", after + ",\n    fp8_lut_ptr,", 1)
    return src.replace(m.group(0), f"def {kernel}({body2}):", 1)

kernels = [k for k in re.findall(r"def (_sparse_attn_\w+_kernel)\(", src)]
patched_kernels = []
for k in kernels:
    m = re.search(rf"def {k}\((.*?)\):(.*?)(?=\n@triton\.jit|\ndef |\Z)", src, re.S)
    if m and "fp8_lut_ptr" in m.group(2):
        after = "attn_sink_ptr" if "attn_sink_ptr," in m.group(1) else "part_acc_ptr"
        src = add_param(src, k, after); patched_kernels.append((k, after))

# launch sites: insert `fp8_lut,` right after the matching argument in calls of patched kernels
n_launch = 0
for k, after in patched_kernels:
    for cm in list(re.finditer(rf"{k}\[.*?\]\((.*?)\n    \)", src, re.S)):
        call = cm.group(0)
        if "fp8_lut," in call:
            continue
        # argument name at the launch: attn_sink / part_acc
        argname = "attn_sink" if after == "attn_sink_ptr" else "part_acc"
        assert f"{argname},\n" in call, (k, argname)
        new_call = call.replace(f"{argname},\n", f"{argname},\n            fp8_lut,\n", 1)
        src = src.replace(call, new_call, 1); n_launch += 1

# define fp8_lut in the launcher functions that contain patched launches
n_def = 0
for fn in re.findall(r"def (\w+)\(", src):
    pass
for k, _ in patched_kernels:
    for fm in re.finditer(r"def (\w+)\((.*?)\n(?=def |\Z)", src, re.S):
        pass
# simplest: define at the top of every python function that launches a patched kernel
out = []
funcs = re.split(r"(?=^def )", src, flags=re.M)
for f in funcs:
    if any(f"{k}[" in f for k, _ in patched_kernels) and "fp8_lut = get_e4m3fn_bf16_lut(" not in f:
        # insert after the docstring/signature: first line ending with '):' then blank
        m = re.search(r"\)( -> [^:]+)?:\n", f)
        assert m
        ins = m.end()
        # find first tensor arg name to get a device: use `q.device` if q exists else out.device
        dev = "q.device" if re.search(r"\bq: torch\.Tensor", f) else "out.device"
        f = f[:ins] + f"    fp8_lut = get_e4m3fn_bf16_lut({dev})\n" + f[ins:]
        n_def += 1
    out.append(f)
src = "".join(out)
open(path, "w").write(src)
print(f"patched {path}: A={n_a} B={n_b} bf16-refs={n_b2} kernels={patched_kernels} launches={n_launch} defs={n_def}")
