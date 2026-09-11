#!/usr/bin/env python3
"""SM8x supporting fixes from the V4 port (f8ea5bb) that are not upstream:
 - mhc/tilelang.py: torch fallback for prenorm-GEMM shapes tilelang cannot tile
 - utils/multi_stream_utils.py: no side-stream forks under breakable cudagraph capture
usage: patch_misc_sm80.py <vllm site dir>"""
import os, sys
site = sys.argv[1]
def edit(rel, repls, done):
    p = os.path.join(site, rel); src = open(p).read()
    if done in src:
        print("already patched", rel); return
    for old, new in repls:
        assert old in src, f"anchor missing in {rel}: {old[:70]!r}"
        src = src.replace(old, new, 1)
    open(p, "w").write(src); print("patched", rel)

edit("model_executor/kernels/mhc/tilelang.py", [
    ("""    assert x.shape[1] % n_splits == 0
    assert (x.shape[1] // n_splits) % n_thr == 0
    use_default_config = tile_n == 12 and n_thr == 512
""", """    assert x.shape[1] % n_splits == 0
    if (x.shape[1] // n_splits) % n_thr != 0:
        # Shape the tilelang kernels cannot tile; the prenorm GEMM output is
        # tiny ([T, hc_mult3] + [T]), so torch is a cheap universal fallback.
        _torch_hc_prenorm_gemm(x, fn, out, sqrsum)
        return
    use_default_config = tile_n == 12 and n_thr == 512
"""),
], "cannot tile; the prenorm GEMM output")

edit("utils/multi_stream_utils.py", [
    ("""    aux_results: list[Any]
    if aux_streams is None or not enable:
""", """    if aux_streams is not None and enable:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

        # Same rule as maybe_execute_in_parallel: no side-stream forks while
        # a breakable capture is recording the surrounding segment.
        if BreakableCUDAGraphCapture.is_active():
            aux_streams = None

    aux_results: list[Any]
    if aux_streams is None or not enable:
"""),
], "no side-stream forks while")
