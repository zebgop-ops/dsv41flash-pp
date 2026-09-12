#!/usr/bin/env python3
"""Route any expert count through the Triton DSv4 top-k kernel (it takes NUM_EXPERTS and pads
to a power of two); upstream only admits 256/384 and falls back to a CUDA kernel with a fixed
expert table, which rejects REAP-pruned checkpoints (272 experts).
usage: <fused_moe/router/dsv4_topk.py>"""
import sys
p = sys.argv[1]; s = open(p).read()
if "dsv41 reap" in s:
    print("already patched"); sys.exit(0)
old = "        and gating_output.shape[1] in (256, 384)\n"
new = "        and 1 < gating_output.shape[1] <= 1024  # dsv41 reap: any count (kernel pads to pow2)\n"
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new)); print("patched", p)
