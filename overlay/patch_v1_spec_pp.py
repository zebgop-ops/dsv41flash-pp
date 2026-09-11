#!/usr/bin/env python3
"""V1 runner + PP + speculative decoding: the drafter is only built on the last PP rank, but the
attention-metadata builder tests `isinstance(self.drafter, ...)` on every rank (AttributeError
in the profile run on ranks 0..pp-2). Give the other ranks a None drafter.
usage: <vllm/v1/worker/gpu_model_runner.py>"""
import sys
path = sys.argv[1]; src = open(path).read()
if "dsv41: non-last PP ranks" in src:
    print("already patched"); sys.exit(0)
old = "        if self.speculative_config and get_pp_group().is_last_rank:\n"
assert src.count(old) == 1, src.count(old)
new = ("        # dsv41: non-last PP ranks never build a drafter but the metadata builder\n"
       "        # isinstance-checks it; None makes those checks false instead of raising.\n"
       "        self.drafter = None  # type: ignore[assignment]\n" + old)
open(path, "w").write(src.replace(old, new, 1)); print("patched", path)
