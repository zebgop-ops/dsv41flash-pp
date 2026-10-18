#!/usr/bin/env python3
"""Spread decodes over the in-flight micro-batches under pipeline parallel (synchronous scheduling, i.e. with
a CPU drafter): schedule at most ceil(running / pp_size) decoding requests per step, so with 4 stages and 4
requests each micro-batch carries one request. Otherwise requests that arrive together stay in lockstep in
one batch: no pipelining across stages, and with drafts a 4-request step has up to 16 verify rows, which on
EXL3 leaves the cooperative MoE kernel's 8-row range. Same idea as Morrowmake's VLLM_PP_SPREAD_DECODES.
DSV41_PP_SPREAD=0 disables. usage: <v1/core/sched/scheduler.py>"""
import sys
p = sys.argv[1]; s = open(p).read()
if "dsv41 pp spread" in s:
    print("already patched"); sys.exit(0)
old_a = '''        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
'''
new_a = '''        # First, schedule the RUNNING requests.
        req_index = 0
        # dsv41 pp spread: cap decodes per micro-batch under PP without async scheduling
        _dsv41_spread_cap = 0
        if (
            self.use_pp
            and not self.scheduler_config.async_scheduling
            and os.environ.get("DSV41_PP_SPREAD", "1") == "1"
        ):
            _pp = self.parallel_config.pipeline_parallel_size
            _dsv41_spread_cap = max(1, -(-len(self.running) // _pp))
        _dsv41_decodes = 0
        while req_index < len(self.running) and token_budget > 0:
'''
assert s.count(old_a) == 1; s = s.replace(old_a, new_a)
old_b = '''                <= request.num_computed_tokens
            ):
                req_index += 1
                continue
            num_new_tokens = (
                request.num_tokens_with_spec
'''
new_b = '''                <= request.num_computed_tokens
            ):
                req_index += 1
                continue
            if _dsv41_spread_cap and not request.is_prefill_chunk:  # dsv41 pp spread
                if _dsv41_decodes >= _dsv41_spread_cap:
                    req_index += 1
                    continue
                _dsv41_decodes += 1
            num_new_tokens = (
                request.num_tokens_with_spec
'''
assert s.count(old_b) == 1; s = s.replace(old_b, new_b)
if "\nimport os\n" not in s and "\nimport os " not in s:
    s = s.replace("import itertools\n", "import itertools\nimport os\n", 1) if "import itertools\n" in s else "import os\n" + s
open(p, "w").write(s); print("patched", p)
