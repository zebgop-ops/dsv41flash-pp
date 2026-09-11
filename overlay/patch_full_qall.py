#!/usr/bin/env python3
"""Generalize patch_full_q1: FULL decode graphs for every uniform query length 1..K (K+1 is
vLLM's native family), so partially-kept drafts (adaptive DSpark, short n-gram matches) replay
FULL graphs instead of PIECEWISE ones. usage: <cudagraph_dispatcher.py> <gpu_model_runner.py>"""
import sys
disp, runner = sys.argv[1:3]

def sub(src, old, new):
    assert src.count(old) == 1, (old[:70], src.count(old))
    return src.replace(old, new)

s = open(disp).read()
if "dsv41 qall" in s:
    print("already patched", disp)
else:
    s = sub(s, '''            if uniform_decode_query_len > 1 and os.environ.get("DSV41_FULL_Q1", "1") == "1":
                max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
                for bs, num_active_loras in product(
                    [
                        x
                        for x in self.compilation_config.cudagraph_capture_sizes
                        if x <= max_num_seqs
                    ],
                    lora_cases,
                ):
                    self.add_cudagraph_key(
                        CUDAGraphMode.FULL,
                        self._create_padded_batch_descriptor(
                            bs, True, num_active_loras > 0, num_active_loras,
                            uniform_query_len=1,
                        ),
                    )
''', '''            if uniform_decode_query_len > 1 and os.environ.get("DSV41_FULL_Q1", "1") == "1":
                # dsv41 qall: one FULL family per uniform query length q < K+1
                max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
                for q in range(1, uniform_decode_query_len):
                    for bs, num_active_loras in product(
                        [
                            x
                            for x in self.compilation_config.cudagraph_capture_sizes
                            if x % q == 0 and x // q <= max_num_seqs
                        ],
                        lora_cases,
                    ):
                        self.add_cudagraph_key(
                            CUDAGraphMode.FULL,
                            self._create_padded_batch_descriptor(
                                bs, True, num_active_loras > 0, num_active_loras,
                                uniform_query_len=q,
                            ),
                        )
''')
    open(disp, "w").write(s); print("patched", disp)

s = open(runner).read()
if "dsv41 qall" in s:
    print("already patched", runner)
else:
    s = sub(s, '''        # dsv41 q1: draft-less steps under spec decode are uniform with query length 1
        return (
            uniform_decode_query_len > 1
            and max_num_scheduled_tokens == 1
            and num_tokens == num_reqs
            and os.environ.get("DSV41_FULL_Q1", "1") == "1"
        )
''', '''        # dsv41 qall: any uniform query length below K+1 (draft-less steps, partially kept
        # drafts) dispatches to its own FULL family
        return (
            uniform_decode_query_len > 1
            and 1 <= max_num_scheduled_tokens < uniform_decode_query_len
            and num_tokens == max_num_scheduled_tokens * num_reqs
            and os.environ.get("DSV41_FULL_Q1", "1") == "1"
        )
''')
    open(runner, "w").write(s); print("patched", runner)
