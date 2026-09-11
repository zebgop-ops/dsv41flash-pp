#!/usr/bin/env python3
"""FULL cuda graphs for 1-token decode steps while speculative decoding is on.

vLLM's dispatcher knows one uniform decode query length (1 + num_spec_tokens). Steps in
which the n-gram drafter proposed nothing (1 query token per request) are therefore not
"uniform decode" and fall to PIECEWISE graphs, whose eager attention/indexer/Engram
segments cost ~35 ms/step on this box. This patch adds a second family of FULL decode
graphs captured with query length 1 (num_tokens == num_reqs), dispatches 1-token steps
to them, and stops the capture-size rounding to multiples of (K+1) so sizes 1..3 survive.
DSV41_FULL_Q1=0 disables. usage: <vllm/v1/cudagraph_dispatcher.py> <vllm/v1/worker/gpu_model_runner.py>"""
import sys
disp, runner = sys.argv[1], sys.argv[2]

def sub(src, old, new, count=1):
    assert src.count(old) == count, (old[:60], src.count(old))
    return src.replace(old, new)

# ---------------- dispatcher ----------------
src = open(disp).read()
if "dsv41 q1" in src:
    print("already patched", disp)
else:
    if "\nimport os\n" not in src:
        src = src.replace("\nfrom dataclasses import replace\n", "\nimport os\nfrom dataclasses import replace\n", 1)
    src = sub(src, '''        has_lora: bool,
        num_active_loras: int = 0,
    ) -> BatchDescriptor:
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        uniform_decode_query_len = self.uniform_decode_query_len
        num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]

        if uniform_decode and self.cudagraph_mode.has_mode(CUDAGraphMode.FULL):
            num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
            assert num_tokens_padded % uniform_decode_query_len == 0
''', '''        has_lora: bool,
        num_active_loras: int = 0,
        uniform_query_len: int | None = None,
    ) -> BatchDescriptor:
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        # dsv41 q1: the caller may name the uniform query length (1 for draft-less steps)
        uniform_decode_query_len = uniform_query_len or self.uniform_decode_query_len
        num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]

        if uniform_decode and self.cudagraph_mode.has_mode(CUDAGraphMode.FULL):
            if num_tokens_padded % uniform_decode_query_len != 0:
                # capture sizes are no longer rounded to multiples of (K+1): pick the
                # next size that is, or give up on a uniform (FULL) dispatch
                cands = [
                    s
                    for s in self.compilation_config.cudagraph_capture_sizes
                    if s >= num_tokens and s % uniform_decode_query_len == 0
                ]
                if cands:
                    num_tokens_padded = cands[0]
                else:
                    uniform_decode = False
        if uniform_decode and self.cudagraph_mode.has_mode(CUDAGraphMode.FULL):
            num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
            assert num_tokens_padded % uniform_decode_query_len == 0
''')
    src = sub(src, '''            cudagraph_capture_sizes_for_decode = [
                x
                for x in self.compilation_config.cudagraph_capture_sizes
                if x <= max_num_tokens and x >= uniform_decode_query_len
            ]
            for bs, num_active_loras in product(
                cudagraph_capture_sizes_for_decode, lora_cases
            ):
                self.add_cudagraph_key(
                    CUDAGraphMode.FULL,
                    self._create_padded_batch_descriptor(
                        bs, True, num_active_loras > 0, num_active_loras
                    ),
                )
''', '''            cudagraph_capture_sizes_for_decode = [
                x
                for x in self.compilation_config.cudagraph_capture_sizes
                if x <= max_num_tokens and x >= uniform_decode_query_len
                and x % uniform_decode_query_len == 0  # dsv41 q1: sizes are unrounded
            ]
            for bs, num_active_loras in product(
                cudagraph_capture_sizes_for_decode, lora_cases
            ):
                self.add_cudagraph_key(
                    CUDAGraphMode.FULL,
                    self._create_padded_batch_descriptor(
                        bs, True, num_active_loras > 0, num_active_loras
                    ),
                )
            # dsv41 q1: a second family of FULL decode graphs with query length 1 for
            # steps in which the drafter proposed nothing (num_tokens == num_reqs)
            if uniform_decode_query_len > 1 and os.environ.get("DSV41_FULL_Q1", "1") == "1":
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
''')
    src = sub(src, '''        valid_modes: AbstractSet[CUDAGraphMode] | None = None,
        invalid_modes: AbstractSet[CUDAGraphMode] | None = None,
    ) -> tuple[CUDAGraphMode, BatchDescriptor]:
''', '''        valid_modes: AbstractSet[CUDAGraphMode] | None = None,
        invalid_modes: AbstractSet[CUDAGraphMode] | None = None,
        uniform_query_len: int | None = None,  # dsv41 q1
    ) -> tuple[CUDAGraphMode, BatchDescriptor]:
''')
    src = sub(src, '''        batch_desc = self._create_padded_batch_descriptor(
            num_tokens, normalized_uniform, has_lora, effective_num_active_loras
        )
''', '''        batch_desc = self._create_padded_batch_descriptor(
            num_tokens, normalized_uniform, has_lora, effective_num_active_loras,
            uniform_query_len=uniform_query_len if normalized_uniform else None,
        )
''')
    open(disp, "w").write(src); print("patched", disp)

# ---------------- runner ----------------
src = open(runner).read()
if "dsv41 q1" in src:
    print("already patched", runner)
else:
    if "\nimport os\n" not in src:
        src = src.replace("\nimport torch\n", "\nimport os\nimport torch\n", 1)
    src = sub(src, '''        return (
            (
                (max_num_scheduled_tokens == uniform_decode_query_len)
                and (num_tokens == max_num_scheduled_tokens * num_reqs)
            )
            if force_uniform_decode is None
            else force_uniform_decode
        )
''', '''        if force_uniform_decode is not None:
            return force_uniform_decode
        if (
            (max_num_scheduled_tokens == uniform_decode_query_len)
            and (num_tokens == max_num_scheduled_tokens * num_reqs)
        ):
            return True
        # dsv41 q1: draft-less steps under spec decode are uniform with query length 1
        return (
            uniform_decode_query_len > 1
            and max_num_scheduled_tokens == 1
            and num_tokens == num_reqs
            and os.environ.get("DSV41_FULL_Q1", "1") == "1"
        )
''')
    src = sub(src, '''                uniform_decode=uniform_decode,
                num_active_loras=num_active_loras,
                valid_modes={CUDAGraphMode.NONE} if force_eager else valid_modes,
''', '''                uniform_decode=uniform_decode,
                num_active_loras=num_active_loras,
                valid_modes={CUDAGraphMode.NONE} if force_eager else valid_modes,
                uniform_query_len=max_num_scheduled_tokens if uniform_decode else None,
''')
    src = sub(src, '''        profile_seq_lens: int | None = None,
        randomize_inputs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
''', '''        profile_seq_lens: int | None = None,
        randomize_inputs: bool = False,
        uniform_query_len: int | None = None,  # dsv41 q1
    ) -> tuple[torch.Tensor, torch.Tensor]:
''')
    src = sub(src, '''        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens
''', '''        max_query_len = (
            (uniform_query_len or self.uniform_decode_query_len)
            if uniform_decode
            else num_tokens
        )
''')
    src = sub(src, '''                uniform_decode=desc.uniform,
''', '''                uniform_decode=desc.uniform,
                uniform_query_len=(
                    desc.num_tokens // desc.num_reqs
                    if (desc.uniform and desc.num_reqs)
                    else None
                ),
''', count=2)
    src = sub(src, '''        cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
''', '''        if (
            self.uniform_decode_query_len > 1
            and os.environ.get("DSV41_FULL_Q1", "1") == "1"
        ):
            # dsv41 q1: keep capture sizes 1..K (the dispatcher filters the K+1 family)
            self.compilation_config.adjust_cudagraph_sizes_for_spec_decode = (
                lambda *a, **k: None
            )
        cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
''')
    open(runner, "w").write(src); print("patched", runner)
