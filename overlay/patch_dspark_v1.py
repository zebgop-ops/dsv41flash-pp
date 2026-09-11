#!/usr/bin/env python3
"""DSpark on the V1 model runner under PP (hybrid/dspark_proposer.py): drafter selection,
dspark_target_layer_ids as the aux-hidden-state layers, an "only" regex for the draft's
checkpoint read, and a non-pipelined draft parallel config.
usage: <gpu_model_runner.py> <weight_utils.py> <config/speculative.py> <config/vllm.py>"""
import sys
runner, wu, spec, vcfg = sys.argv[1:5]

def sub(src, old, new, count=1):
    assert src.count(old) == count, (old[:70], src.count(old))
    return src.replace(old, new)

s = open(runner).read()
if "dsv41 dspark" in s:
    print("already patched", runner)
else:
    s = sub(s, '''            elif self.speculative_config.use_dflash():
                self.drafter = DFlashProposer(self.vllm_config, self.device, self)
                self.use_aux_hidden_state_outputs = True
''', '''            elif self.speculative_config.use_dspark():
                # dsv41 dspark: V1-runner port of the V2 DSpark speculator
                from hybrid.dspark_proposer import DSparkProposer

                self.drafter = DSparkProposer(self.vllm_config, self.device, self)
                self.use_aux_hidden_state_outputs = True
            elif self.speculative_config.use_dflash():
                self.drafter = DFlashProposer(self.vllm_config, self.device, self)
                self.use_aux_hidden_state_outputs = True
''')
    s = sub(s, '''            if eagle_config and isinstance(eagle_config, dict):
                layer_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")

        if layer_ids and isinstance(layer_ids, (list, tuple)):
''', '''            if eagle_config and isinstance(eagle_config, dict):
                layer_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")

        if not layer_ids:
            # dsv41 dspark: v4.1 reads the attention *inputs* of its target layers and the
            # model captures the entry stream of layer L when idx + 1 == L (ids as-is);
            # v4 ids are capture-after and keep the +1 (mirrors V2 eagle3_utils).
            dspark_layer_ids = getattr(hf_config, "dspark_target_layer_ids", None)
            if dspark_layer_ids:
                if getattr(hf_config, "model_type", None) == "deepseek_v41":
                    layer_ids = list(dspark_layer_ids)
                else:
                    layer_ids = [i + 1 for i in dspark_layer_ids]

        if layer_ids and isinstance(layer_ids, (list, tuple)):
''')
    open(runner, "w").write(s); print("patched", runner)

s = open(wu).read()
if "_DSV41_ONLY_RE" in s:
    print("already patched", wu)
else:
    s = sub(s, '''def _dsv41_skip(name: str) -> bool:
    return _DSV41_SKIP_RE is not None and _DSV41_SKIP_RE.search(name) is not None
''', '''_DSV41_ONLY_RE = None  # set transiently by hybrid/dspark_proposer.py around the draft load


def _dsv41_skip(name: str) -> bool:
    if _DSV41_ONLY_RE is not None and _DSV41_ONLY_RE.search(name) is None:
        return True
    return _DSV41_SKIP_RE is not None and _DSV41_SKIP_RE.search(name) is not None
''')
    open(wu, "w").write(s); print("patched", wu)

s = open(spec).read()
if "dsv41 dspark" in s:
    print("already patched", spec)
else:
    s = sub(s, '''                self.draft_parallel_config = (
                    SpeculativeConfig.create_draft_parallel_config(
                        self.target_parallel_config, self.draft_tensor_parallel_size
                    )
                )
''', '''                self.draft_parallel_config = (
                    SpeculativeConfig.create_draft_parallel_config(
                        self.target_parallel_config, self.draft_tensor_parallel_size
                    )
                )
                if (
                    self.method == "dspark"
                    and self.draft_parallel_config.pipeline_parallel_size > 1
                ):
                    # dsv41 dspark: the draft is not pipelined, it lives whole on the
                    # last rank (verify_with_parallel_config would demand SupportsPP)
                    self.draft_parallel_config.pipeline_parallel_size = 1
''')
    open(spec, "w").write(s); print("patched", spec)

s = open(vcfg).read()
if "dsv41 dspark" in s:
    print("already patched", vcfg)
else:
    s = sub(s, '''            if self.speculative_config.method == "dspark":
                unsupported.append("dspark speculative decoding")
''', '''            if self.speculative_config.method == "dspark" and not _dsv41_dspark_v1():
                unsupported.append("dspark speculative decoding")
''')
    s = sub(s, "\nlogger = init_logger(__name__)\n", '''
logger = init_logger(__name__)


def _dsv41_dspark_v1() -> bool:
    """dsv41 dspark: the overlay ports DSpark to the V1 runner (hybrid/dspark_proposer.py)."""
    import os

    return os.environ.get("DSV41_DSPARK_V1", "1") == "1"
''')
    open(vcfg, "w").write(s); print("patched", vcfg)

# non-last PP ranks have no drafter (patch_v1_spec_pp sets it to None); the eagle-family
# gates below must not assert there
s = open(runner).read()
if "self.drafter is not None and (" in s:
    print("already patched (pp gates)", runner)
else:
    for old in (
        '''            if self.speculative_config and (
                self.speculative_config.use_eagle()
                or self.speculative_config.uses_draft_model()
                or self.speculative_config.uses_extract_hidden_states()
            ):
''',
        '''        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_draft_model()
            or self.speculative_config.uses_extract_hidden_states()
        ):
''',
        '''        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_draft_model()
        ):
''',
    ):
        new = old.replace("if self.speculative_config and (", "if self.speculative_config and self.drafter is not None and (")
        s = sub(s, old, new)
    open(runner, "w").write(s); print("patched (pp gates)", runner)

# PP + async scheduling broadcasts sampled ids as [num_reqs, 1]; with drafts they are
# [num_reqs, K+1]. Use the synchronous PP path (patch_pp_spec_tokens) for dspark.
s = open(vcfg).read()
if "dsv41 dspark: async" in s:
    print("already patched (async)", vcfg)
else:
    s = sub(s, '''                self.scheduler_config.async_scheduling = False
            elif (
                self.speculative_config is not None
                and self.speculative_config.method not in get_args(EagleModelTypes)
''', '''                self.scheduler_config.async_scheduling = False
            elif (
                self.speculative_config is not None
                and self.speculative_config.method == "dspark"
                and self.parallel_config.pipeline_parallel_size > 1
            ):
                # dsv41 dspark: async scheduling under PP needs [num_reqs, 1] sampled
                # ids; the drafted [num_reqs, K+1] path is handled synchronously
                logger.warning_once(
                    "Async scheduling disabled for dspark under pipeline parallel."
                )
                self.scheduler_config.async_scheduling = False
            elif (
                self.speculative_config is not None
                and self.speculative_config.method not in get_args(EagleModelTypes)
''')
    open(vcfg, "w").write(s); print("patched (async)", vcfg)
