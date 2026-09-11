#!/usr/bin/env python3
"""Keep prefill-shaped batches out of PIECEWISE graphs (run them eagerly). Decode-shaped
batches (max query len <= 1 + num_spec_tokens) still use piecewise graphs; uniform decode
batches use FULL graphs. Small padded prefills in piecewise graphs were the one remaining
shape that did not reproduce eager numerics on this stack. DSV41_PREFILL_EAGER=0 disables.
usage: <vllm/v1/worker/gpu_model_runner.py>"""
import sys
path = sys.argv[1]; src = open(path).read()
if "dsv41: prefill-shaped batches" in src:
    print("already patched"); sys.exit(0)
old = '''        cudagraph_mode, batch_descriptor = dispatch_cudagraph(
            num_tokens_padded, disable_full=use_cascade_attn or has_encoder_output
        )
        num_tokens_padded = batch_descriptor.num_tokens
'''
new = '''        cudagraph_mode, batch_descriptor = dispatch_cudagraph(
            num_tokens_padded, disable_full=use_cascade_attn or has_encoder_output
        )
        # dsv41: prefill-shaped batches run eagerly instead of in piecewise graphs
        if (
            cudagraph_mode == CUDAGraphMode.PIECEWISE
            and force_uniform_decode is None  # real steps only (capture/dummy runs force it)
            and not force_eager
            and os.environ.get("DSV41_PREFILL_EAGER", "1") == "1"
            and max_num_scheduled_tokens > self.uniform_decode_query_len
        ):
            cudagraph_mode, batch_descriptor = dispatch_cudagraph(
                num_tokens_padded, valid_modes={CUDAGraphMode.NONE}
            )
        if os.environ.get("DSV41_DEBUG_SPEC") == "1" and force_uniform_decode is None and num_tokens <= 8:
            logger.info("DSV41 DISPATCH tokens=%d reqs=%d maxq=%d uniform=%s -> %s %s",
                        num_tokens, num_reqs, max_num_scheduled_tokens, uniform_decode,
                        cudagraph_mode, batch_descriptor)
        num_tokens_padded = batch_descriptor.num_tokens
'''
assert src.count(old) == 1, src.count(old)
src = src.replace(old, new, 1)
if "\nimport os\n" not in src:
    src = src.replace("\nimport torch\n", "\nimport os\nimport torch\n", 1)
open(path, "w").write(src); print("patched", path)
