#!/bin/bash
# DeepSeek-V4.1-Flash-1017 hybrid: LibertAIDAI's EXL3 2.9 bpw routed experts (the only part that has to shrink
# to fit four cards) with every other tensor exactly as the abliterated 1017 checkpoint stores it: FP8 attention,
# shared experts, indexer and Engram projections (Marlin MXFP8 on sm_80), bf16 compressor / indexer-wk / head.
# The abliteration only edited attn.wo_b and ffn.shared_experts.w2; the routed experts are bit-identical to the
# original, so EXL3 experts quantized from the original are the right experts for the abliterated model.
# Build the snapshot first (relative symlinks + merged index + config):
#   python3 -I tools/make-hybrid.py <abliterated snapshot> <EXL3 snapshot> \
#     ~/.cache/huggingface/hub/models--local--DSv41-1017-EXL3-hybrid/snapshots/v1
# DSV41_INDEX_FILTER makes the loader read each tensor only from the shard the merged index names.
set -u
export DSV41_HF_REPO=${DSV41_HF_REPO:-local/DSv41-1017-EXL3-hybrid}
export DSV41_INDEX_FILTER=${DSV41_INDEX_FILTER:-/hf/hub/models--local--DSv41-1017-EXL3-hybrid/snapshots/v1}
export DSV41_SERVED=${DSV41_SERVED:-DSv41Flash1017EXL3}
export DSV41_EXL3_DENSE=0      # no EXL3 dense linears: keep vLLM's aux streams and the shared-experts stream
# DSpark, the checkpoint's own drafter (mtp.0-2, original FP4/FP8 bytes, 8.4 GiB with its embedding), taps layers 37-39,
# so it lives on the last rank; 10,11,11,8 makes room for it (cuts at 10 and 21 sit inside index groups, covered by the
# PP shadow plan). Measured vs n-gram K=3 on this build: prose 31-32 vs 26-27 tok/s, code edit 68-69 vs 58-65,
# 4 streams 88-91 vs 82-85 tok/s aggregate; prefill 5-9% slower (11-layer ranks). DSV41_SPEC_METHOD=ngram goes back.
export DSV41_SPEC_METHOD=${DSV41_SPEC_METHOD:-dspark}
if [ "$DSV41_SPEC_METHOD" = dspark ]; then
  export DSV41_PARTITION=${DSV41_PARTITION:-10,11,11,8}
  export DSV41_SPEC=${DSV41_SPEC:-5}
fi
# 8 concurrent requests: with decodes spread over the 4 micro-batches (DSV41_PP_SPREAD) a step carries at most
# 2 requests x 6 verify rows, so the 4-request capture sizes cover it; the launcher's default for 8 (up to 48 rows)
# reserves 8.4 GiB of graph pool per rank and leaves no KV. 8 prose streams: 144 tok/s in one wave vs 87 in two.
export DSV41_SEQS=${DSV41_SEQS:-8}
if [ "$DSV41_SPEC_METHOD" = dspark ] && [ "$DSV41_SPEC" = 5 ]; then
  export DSV41_CG_SIZES=${DSV41_CG_SIZES:-[1,2,3,4,5,6,8,10,12,18,24]}
fi
exec /home/r/dsv41-run/run-dsv41-exl3-pp4.sh "$@"
