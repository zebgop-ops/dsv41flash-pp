#!/bin/bash
# DeepSeek-V4.1-Flash-1017 as LibertAIDAI's EXL3 2.9 bpw pack: routed experts at 3 bits (layers 18-22 at 2),
# attention / shared experts / compressor / indexer at 5 bits, lm_head 6, everything else as DeepSeek stored it.
# 196 GiB, so all 40 layers' experts fit on the four cards (10,10,10,10; no CPU experts). The pack carries no
# Engram tables: they are read from the 1017 MXFP4 checkpoint (shards 47/48 are byte-identical between the
# official and the abliterated 1017 repos). Kernels: exllamav3 1.6.3 built for sm_80 (exl3/build-exllamav3.sh),
# wired in by overlay/exl3 + overlay/patch_exl3.py. Pack-reported quality: perplexity +0.4% prose / +1.8% code.
set -u
export DSV41_HF_REPO=${DSV41_HF_REPO:-LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw}
export DSV41_NAME=${DSV41_NAME:-dsv41-exl3-pp}
export DSV41_PORT=${DSV41_PORT:-8008}
export DSV41_SERVED=${DSV41_SERVED:-DSv41Flash1017EXL3}
export DSV41_EXL3=1
export DSV41_ENGRAM_MODEL_DIR=${DSV41_ENGRAM_MODEL_DIR:-/hf/hub/models--mrfakename--DeepSeek-V4.1-Flash-1017-abliterated/snapshots/517efab9ccfef7f714dbd7d679f181a86f43ccf9}
export DSV41_PARTITION=${DSV41_PARTITION:-10,10,10,10}
export DSV41_CPU_EXPERT_LAYERS=${DSV41_CPU_EXPERT_LAYERS:-none}
export DSV41_UTIL=${DSV41_UTIL:-0.91}       # at 0.93 ranks 0 and 3 sat at 60.1-60.5 GiB (cards at 61-63 have crashed); 0.91 still leaves ~4 GiB of KV per rank
export DSV41_MAXLEN=${DSV41_MAXLEN:-262144}
export DSV41_BATCHED=${DSV41_BATCHED:-1088}   # > max_tokens_per_mm_item (1025) for the image profiling pass
# n-gram speculation (K=3, prompt lookup min 5 / max 8) is on: with the deferred per-batch draft fetch
# (core.py, DSV41_DEFER_DRAFT), the PP input-buffer guard (patch_pp_input_event) and decodes spread over
# the micro-batches (patch_pp_spread, DSV41_PP_SPREAD) the four pipeline stages stay busy even though vLLM
# runs n-gram drafting synchronously. Measured vs DSV41_SPEC=0: code edits ~2x (57.8 vs 27.8 tok/s),
# 4 concurrent streams 72-82 vs 66-71 tok/s, 4 concurrent code edits 136-156 tok/s; free prose on one
# stream ~10% slower (each draft-less step pays the synchronous host round trip, ~6 ms).
export DSV41_SPEC=${DSV41_SPEC:-3}
# Prefix caching: a multi-turn chat or agent loop re-sends its whole history every turn; with the cache only the
# new turn is prefilled. Validated against no-cache outputs with tools/prefixtest.py (FINDINGS §11).
export DSV41_PREFIX_CACHE=${DSV41_PREFIX_CACHE:-1}
# Dense EXL3 linears (attention, shared experts, compressor, indexer; not the head) re-stored at load as
# Marlin W8A16 (8-bit, bf16 scales per 128 inputs): 1.46x faster at 1 token, 2.24x at 4, wikitext ppl
# -0.10% (CI includes 0) vs the EXL3 kernels; costs ~0.6 GiB of KV per rank. EXL3_DENSE_W8=0 turns it off.
export EXL3_DENSE_W8=${EXL3_DENSE_W8:-1}
exec /home/r/dsv41-run/run-dsv41-pp4.sh "$@"
