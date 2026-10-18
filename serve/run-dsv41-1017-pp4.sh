#!/bin/bash
# DeepSeek-V4.1-Flash-1017 (mrfakename's abliterated build of deepseek-ai/DeepSeek-V4.1-Flash-1017):
# same architecture, config and tokenizer as the 0909 checkpoint, new weights (DeepSeek's "minor
# update": reasoning/coding/agentic gains, shorter outputs) with a rank-2 refusal subspace projected
# out of attn.wo_b and ffn.shared_experts.w2 in the 40 main layers (routed experts, embeddings,
# Engram tables and the DSpark layers are bit-identical to the official 1017). The repo ships its own
# Engram shards 47/48, so nothing needs linking. Full 384 experts per layer -> 6.72 GiB/layer, so the
# base model's layout applies: 9 cold layers' experts on the CPU, partition 8,12,8,12, n-gram K=3.
# Same overlay/launcher as the base model; own container name, port and served name.
set -u
export DSV41_HF_REPO=${DSV41_HF_REPO:-mrfakename/DeepSeek-V4.1-Flash-1017-abliterated}
export DSV41_NAME=${DSV41_NAME:-dsv41-1017-pp}
export DSV41_PORT=${DSV41_PORT:-8007}   # :8006 belongs to GLM53UM (glm53u-mm)
export DSV41_SERVED=${DSV41_SERVED:-DSv41Flash1017}
export DSV41_UTIL=${DSV41_UTIL:-0.93}     # cards driven to 61-63 GiB have crashed; keep ~4 GiB of slack
# At 0.97 the base layout (8,12,8,12 with 9 CPU expert layers 16-19,35-39) left rank 0 only 0.91 GiB of KV,
# i.e. ~61 GiB in use. At 0.93 (budget 60.9 GiB) a rank with 8 GPU expert layers cannot allocate a KV cache:
# measured after profiling, 10 CPU layers (7,16-19,35-39) gave 53.9 / 60.3 / 59.4 / 56.2 GiB per rank and
# vLLM found no room on ranks 1-2. So every rank keeps 7 GPU expert layers (12 cold layers on the CPU, ~3 ms
# more per decode step than the 9-layer base layout; CPU experts are DRAM-bound at ~2.4 ms per layer-step).
export DSV41_PARTITION=${DSV41_PARTITION:-8,12,8,12}
export DSV41_CPU_EXPERT_LAYERS=${DSV41_CPU_EXPERT_LAYERS:-7,15-19,27,35-39}
export DSV41_MAXLEN=${DSV41_MAXLEN:-262144}   # KV headroom is ~4 GiB per rank with 7 GPU expert layers (~1 KB/token/rank)
# Smaller prefill chunks halve the eager-prefill activation peak that the profiler reserves on every rank
# (~1 GiB each); prefill here is bound by the GPU attention path, not by chunk size.
export DSV41_BATCHED=${DSV41_BATCHED:-1088}   # must exceed max_tokens_per_mm_item (1025) for the image profiling pass
exec /home/r/dsv41-run/run-dsv41-pp4.sh "$@"
