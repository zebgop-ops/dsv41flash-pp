#!/bin/bash
# DeepSeek-V4.1-Flash-REAP-272E (LibertAIDAI): the base model with its 384 routed experts per layer
# pruned to 272 (REAP), same dense weights, same Engram tables (shards 47/48 must be linked in
# from the base checkpoint: see link-reap-engram.sh). 4.76 GiB of experts per layer instead of
# 6.72, so all 40 layers' experts fit on the GPUs with a 10,10,10,10 partition (none on the CPU).
# Same overlay/launcher as the base model; own container name, port and served name.
set -u
export DSV41_HF_REPO=${DSV41_HF_REPO:-LibertAIDAI/DeepSeek-V4.1-Flash-REAP-272E}
export DSV41_NAME=${DSV41_NAME:-dsv41reap-pp}
export DSV41_PORT=${DSV41_PORT:-8005}
export DSV41_SERVED=${DSV41_SERVED:-DSv41ReapFlash}
# Every expert on the GPUs: 10 layers per rank (~52 GiB), no CPU expert layers, utilization 0.93
# so every card keeps ~5 GiB of slack (cards run to the top have crashed). Cuts at 10 and 30 sit
# inside index groups; the PP shadow plan covers them by replaying the group's index source
# (8 on rank 1; 20+28 on rank 3) with top-k. Drafting is n-gram prompt lookup: the model's own
# DSpark drafter (DSV41_SPEC_METHOD=dspark, 8.4 GiB on the last rank) only fits with that rank's
# last two layers' experts on the CPU (DSV41_CPU_EXPERT_LAYERS=38-39), which we do not do by default.
export DSV41_SPEC_METHOD=${DSV41_SPEC_METHOD:-ngram}
export DSV41_PARTITION=${DSV41_PARTITION:-10,10,10,10}
if [ "$DSV41_SPEC_METHOD" = dspark ]; then
  export DSV41_CPU_EXPERT_LAYERS=${DSV41_CPU_EXPERT_LAYERS:-38-39}
else
  export DSV41_CPU_EXPERT_LAYERS=${DSV41_CPU_EXPERT_LAYERS:-none}
fi
export DSV41_UTIL=${DSV41_UTIL:-0.93}       # ~3.5 GiB of slack per card (61-63 GiB used has crashed before)
export DSV41_MAXLEN=${DSV41_MAXLEN:-524288}   # 512k; the KV pool holds several of these
exec /home/r/dsv41-run/run-dsv41-pp4.sh "$@"
