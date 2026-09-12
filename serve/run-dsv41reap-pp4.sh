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
# Every expert on the GPUs. Cuts at 10 and 30/31 sit inside index groups, which the PP shadow
# plan covers by replaying the group's index source (8 on rank 1; 20+28 on rank 3) with top-k.
# DSpark (the default here: prose 34 tok/s, code 90+ tok/s incl. prefill) keeps its 8.4 GiB
# draft on rank 3, so that rank gets 9 layers and rank 2 takes 11 (10 x 4.76 GiB of experts
# + dense + head fit a 64 GiB card with ~6 GiB left for KV).
export DSV41_SPEC_METHOD=${DSV41_SPEC_METHOD:-dspark}
if [ "$DSV41_SPEC_METHOD" = dspark ]; then
  export DSV41_PARTITION=${DSV41_PARTITION:-10,10,11,9}
else
  export DSV41_PARTITION=${DSV41_PARTITION:-10,10,10,10}
fi
export DSV41_CPU_EXPERT_LAYERS=${DSV41_CPU_EXPERT_LAYERS:-none}
exec /home/r/dsv41-run/run-dsv41-pp4.sh "$@"
