#!/bin/bash
# The REAP repos ship without the two Engram shards (47/48, 189 GiB, byte-identical to the base
# model's). Link the base checkpoint's shards into the REAP snapshot so the index resolves.
set -eu
HF=${DSV41_HF:-/home/r/.cache/huggingface}
BASE=$(ls -d "$HF"/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/*/ | head -1)
REAP=$(ls -d "$HF"/hub/models--LibertAIDAI--DeepSeek-V4.1-Flash-REAP-272E/snapshots/*/ | head -1)
for s in model-00047-of-00048.safetensors model-00048-of-00048.safetensors; do
  if [ -e "$REAP/$s" ]; then echo "present: $REAP$s"; continue; fi
  # relative link: the cache is bind-mounted at /hf inside the container, so an absolute
  # host path would not resolve there
  tgt=$(realpath --relative-to="$REAP" "$(readlink -f "$BASE/$s")")
  ln -s "$tgt" "$REAP/$s"; echo "linked: $REAP$s -> $tgt"
done
