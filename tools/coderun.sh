#!/bin/bash
# Execute model-generated code in a throwaway container: no network, no host filesystem
# beyond a read-only input dir, tmpfs for scratch, memory/CPU/pid caps, unprivileged user.
# usage: coderun.sh <solutions.json> <humanevalplus|mbppplus> [timeout_s] > results.json
set -eu
SOL=$(realpath "$1"); BENCH=$2; TMO=${3:-30}
IMG=${DSV41_IMG:-vllm/vllm-openai:deepseekv41-flash-0909}   # already local, has numpy
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
cp "$SOL" "$WORK/solutions.json"
cp "/home/r/dsv41-run/eval/$BENCH.json" "$WORK/$BENCH.json"
cp /home/r/dsv41-run/tools/coderun.py "$WORK/coderun.py"
chmod -R a+rX "$WORK"   # the sandbox runs as nobody
docker run --rm --network none --memory 4g --cpus 4 --pids-limit 512 \
  --security-opt no-new-privileges --cap-drop ALL --user 65534:65534 \
  --tmpfs /tmp:rw,exec,size=512m,mode=1777 \
  -v "$WORK:/in:ro" --entrypoint python3 "$IMG" /in/coderun.py "$BENCH" "$TMO" \
  | sed -n 's/^RESULT //p'
