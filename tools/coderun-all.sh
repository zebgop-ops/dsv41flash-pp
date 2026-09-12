#!/bin/bash
# Drive HumanEval+ / MBPP+ across the servers on this box, one at a time: boot, generate at a
# 4k token budget, retry anything that hit the cap at 16k, score in the sandbox, move on.
# usage: coderun-all.sh <tag:launcher:port:model:container> ...
set -u
cd /home/r/dsv41-run
for spec in "$@"; do
  IFS=: read -r tag launcher port model container <<< "$spec"
  echo "===== $tag  $(date -u +%T)"
  for c in dsv41-pp dsv41reap-pp qwen38-pp glm53flash-pp; do
    [ "$c" = "$container" ] || docker stop -t 60 "$c" >/dev/null 2>&1
  done
  eval "$launcher" > "boot-code-$tag.log" 2>&1
  for i in $(seq 1 120); do curl -s -m 5 "localhost:$port/health" -o /dev/null && break; sleep 15; done
  curl -s -m 5 "localhost:$port/health" -o /dev/null || { echo "  $tag FAILED to come up"; continue; }
  echo "  up at $(date -u +%T)"
  for b in humanevalplus mbppplus; do
    DSV41_MODEL=$model timeout 10800 python3 tools/codeeval.py "http://localhost:$port" $b \
      "eval/gen-$tag-$b.json" 0 8 4096 2>&1 | tail -1
    DSV41_MODEL=$model DSV41_RETRY="eval/gen-$tag-$b.json" timeout 7200 python3 tools/codeeval.py \
      "http://localhost:$port" $b "eval/genr-$tag-$b.json" 0 8 16384 2>&1 | tail -1
    tools/coderun.sh "eval/genr-$tag-$b.json" $b 30 > "eval/score-$tag-$b.json" 2>/dev/null
    python3 -c "
import json; d=json.load(open('eval/score-$tag-$b.json'))
print(f'  $tag $b: {d[\"passed\"]}/{d[\"n\"]} = {d[\"pass_rate\"]:.3f}')"
  done
done
echo "===== all done $(date -u +%T)"
