#!/bin/bash
# HumanEval+ / MBPP+ on the running DSv41Flash1017 server (:8007): generate at a 4k token budget,
# retry anything that hit the cap at 16k, score in the sandbox. Same procedure and tasks as every
# other model in eval/CROSS-MODEL.md. Waits for eval-1017.sh to finish first (one load at a time).
set -u
tag=${1:-1017}; URL=http://localhost:${DSV41_PORT:-8007}
export DSV41_MODEL=${DSV41_MODEL:-DSv41Flash1017}
cd /home/r/dsv41-run
until grep -q "== 1017 done\|server never came up\|container .* is gone" eval-1017-run.log 2>/dev/null; do sleep 30; done
grep -q "== 1017 done" eval-1017-run.log || { echo "eval-1017.sh did not finish cleanly"; exit 1; }
for b in humanevalplus mbppplus; do
  echo "== $b generate $(date -u +%T)"
  timeout 14400 python3 tools/codeeval.py "$URL" $b "eval/gen-$tag-$b.json" 0 8 4096 2>&1 | tail -1
  echo "== $b retry $(date -u +%T)"
  DSV41_RETRY="eval/gen-$tag-$b.json" timeout 7200 python3 tools/codeeval.py "$URL" $b "eval/genr-$tag-$b.json" 0 8 16384 2>&1 | tail -1
  tools/coderun.sh "eval/genr-$tag-$b.json" $b 30 > "eval/score-$tag-$b.json" 2>/dev/null
  python3 -c "
import json; d=json.load(open('eval/score-$tag-$b.json'))
print(f'  $tag $b: {d[\"passed\"]}/{d[\"n\"]} = {d[\"pass_rate\"]:.3f}')"
done
echo "== code eval done $(date -u +%T)"
