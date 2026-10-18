#!/bin/bash
# Quality on the running EXL3 server: MMLU-1000 (paired vs the 0909 base), then HumanEval+ / MBPP+.
set -u; cd /home/r/dsv41-run; URL=http://localhost:8008; export DSV41_MODEL=DSv41Flash1017EXL3; tag=${1:-exl3f}
echo "== $tag MMLU $(date -u +%T)"
DSV41_MC_DUMP=eval/mc-$tag-1000.json python3 tools/quality.py mc $URL eval/mmlu-test.json 1000 0 2>&1 | tail -1 > eval/$tag-mmlu.json
grep -o '"accuracy": [0-9.]*' eval/$tag-mmlu.json
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
echo "== $tag quality done $(date -u +%T)"
