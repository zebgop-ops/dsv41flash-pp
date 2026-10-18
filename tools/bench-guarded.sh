#!/bin/bash
# Remaining quality benchmarks on the :8008 server, yielding to live traffic: before each step wait until vllm-front
# has seen no request for QUIET minutes; while a step runs, stop it (process group) as soon as a request arrives,
# then wait for quiet again and rerun it (generation steps resume from their per-item caches).
# usage: bench-guarded.sh <tag>      -> eval/*-<tag>-*, log on stdout
set -u; cd /home/r/dsv41-run
tag=$1; URL=http://localhost:8008; export DSV41_MODEL=DSv41Flash1017EXL3; QUIET=${QUIET:-10}
posts() { journalctl -u vllm-front --since "$1 min ago" --no-pager 2>/dev/null | grep -c POST; }
wait_quiet() { local said=0; while [ "$(posts $QUIET)" -gt 0 ]; do [ $said = 0 ] && echo "   waiting for $QUIET quiet minutes ($(date -u +%T))" && said=1; sleep 60; done; }
run_step() {   # run_step <name> <done-file> <command...>
  local name=$1 out=$2; shift 2
  while [ ! -s "$out" ]; do
    wait_quiet
    echo "== $name $(date -u +%T)"
    setsid bash -c "$*" & local pid=$!
    while kill -0 $pid 2>/dev/null; do
      sleep 20
      if [ "$(posts 1)" -gt 0 ]; then
        echo "   live traffic: stopping $name ($(date -u +%T))"; kill -TERM -- -$pid 2>/dev/null; sleep 5; kill -KILL -- -$pid 2>/dev/null
        wait $pid 2>/dev/null; continue 2
      fi
    done
    wait $pid; local rc=$?
    if [ $rc -ne 0 ] || [ ! -s "$out" ]; then echo "   $name failed (exit $rc)"; return 1; fi
  done
  echo "   $name done"
}
run_step "stdlib code perplexity" eval/$tag-stdlib.json \
  "python3 tools/quality.py ppl $URL eval/stdlib-code.txt 32768 8192 2>&1 | tail -1 > eval/$tag-stdlib.tmp && grep -q perplexity eval/$tag-stdlib.tmp && mv eval/$tag-stdlib.tmp eval/$tag-stdlib.json"
for b in humanevalplus mbppplus; do
  run_step "$b generate" eval/gen-$tag-$b.json \
    "DSV41_CACHE=eval/gen-$tag-$b.jsonl python3 tools/codeeval.py $URL $b eval/gen-$tag-$b.json 0 8 4096 2>&1 | tail -2"
  run_step "$b retry capped" eval/genr-$tag-$b.json \
    "DSV41_CACHE=eval/genr-$tag-$b.jsonl DSV41_RETRY=eval/gen-$tag-$b.json python3 tools/codeeval.py $URL $b eval/genr-$tag-$b.json 0 8 16384 2>&1 | tail -2"
  run_step "$b score (sandbox)" eval/score-$tag-$b.json \
    "tools/coderun.sh eval/genr-$tag-$b.json $b 30 > eval/score-$tag-$b.tmp 2>/dev/null && mv eval/score-$tag-$b.tmp eval/score-$tag-$b.json"
  python3 -c "
import json; d=json.load(open('eval/score-$tag-$b.json'))
print(f'   $tag $b: {d[\"passed\"]}/{d[\"n\"]} = {d[\"pass_rate\"]:.3f}')"
done
run_step "GPQA-Diamond" eval/gpqa-$tag.json "python3 tools/gpqa.py $URL $DSV41_MODEL eval/gpqa-$tag.json 2>&1 | grep -E 'SUMMARY|gpqa:'"
echo "== all done $(date -u +%T)"
