#!/bin/bash
# Post-boot evaluation of DSv41Flash1017 on :8007: wait for the server, smoke + top-5 logprobs,
# speed (specbench thinking on, streaming prose/code thinking off), perplexity (wikitext-2, stdlib
# code) and MMLU-1000 (same 1000 questions, seed 0, as every other model on this box), then the
# paired comparison against the 0909 base model. HumanEval+/MBPP+ run separately (coderun-all.sh).
# usage: eval-1017.sh [tag]   -> bench-<tag>.txt, lp-<tag>.json, spec-<tag>.json, eval/<tag>.txt, eval/mc-<tag>-1000.json
set -u
tag=${1:-1017}; URL=http://localhost:${DSV41_PORT:-8007}; NAME=${DSV41_NAME:-dsv41-1017-pp}
export DSV41_MODEL=${DSV41_MODEL:-DSv41Flash1017}
cd /home/r/dsv41-run
for i in $(seq 1 240); do curl -s -m 5 "$URL/health" -o /dev/null && break; docker ps --format '{{.Names}}' | grep -qx "$NAME" || { echo "container $NAME is gone"; exit 1; }; sleep 15; done
curl -s -m 5 "$URL/health" -o /dev/null || { echo "server never came up"; exit 1; }
echo "== $tag up $(date -u +%FT%T)"
docker logs "$NAME" 2>&1 | grep -h "Available KV cache memory\|GPU KV cache size" | sed 's/.*INFO//' | cut -c1-140
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "--- smoke"; curl -s -m 600 $URL/v1/chat/completions -H 'content-type: application/json' -d "{\"model\":\"$DSV41_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23? Answer with the number only.\"}],\"max_tokens\":400,\"temperature\":0,\"chat_template_kwargs\":{\"thinking\":false}}" | python3 -c "import json,sys;r=json.load(sys.stdin);print(repr(r['choices'][0]['message']['content'][:200]), r['usage'])"
echo "--- lpcheck"; python3 tools/lpcheck.py lp-$tag.json - $URL 2>&1 | tail -4
echo "--- specbench (thinking on, 400 tokens incl. prefill)"; python3 tools/specbench.py $URL 400 spec-$tag.json 2>&1 | grep -v "tail:" | tail -6
echo "--- itl prose (thinking off, streaming)"; python3 tools/itl.py $URL 400 | tail -2
echo "--- itl code edit (thinking off, streaming)"; python3 tools/itl.py $URL 400 "Here is a Python file:\n$(sed -n '/^CODE = /,/^'"'''"'$/p' tools/specbench.py | sed '1d;$d')\nRewrite the whole file with one change: add a --dry-run flag that prints each job's command instead of running it. Output the complete updated file in a single code block, nothing else." | tail -2
{
  echo "=== $tag $(date -u +%FT%T) ==="
  echo "--- wikitext-2 test perplexity"; python3 tools/quality.py ppl $URL eval/wikitext2-test.txt 73728 8192 2>&1 | tail -1
  echo "--- python stdlib code perplexity"; python3 tools/quality.py ppl $URL eval/stdlib-code.txt 2>&1 | tail -1
  echo "=== $tag MMLU 1000 paired $(date -u +%FT%T) ==="; DSV41_MC_DUMP=eval/mc-$tag-1000.json python3 tools/quality.py mc $URL eval/mmlu-test.json 1000 0 2>&1 | tail -1
} > eval/$tag.txt 2>&1
grep -o '"ppl": [0-9.]*\|"accuracy": [0-9.]*\|"mean_margin": [0-9.]*' eval/$tag.txt
echo "--- paired vs 0909 base"; python3 tools/qcompare.py eval/base.txt eval/$tag.txt eval/mc-base-1000.json eval/mc-$tag-1000.json 2>&1 | head -30
echo "== $tag done $(date -u +%FT%T)"
