#!/bin/bash
# Wait for the server, then run lpcheck + specbench (thinking on) + streaming prose (thinking off).
# usage: fullbench.sh <tag>   -> bench-<tag>.txt, spec-<tag>.json, lp-<tag>.json
tag=${1:?tag}; URL=${2:-http://localhost:8004}; cd /home/r/dsv41-run
until docker logs ${DSV41_NAME:-dsv41-pp} 2>&1 | grep -q "Application startup complete\|Traceback"; do sleep 5; done
{
  echo "== $tag  $(date -u +%FT%T)"
  docker exec ${DSV41_NAME:-dsv41-pp} env | grep -o "DSV41_SPEC[A-Z_]*=[^ ]*" | tr '\n' ' '; echo
  docker logs ${DSV41_NAME:-dsv41-pp} 2>&1 | grep -o "'speculative_config': {[^}]*}" | head -1
  python3 tools/lpcheck.py /home/r/dsv41-run/lp-$tag.json /home/r/dsv41-run/lp-eager.json 2>&1 | grep "dlogprob"
  python3 tools/specbench.py $URL 400 /home/r/dsv41-run/spec-$tag.json /home/r/dsv41-run/fdo-outputs.json 2>&1 | grep -v "tail:"
  python3 tools/itl.py $URL 400 | tail -3
  python3 tools/itl.py $URL 400 | tail -1
  python3 tools/itl.py $URL 400 "Here is a Python file:\n$(sed -n '/^CODE = /,/^'"'''"'$/p' tools/specbench.py | sed '1d;$d')\nRewrite the whole file with one change: add a --dry-run flag that prints each job's command instead of running it. Output the complete updated file in a single code block, nothing else." | tail -3
} > bench-$tag.txt 2>&1
cat bench-$tag.txt
