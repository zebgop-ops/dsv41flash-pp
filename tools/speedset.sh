#!/bin/bash
# Speed set for one running server: prose and code-edit streaming (thinking off), specbench (thinking on),
# and 4 concurrent prose streams. usage: speedset.sh <url> <model> <tag>   -> prints one block
URL=$1; export DSV41_MODEL=$2; tag=$3; cd /home/r/dsv41-run
echo "== $tag $(date -u +%T)"
echo -n "prose 400 (stream): "; timeout 600 python3 tools/itl.py $URL 400 2>&1 | tail -1 | grep -o "total [0-9]* chunks in [0-9.]*s"
CE="Here is a Python file:\n$(sed -n '/^CODE = /,/^'"'''"'$/p' tools/specbench.py | sed '1d;$d')\nRewrite the whole file with one change: add a --dry-run flag that prints each job's command instead of running it. Output the complete updated file in a single code block, nothing else."
echo -n "code edit 400 (stream): "; timeout 600 python3 tools/itl.py $URL 400 "$CE" 2>&1 | tail -1 | grep -o "total [0-9]* chunks in [0-9.]*s"
timeout 900 python3 tools/specbench.py $URL 400 2>&1 | grep -v "tail:" | grep "tok/s" | sed 's/^/specbench /'
S=$(mktemp -d); t0=$(date +%s.%N)
for i in 1 2 3 4; do (timeout 900 python3 tools/itl.py $URL 200 "Write a long story about lighthouse keeper number $i and the sea." 2>&1 | tail -1 > $S/$i) & done; wait
t1=$(date +%s.%N); echo "4 streams x 200 tokens: $(python3 -c "print(f'{$t1-$t0:.1f} s wall, {800/($t1-$t0):.1f} tok/s aggregate')")"; rm -rf $S
