#!/bin/bash
# N concurrent prose streams of 200 tokens each (thinking off): aggregate throughput and slowest stream.
# usage: streams.sh <url> <model> <n> [tokens]
URL=$1; export DSV41_MODEL=$2; N=$3; T=${4:-200}; cd /home/r/dsv41-run
S=$(mktemp -d); t0=$(date +%s.%N)
for i in $(seq 1 $N); do (timeout 1200 python3 tools/itl.py $URL $T "Write a long story about lighthouse keeper number $i and the sea." 2>&1 | tail -1 > $S/$i) & done; wait
t1=$(date +%s.%N)
echo "$N streams x $T tokens: $(python3 -c "print(f'{$t1-$t0:.1f} s wall, {$N*$T/($t1-$t0):.1f} tok/s aggregate')") | per stream: $(grep -ho 'in [0-9.]*s' $S/* | tr '\n' ' ')"; rm -rf $S
