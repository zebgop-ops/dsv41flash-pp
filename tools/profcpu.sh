#!/bin/bash
# Extract the DSV41 PROFILE tables (device-time + self-CPU) per rank from the running
# container's log into <outdir>/rank<N>.txt. usage: profcpu.sh <outdir>
out=${1:?outdir}; mkdir -p "$out"
for r in 0 1 2 3; do
  docker logs dsv41-pp 2>&1 | grep "Worker_PP$r " | sed -n "/DSV41 PROFILE rank $r/,/^(Worker_PP$r [^)]*) [A-Z]* .*\[gpu_worker.py:[0-9]*\] [^ ]/p" \
    | grep "gpu_worker.py" | sed 's/^.*gpu_worker.py:[0-9]*\] //' > "$out/rank$r.txt"
  echo "rank $r: $(wc -l < "$out/rank$r.txt") lines; $(grep -m1 'self-cpu total' "$out/rank$r.txt" | grep -o 'self-cpu total.*')"
done
