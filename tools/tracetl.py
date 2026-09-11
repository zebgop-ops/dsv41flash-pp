#!/usr/bin/env python3
"""Print the cross-rank timeline of traced steps (DSV41_DEBUG_TRACE=1) relative to the
engine's execute_model issue time. usage: tracetl.py <trace.txt>"""
import re, sys, collections
core = {}; ranks = collections.defaultdict(dict)
for line in open(sys.argv[1]):
    m = re.match(r"core step (\d+) T=(\d+) exec_issue=([\d.]+)", line)
    if m: core[int(m[1])] = float(m[3]); continue
    m = re.match(r"rank (\d+) step (\d+) T=(\d+) (.*)", line)
    if m:
        kv = dict((k, float(v)) for k, v in re.findall(r"(\w+)=([\d.]+)", m[4]))
        ranks[int(m[2])][int(m[1])] = (int(m[3]), kv)
for step in sorted(ranks):
    t0 = core.get(step)
    print(f"step {step}" + ("" if t0 else " (no core stamp)"))
    for r in sorted(ranks[step]):
        T, kv = ranks[step][r]
        base = t0 if t0 else kv["in"]
        print(f"  rank {r} T={T}: " + " ".join(f"{k}={1e3*(v-base):7.1f}" for k, v in kv.items()))
