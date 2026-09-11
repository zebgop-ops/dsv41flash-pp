#!/usr/bin/env python3
"""Raw completion with ~N filler tokens (no chat template). usage: plen.py N [url]"""
import json, sys, time, urllib.request
N = int(sys.argv[1]); URL = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8004"
words = ["alpha", "beta", "gamma", "delta", "sigma", "omega", "kappa", "theta"]
doc = " ".join(f"item {i} is {words[i % 8]} {i * 7 % 13}." for i in range(N // 6))
body = {"model": "DSv41Flash", "prompt": doc + "\nThe last item number mentioned above was", "max_tokens": 8, "temperature": 0.0}
req = urllib.request.Request(URL + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time()
try:
    r = json.load(urllib.request.urlopen(req, timeout=1800))
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()[:300]); sys.exit(2)
print(f"prompt_tokens={r['usage']['prompt_tokens']} text={r['choices'][0]['text']!r} time={time.time()-t0:.1f}s")
