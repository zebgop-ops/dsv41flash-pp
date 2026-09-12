#!/usr/bin/env python3
"""Stream a greedy prose completion and print tokens/s per 50-token window (spec decode
emits several tokens per chunk; rate is computed from usage-free token counting via
completion chunks' text length is unreliable, so we count SSE chunks' token ids via
logprobs=0? Simpler: use include_usage per chunk is not available -> count chunks and
report chunk rate plus final usage)."""
import os, json, sys, time, urllib.request
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8004"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
PROMPT = sys.argv[3] if len(sys.argv) > 3 else "Write a long, detailed essay about the history of lighthouses, their engineering, and their keepers."
body = {"model": os.environ.get("DSV41_MODEL", "DSv41Flash"), "messages": [{"role": "user", "content": PROMPT}], "max_tokens": N, "temperature": 0,
        "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": False}}
req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time(); last = t0; chunks = 0; win = []; usage = None; wstart = t0
for raw in urllib.request.urlopen(req, timeout=900):
    line = raw.decode().strip()
    if not line.startswith("data:"): continue
    payload = line[5:].strip()
    if payload == "[DONE]": break
    d = json.loads(payload)
    if d.get("usage"): usage = d["usage"]
    if d.get("choices") and (d["choices"][0].get("delta") or {}).get("content") is not None:
        chunks += 1; now = time.time()
        if chunks % 50 == 0:
            print(f"chunks {chunks-49:4d}-{chunks:4d}: {50/(now-wstart):5.1f} chunks/s  ({1e3*(now-wstart)/50:.0f} ms/chunk)"); wstart = now
print(f"total {chunks} chunks in {time.time()-t0:.1f}s; usage {usage}")
