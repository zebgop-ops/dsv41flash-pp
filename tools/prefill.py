#!/usr/bin/env python3
"""Prefill throughput vs prompt length: one request per length, max_tokens=1, thinking off.
usage: prefill.py [url] [lengths, comma-separated tokens]"""
import json, os, sys, time, urllib.request
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8004"
LENS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "512,2048,8192,16384,29000").split(",")]
MODEL = os.environ.get("DSV41_MODEL", "DSv41Flash")
para = ("The lighthouse keeper logged the weather every hour: wind from the northwest, seas moderate, "
        "visibility fair, lamp trimmed and burning steady through the night watch. ")
for n in LENS:
    text = (para * (n // 40 + 2))[: n * 4]  # ~4 chars/token
    body = {"model": MODEL, "messages": [{"role": "user", "content": text + "\n\nReply with the single word OK."}],
            "max_tokens": 1, "temperature": 0, "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time(); r = json.loads(urllib.request.urlopen(req, timeout=1800).read()); dt = time.time() - t
    p = r["usage"]["prompt_tokens"]
    print(f"prompt {p:6d} tokens: {dt:6.1f} s  = {p/dt:6.0f} tok/s  ({1000*dt/(p/2048):.0f} ms per 2048-token chunk)")
