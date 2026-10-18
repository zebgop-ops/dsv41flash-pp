#!/usr/bin/env python3
"""Prefix-cache correctness and TTFT probe. Same ~6k-token document, several requests that share it:
P1 = doc + Q1, P2 = doc + Q2 (shares the doc prefix), P1 again (full hit), and a multi-turn follow-up
(doc + Q1, the assistant's answer, Q3). Greedy, thinking off, top-5 logprobs on the first tokens.
usage: prefixtest.py <url> <model> <save.json> [compare.json]"""
import json, sys, time, urllib.request
URL, MODEL, SAVE = sys.argv[1], sys.argv[2], sys.argv[3]
CMP = sys.argv[4] if len(sys.argv) > 4 else None
doc = open("/home/r/dsv41-run/eval/stdlib-code.txt").read()[:24000]
def chat(messages, n=120):
    body = {"model": MODEL, "messages": messages, "max_tokens": n, "temperature": 0, "logprobs": True, "top_logprobs": 5,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": False}}
    t0 = time.time(); ttft = None; text = ""; lps = []; usage = None
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            for ch in d.get("choices", []):
                c = ch.get("delta", {}).get("content")
                if c:
                    if ttft is None: ttft = time.time() - t0
                    text += c
                lp = (ch.get("logprobs") or {}).get("content") or []
                for e in lp:
                    if len(lps) < 8: lps.append({t["token"]: round(t["logprob"], 3) for t in e["top_logprobs"]})
    return {"text": text, "ttft": ttft, "total": time.time() - t0, "prompt_tokens": (usage or {}).get("prompt_tokens"), "top": lps}
sysmsg = {"role": "system", "content": "You are a careful code reviewer."}
P1 = [sysmsg, {"role": "user", "content": "Here is some Python source:\n\n" + doc + "\n\nWhich module does the first function come from? Answer in two sentences."}]
P2 = [sysmsg, {"role": "user", "content": "Here is some Python source:\n\n" + doc + "\n\nList three functions defined above, one per line."}]
out = {}
out["P1"] = chat(P1)
out["P2"] = chat(P2)
out["P1_again"] = chat(P1)
out["multiturn"] = chat(P1 + [{"role": "assistant", "content": out["P1"]["text"]}, {"role": "user", "content": "Now name one class defined above."}])
json.dump(out, open(SAVE, "w"), indent=1)
for k, v in out.items():
    print(f"{k:10s} prompt {v['prompt_tokens']} tok, TTFT {v['ttft']:.2f}s, total {v['total']:.2f}s, text {v['text'][:70]!r}")
if CMP:
    ref = json.load(open(CMP))
    for k in out:
        a, b = out[k], ref[k]
        same = a["text"] == b["text"]
        i = next((j for j, (x, y) in enumerate(zip(a["text"], b["text"])) if x != y), min(len(a["text"]), len(b["text"])))
        d = max((abs(a["top"][p][t] - b["top"][p][t]) for p in range(min(len(a["top"]), len(b["top"])))
                 for t in a["top"][p] if t in b["top"][p]), default=0)
        print(f"{k:10s} vs reference: text {'IDENTICAL' if same else f'differs at char {i}'}; max |dlogprob| on shared top tokens {d:.3f}")
