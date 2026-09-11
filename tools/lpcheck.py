"""Top-5 logprobs at the first 3 generated positions for a fixed prompt set; save/compare.
usage: lpcheck.py <save.json> [compare.json] [url]"""
import json, sys, urllib.request
SAVE = sys.argv[1]; CMP = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
URL = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8004"
PROMPTS = ["The capital of France is", "Q: What is the capital of France?\nA:", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"]
out = {}
for p in PROMPTS:
    body = {"model": "DSv41Flash", "prompt": p, "max_tokens": 3, "temperature": 0.0, "logprobs": 5}
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600))
    c = r["choices"][0]
    out[p] = {"text": c["text"], "top": [{k: round(v, 3) for k, v in lp.items()} for lp in c["logprobs"]["top_logprobs"]]}
    print(repr(p[:30]), "->", repr(c["text"]), "| pos0 top:", sorted(out[p]["top"][0].items(), key=lambda kv: -kv[1])[:3])
json.dump(out, open(SAVE, "w"), indent=1)
if CMP:
    ref = json.load(open(CMP))
    for p in PROMPTS:
        a, b = ref[p], out[p]
        worst = 0.0
        for la, lb in zip(a["top"], b["top"]):
            for k in set(la) & set(lb):
                worst = max(worst, abs(la[k] - lb[k]))
        print(f"vs {CMP}: text {'same' if a['text']==b['text'] else 'DIFFERENT'}; max |dlogprob| on shared top tokens {worst:.3f}  ({p[:25]!r})")
