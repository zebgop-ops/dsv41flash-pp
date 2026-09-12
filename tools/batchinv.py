"""Send the lpcheck prompts concurrently (one prefill batch) and compare position-0 top-5
logprobs with the sequential eager reference. usage: batchinv.py <ref.json> [url]"""
import os, json, sys, urllib.request, concurrent.futures as cf
REF = json.load(open(sys.argv[1])); URL = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8004"
PROMPTS = list(REF.keys())
def one(p):
    body = {"model": os.environ.get("DSV41_MODEL", "DSv41Flash"), "prompt": p, "max_tokens": 3, "temperature": 0.0, "logprobs": 5}
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600))
    return p, r["choices"][0]
with cf.ThreadPoolExecutor(3) as ex:
    res = dict(ex.map(one, PROMPTS))
for p in PROMPTS:
    lp = {k: round(v, 3) for k, v in res[p]["logprobs"]["top_logprobs"][0].items()}
    ref = REF[p]["top"][0]; shared = set(lp) & set(ref)
    worst = max((abs(lp[k]-ref[k]) for k in shared), default=float("nan"))
    print(f"{p[:26]!r}: concurrent pos0 vs sequential eager: shared {len(shared)}/5 max|d| {worst:.3f}")
