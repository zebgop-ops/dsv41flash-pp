#!/usr/bin/env python3
"""GPQA-Diamond (198 questions) on a served model, same prompt / budget / concurrency as
glm53-run/eval/run_gpqa.py (the GLM53UM numbers): greedy, server-default reasoning, 65,536-token budget,
6 concurrent requests, final line "Answer: <letter>". Chat only, nothing is executed.
Resumable: each finished item is appended to <out>.jsonl and skipped on a rerun.
usage: gpqa.py <url> <model> <out.json> [data=/home/r/glm53-run/eval/data/gpqa_diamond.json]"""
import concurrent.futures as cf, json, os, re, sys, time, urllib.request

URL, MODEL, OUT = sys.argv[1] + "/v1/chat/completions", sys.argv[2], sys.argv[3]
DATA = sys.argv[4] if len(sys.argv) > 4 else "/home/r/glm53-run/eval/data/gpqa_diamond.json"
rows = json.load(open(DATA))["rows"]
CACHE = OUT + "l" if OUT.endswith(".json") else OUT + ".jsonl"

def chat(prompt, max_tokens=65536):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0}
    err = None
    for _ in range(3):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"}), timeout=7200))
            m = r["choices"][0]["message"]
            return (m.get("content") or ""), r["usage"]["completion_tokens"], r["choices"][0].get("finish_reason")
        except Exception as e:
            err = e; time.sleep(5)
    return f"__ERROR__ {err}", 0, "error"

def item(i, row):
    content, toks, fin = chat(row["question"] + "\n\nThink it through carefully, then finish with a final line of the form:\nAnswer: <letter>")
    m = re.findall(r"Answer:\s*\**\s*\(?([ABCD])\)?", content, re.I); pred = m[-1].upper() if m else None
    if pred is None:
        m = re.findall(r"\b([ABCD])\b", content[-80:]); pred = m[-1] if m else None
    gold = str(row["answer"]).strip()[0].upper()
    return {"i": i, "ok": pred == gold, "gold": gold, "pred": pred, "tokens": toks, "finish": fin}

res = {}
if os.path.exists(CACHE):
    for line in open(CACHE):
        if line.strip():
            d = json.loads(line); res[d["i"]] = d
todo = [(i, r) for i, r in enumerate(rows) if i not in res]
print(f"gpqa: {len(res)} cached, {len(todo)} to run", flush=True)
t = time.time()
with open(CACHE, "a") as cache, cf.ThreadPoolExecutor(6) as ex:
    for r in ex.map(lambda a: item(*a), todo):
        if r["finish"] != "error":
            cache.write(json.dumps(r) + "\n"); cache.flush()
        res[r["i"]] = r
        if len(res) % 20 == 0:
            v = list(res.values())
            print(f"  gpqa: {len(v)}/{len(rows)} acc so far {sum(x['ok'] for x in v)/len(v):.3f} "
                  f"(truncated {sum(x['finish']=='length' for x in v)})", flush=True)
v = [res[i] for i in sorted(res)]
json.dump(v, open(OUT, "w"), indent=1)
acc = sum(x["ok"] for x in v) / len(v); toks = sum(x["tokens"] for x in v)
print(f"SUMMARY {MODEL} | gpqa_diamond: {sum(x['ok'] for x in v)}/{len(v)} = {acc:.3f} | truncated "
      f"{sum(x['finish']=='length' for x in v)} | errors {sum(x['finish']=='error' for x in v)} | unparsed "
      f"{sum(x['pred'] is None for x in v)} | {toks} completion tokens ({toks//len(v)} avg) in {time.time()-t:.0f}s this run", flush=True)
