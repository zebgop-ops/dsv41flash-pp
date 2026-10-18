#!/usr/bin/env python3
"""GSM8K (first 200 test items) exactly as glm53-run/eval/run_eval.py scores it: same prompt, '####' answer
parsing, greedy, max_tokens 12000, server-default reasoning, concurrency 6. GSM8K only (run_eval.py's HumanEval
half executes model code on the host). usage: gsm8k.py <tag> <url> <model>  -> eval/gsm8k-<tag>.json"""
import json, re, sys, time, urllib.request, concurrent.futures as cf
tag, base, MODEL = sys.argv[1], sys.argv[2], sys.argv[3]
URL = base + "/v1/chat/completions"
DATA = "/home/r/glm53-run/eval/data/gsm8k_200.json"
def chat(prompt, max_tokens=12000):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0}
    err = None
    for _ in range(3):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}), timeout=3600))
            m = r["choices"][0]["message"]
            return (m.get("content") or ""), r["usage"]["completion_tokens"], r["choices"][0].get("finish_reason")
        except Exception as e:  # noqa: BLE001
            err = e; time.sleep(5)
    return f"__ERROR__ {err}", 0, "error"
def num(s):
    s = s.replace(",", "").replace("$", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s); return m[-1] if m else None
def item(i, row):
    content, toks, fin = chat(row["question"] + "\n\nSolve step by step, then give the final answer as a number on the last line in the form: #### <number>")
    gold = num(row["answer"].split("####")[-1]); tail = content.split("####")[-1] if "####" in content else content[-200:]
    pred = num(tail)
    ok = pred is not None and gold is not None and abs(float(pred) - float(gold)) < 1e-6
    return {"i": i, "ok": ok, "gold": gold, "pred": pred, "tokens": toks, "finish": fin}
data = json.load(open(DATA)); t = time.time(); res = []
with cf.ThreadPoolExecutor(6) as ex:
    for r in ex.map(lambda a: item(*a), enumerate(data)):
        res.append(r)
        if len(res) % 20 == 0:
            print(f"  gsm8k: {len(res)}/{len(data)} acc so far {sum(x['ok'] for x in res)/len(res):.3f}", flush=True)
json.dump(res, open(f"/home/r/dsv41-run/eval/gsm8k-{tag}.json", "w"), indent=1)
acc = sum(x["ok"] for x in res) / len(res)
print(f"SUMMARY {tag} {MODEL} | gsm8k: {sum(x['ok'] for x in res)}/{len(res)} = {acc:.3f} | truncated "
      f"{sum(x['finish']=='length' for x in res)} | errors {sum(x['finish']=='error' for x in res)} | "
      f"{sum(x['tokens'] for x in res)} completion tokens in {time.time()-t:.0f}s", flush=True)
