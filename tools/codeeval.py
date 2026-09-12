#!/usr/bin/env python3
"""Generate HumanEval+ / MBPP+ solutions from a served model (stdlib only).

Generation only: nothing here executes model output. `tools/coderun.py` runs the
solutions inside a locked-down container. Each model is asked exactly the same way and
left to think as much as its own template wants, so this measures the server as deployed.

usage: codeeval.py <url> humanevalplus|mbppplus <out.json> [n] [concurrency] [max_tokens]
env:   DSV41_MODEL (served model name)
"""
import concurrent.futures as cf
import json, os, re, sys, time, urllib.request

MODEL = os.environ.get("DSV41_MODEL", "DSv41Flash")
INSTR = ("Output only the complete function in a single ```python code block, including any "
         "imports it needs. No tests, no explanation, no example usage.")


def build_prompt(row, bench):
    if bench == "humanevalplus":
        return (f"Complete this Python function.\n\n```python\n{row['prompt']}```\n\n{INSTR}")
    first_test = (row.get("test_list") or [""])[0]
    return (f"{row['prompt'].strip()}\n\nYour function must satisfy this test:\n"
            f"{first_test}\n\n{INSTR}")


def extract_code(text):
    """Last ```python block, else the last fenced block, else the raw text."""
    if "</think>" in text:                      # some templates leave the trace inline
        text = text.rsplit("</think>", 1)[1]
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return blocks[-1]
    return text


def generate(url, row, bench, max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": build_prompt(row, bench)}],
            "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    msg = r["choices"][0]["message"]
    content = msg.get("content") or ""
    return {"task_id": row["task_id"], "code": extract_code(content),
            "finish_reason": r["choices"][0].get("finish_reason"),
            "completion_tokens": r["usage"]["completion_tokens"],
            "reasoning_chars": len(msg.get("reasoning") or ""),
            "seconds": round(time.time() - t0, 1)}


def main():
    url, bench, out = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    conc = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    max_tokens = int(sys.argv[6]) if len(sys.argv) > 6 else 4096
    rows = json.load(open(f"eval/{bench}.json"))
    if n:
        rows = rows[:n]
    # DSV41_RETRY=<earlier run>: regenerate only the tasks that hit the token cap, at the
    # larger budget given here, then merge. Every model gets the same treatment, so a model
    # that thinks past the cap is not scored as a failure for it.
    prev_path = os.environ.get("DSV41_RETRY")
    prev = None
    if prev_path:
        prev = json.load(open(prev_path))
        stuck = {str(d["task_id"]) for d in prev["solutions"]
                 if d.get("finish_reason") == "length" or d.get("error")}
        rows = [r for r in rows if str(r["task_id"]) in stuck]
        print(f"retrying {len(rows)} capped/failed tasks at max_tokens={max_tokens}")
        if not rows:
            json.dump(prev, open(out, "w"), indent=1)
            print(f"nothing to retry; copied {prev_path} -> {out}")
            return
    t0 = time.time()
    done = []
    with cf.ThreadPoolExecutor(conc) as ex:
        futs = {ex.submit(generate, url, r, bench, max_tokens): r["task_id"] for r in rows}
        for k, f in enumerate(cf.as_completed(futs)):
            try:
                done.append(f.result())
            except Exception as e:
                done.append({"task_id": futs[f], "code": "", "error": str(e)[:200]})
            if (k + 1) % 25 == 0 or k + 1 == len(rows):
                print(f"  {k+1:4d}/{len(rows)} generated ({time.time()-t0:5.0f} s)", flush=True)
    if prev:
        fixed = {str(d["task_id"]): d for d in done}
        done = [fixed.get(str(d["task_id"]), d) for d in prev["solutions"]]
        print(f"merged {len(fixed)} retried solutions into {len(done)}")
    done.sort(key=lambda d: str(d["task_id"]))
    json.dump({"model": MODEL, "bench": bench, "seconds": round(time.time() - t0, 1),
               "solutions": done}, open(out, "w"), indent=1)
    tok = sum(d.get("completion_tokens", 0) for d in done)
    trunc = sum(1 for d in done if d.get("finish_reason") == "length")
    print(f"wrote {out}: {len(done)} solutions, {tok} completion tokens, "
          f"{trunc} hit the token cap, {time.time()-t0:.0f} s wall")


if __name__ == "__main__":
    main()
