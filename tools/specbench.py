#!/usr/bin/env python3
"""Decode-rate + spec-decode acceptance on two workloads: free prose and a context-heavy
code edit (where n-gram/prompt-lookup drafts hit). usage: specbench.py [url] [max_tokens]"""
import os, json, sys, time, urllib.request
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8004"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
SAVE = sys.argv[3] if len(sys.argv) > 3 else None        # save outputs to this json
COMPARE = sys.argv[4] if len(sys.argv) > 4 else None     # compare with outputs saved earlier
outputs = {}
CODE = '''import argparse, json, os, sys
from dataclasses import dataclass, field

@dataclass
class Job:
    name: str
    cmd: list[str]
    env: dict[str, str] = field(default_factory=dict)
    retries: int = 3
    timeout_s: float = 60.0

def load_jobs(path: str) -> list[Job]:
    with open(path) as f:
        raw = json.load(f)
    jobs = []
    for item in raw["jobs"]:
        jobs.append(Job(name=item["name"], cmd=item["cmd"], env=item.get("env", {}),
                        retries=item.get("retries", 3), timeout_s=item.get("timeout_s", 60.0)))
    return jobs

def run_job(job: Job) -> int:
    import subprocess
    env = dict(os.environ); env.update(job.env)
    for attempt in range(job.retries):
        try:
            proc = subprocess.run(job.cmd, env=env, timeout=job.timeout_s, capture_output=True, text=True)
            if proc.returncode == 0:
                return 0
            print(f"[{job.name}] attempt {attempt+1} failed rc={proc.returncode}: {proc.stderr[:200]}", file=sys.stderr)
        except subprocess.TimeoutExpired:
            print(f"[{job.name}] attempt {attempt+1} timed out after {job.timeout_s}s", file=sys.stderr)
    return 1

def main(argv=None):
    ap = argparse.ArgumentParser(description="run jobs from a json file")
    ap.add_argument("config")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(argv)
    jobs = load_jobs(args.config)
    if args.only:
        jobs = [j for j in jobs if j.name in set(args.only)]
    failures = sum(run_job(j) for j in jobs)
    print(f"{len(jobs) - failures}/{len(jobs)} jobs succeeded")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
'''
TASKS = {
    "prose": "Write a long, detailed essay about the history of lighthouses, their engineering, and their keepers.",
    "code-edit": "Here is a Python file:\n```python\n" + CODE + "```\nRewrite the whole file with one change: add a `--dry-run` flag that prints each job's command instead of running it. Output the complete updated file in a single code block, nothing else.",
}
def metrics():
    m = {}
    for line in urllib.request.urlopen(URL + "/metrics", timeout=10).read().decode().splitlines():
        for k in ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_accepted_tokens_total", "vllm:generation_tokens_total"):
            if line.startswith(k):
                m[k] = m.get(k, 0.0) + float(line.rsplit(" ", 1)[-1])
    return m
TASKS["code-edit"] = ("Here is a Python file:\n```python\n" + CODE + "```\nThe same file rewritten with one change, a `--dry-run` flag that prints each job's command instead of running it:\n```python\n")
for name, prompt in TASKS.items():
    m0 = metrics()
    if name == "code-edit":  # raw completion: the model continues straight into code (no reasoning)
        body = {"model": os.environ.get("DSV41_MODEL", "DSv41Flash"), "prompt": prompt, "max_tokens": N, "temperature": 0.0}
        req = urllib.request.Request(URL + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    else:
        body = {"model": os.environ.get("DSV41_MODEL", "DSv41Flash"), "messages": [{"role": "user", "content": prompt}], "max_tokens": N, "temperature": 0.0}
        req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); r = json.load(urllib.request.urlopen(req, timeout=1800)); dt = time.time() - t0
    m1 = metrics(); u = r["usage"]
    text = r["choices"][0].get("text") if name == "code-edit" else (r["choices"][0]["message"].get("content") or "")
    drafts = m1.get("vllm:spec_decode_num_drafts_total", 0) - m0.get("vllm:spec_decode_num_drafts_total", 0)
    dtok = m1.get("vllm:spec_decode_num_draft_tokens_total", 0) - m0.get("vllm:spec_decode_num_draft_tokens_total", 0)
    acc = m1.get("vllm:spec_decode_num_accepted_tokens_total", 0) - m0.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    spec = f"drafts {drafts:.0f} draft-tokens {dtok:.0f} accepted {acc:.0f} ({acc/max(dtok,1)*100:.0f}% of drafted, {acc/max(drafts,1):.2f}/step)" if drafts else "no spec"
    print(f"{name:10s}: {u['completion_tokens']} tokens in {dt:.1f}s = {u['completion_tokens']/dt:.1f} tok/s (incl. prefill of {u['prompt_tokens']} tokens) | {spec}", flush=True)
    print("   tail:", repr((text or "")[-120:]))
    full = (text or "") if name == "code-edit" else json.dumps({"reasoning": r["choices"][0]["message"].get("reasoning"), "content": r["choices"][0]["message"].get("content")})
    outputs[name] = full
if SAVE:
    json.dump(outputs, open(SAVE, "w"))
if COMPARE:
    ref = json.load(open(COMPARE))
    for k, v in outputs.items():
        same = ref.get(k) == v
        print(f"greedy output identical to {COMPARE} for {k}: {same}" + ("" if same else f"  (first diff at char {next((i for i,(a,b) in enumerate(zip(ref.get(k,''), v)) if a!=b), min(len(ref.get(k,'')), len(v)))})"))
