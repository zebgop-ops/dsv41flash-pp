#!/usr/bin/env python3
"""Execute generated solutions against the EvalPlus tests. RUNS INSIDE THE SANDBOX ONLY.

Reads /in/solutions.json and /in/<bench>.json, runs each solution + its test file as a
separate subprocess with a wall-clock timeout, prints one JSON result line to stdout.
The host wrapper (coderun.sh) provides the isolation: no network, tmpfs, read-only input,
memory/pid caps.
"""
import json, subprocess, sys, tempfile, os

bench = sys.argv[1]
timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 30
sols = json.load(open("/in/solutions.json"))["solutions"]
rows = {str(r["task_id"]): r for r in json.load(open(f"/in/{bench}.json"))}

results = []
for s in sols:
    tid = str(s["task_id"])
    row = rows[tid]
    code = s.get("code") or ""
    parts = [code, "\n\n"]
    if bench == "mbppplus":
        parts = ["\n".join(row.get("test_imports") or []), "\n", code, "\n\n", row["test"]]
    else:
        parts = [code, "\n\n", row["test"], f"\ncheck({row['entry_point']})\n"]
    script = "".join(parts)
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir="/tmp", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout,
                           cwd="/tmp", env={"PATH": "/usr/bin:/bin", "HOME": "/tmp",
                                            "PYTHONDONTWRITEBYTECODE": "1"})
        ok = p.returncode == 0
        err = "" if ok else (p.stderr[-300:].decode(errors="replace"))
    except subprocess.TimeoutExpired:
        ok, err = False, "TIMEOUT"
    except Exception as e:
        ok, err = False, f"RUNNER: {e}"
    finally:
        os.unlink(path)
    results.append({"task_id": tid, "pass": ok, "error": err})

passed = sum(r["pass"] for r in results)
print("RESULT " + json.dumps({"bench": bench, "n": len(results), "passed": passed,
                              "pass_rate": passed / max(1, len(results)),
                              "results": results}))
