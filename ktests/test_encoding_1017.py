#!/usr/bin/env python3
"""Replay DeepSeek's golden chat-encoding fixtures (encoding/tests/test_input_N.json ->
test_output_N.txt, shipped in the DeepSeek-V4.1-Flash-1017 repo) through vLLM's DeepSeek V4.1
tokenizer wrapper (apply_chat_template, tokenize=False). Runs inside the serving image with the
overlay tokenizer files mounted; prints PASS/FAIL per case and the first differing line.
usage: test_encoding_1017.py <fixtures dir> <tokenizer dir (HF snapshot)>"""
import copy, json, sys, difflib
from pathlib import Path
from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer

fix, tokdir = Path(sys.argv[1]), sys.argv[2]
tok = DeepseekV41Tokenizer.from_pretrained(tokdir)
ok = 0; cases = sorted(fix.glob("test_input_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
for inp in cases:
    n = inp.stem.split("_")[-1]
    data = json.load(open(inp))
    case = data if isinstance(data, dict) else {"messages": data}
    kw = {"tokenize": False, "thinking": (case.get("thinking_mode") or "chat") == "thinking"}
    if case.get("reasoning_effort") is not None: kw["reasoning_effort"] = case["reasoning_effort"]
    try:
        got = tok.apply_chat_template(copy.deepcopy(case["messages"]), tools=case.get("tools"), **kw)
    except Exception as e:  # noqa: BLE001
        got = f"<<exception {type(e).__name__}: {e}>>"
    want = (fix / f"test_output_{n}.txt").read_text()
    if got == want:
        ok += 1; print(f"case {n}: PASS")
    else:
        print(f"case {n}: FAIL")
        for line in list(difflib.unified_diff(want.splitlines(), got.splitlines(), "golden", "vllm", lineterm="", n=1))[:14]:
            print("   ", line[:160])
print(f"{ok}/{len(cases)} golden cases match")
sys.exit(0 if ok == len(cases) else 1)
