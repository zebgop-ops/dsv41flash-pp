#!/usr/bin/env python3
"""Skip checkpoint tensors matching DSV41_SKIP_WEIGHT_RE before safetensors touches them
(the 95 GiB Engram tables in SSD mode, the CPU layers' experts), and name the tensor in
any get_tensor failure. usage: patch_weight_iter.py <weight_utils.py>"""
import sys
path = sys.argv[1]; src = open(path).read()
if "DSV41_SKIP_WEIGHT_RE" in src:
    print("already patched"); sys.exit(0)
old = '''def safetensors_weights_iterator(
'''
new = '''import os as _dsv41_os
import re as _dsv41_re

_DSV41_SKIP_RE = (
    _dsv41_re.compile(_dsv41_os.environ["DSV41_SKIP_WEIGHT_RE"])
    if _dsv41_os.environ.get("DSV41_SKIP_WEIGHT_RE")
    else None
)


def _dsv41_skip(name: str) -> bool:
    return _DSV41_SKIP_RE is not None and _DSV41_SKIP_RE.search(name) is not None


def _dsv41_get_tensor(f, name: str):
    try:
        return f.get_tensor(name)
    except Exception as e:  # name the culprit
        raise RuntimeError(f"safetensors get_tensor failed for {name!r}: {e}") from e


def safetensors_weights_iterator(
'''
assert old in src; src = src.replace(old, new, 1)
old2 = '''                    if should_skip_weight(name, local_expert_ids):
                        continue
                    param = f.get_tensor(name)
                    yield name, param
'''
new2 = '''                    if should_skip_weight(name, local_expert_ids) or _dsv41_skip(name):
                        continue
                    param = _dsv41_get_tensor(f, name)
                    yield name, param
'''
assert old2 in src; src = src.replace(old2, new2, 1)
old3 = '''                    if should_skip_weight(name, local_expert_ids):
                        continue
                    state_dict[name] = f.get_tensor(name)
'''
new3 = '''                    if should_skip_weight(name, local_expert_ids) or _dsv41_skip(name):
                        continue
                    state_dict[name] = _dsv41_get_tensor(f, name)
'''
assert old3 in src; src = src.replace(old3, new3, 1)
open(path, "w").write(src); print("patched", path)
