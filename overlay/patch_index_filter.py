#!/usr/bin/env python3
"""Hybrid checkpoints: take each tensor only from the shard the checkpoint's model.safetensors.index.json maps it
to (DSV41_INDEX_FILTER=<snapshot dir>). A hybrid directory links shards of two checkpoints that share tensor
names (norms, gates, attention in both; experts in different formats); without this the loader would read every
tensor of every listed file and the last one loaded would win. Tensors absent from the index are skipped too.
usage: <model_executor/model_loader/weight_utils.py>"""
import sys
p = sys.argv[1]; s = open(p).read()
EAGER_OLD = "                if not should_skip_weight(name, local_expert_ids):\n                    yield name, param\n"
EAGER_NEW = "                if not should_skip_weight(name, local_expert_ids) and not _dsv41_skip_file(name, st_file):\n                    yield name, param\n"
if "dsv41 index filter" in s:
    if s.count(EAGER_OLD) == 1:
        s = s.replace(EAGER_OLD, EAGER_NEW); open(p, "w").write(s); print("patched eager branch", p)
    else:
        print("already patched")
    sys.exit(0)
old_def = '''def _dsv41_skip(name: str) -> bool:
    if _DSV41_ONLY_RE is not None and _DSV41_ONLY_RE.search(name) is None:
        return True
    return _DSV41_SKIP_RE is not None and _DSV41_SKIP_RE.search(name) is not None
'''
new_def = old_def + '''

_DSV41_INDEX_MAP = None  # dsv41 index filter


def _dsv41_index_map():
    global _DSV41_INDEX_MAP
    d = _dsv41_os.environ.get("DSV41_INDEX_FILTER")
    if not d:
        return None
    if _DSV41_INDEX_MAP is None:
        import json as _json
        _DSV41_INDEX_MAP = _json.load(open(_dsv41_os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    return _DSV41_INDEX_MAP


def _dsv41_skip_file(name: str, st_file: str) -> bool:
    """True when the hybrid index maps `name` to a different shard (or not at all)."""
    m = _dsv41_index_map()
    return m is not None and m.get(name) != _dsv41_os.path.basename(st_file)
'''
assert s.count(old_def) == 1; s = s.replace(old_def, new_def)
old_use = "if should_skip_weight(name, local_expert_ids) or _dsv41_skip(name):"
assert s.count(old_use) == 2, s.count(old_use)
s = s.replace(old_use, "if should_skip_weight(name, local_expert_ids) or _dsv41_skip(name) or _dsv41_skip_file(name, st_file):")
assert s.count(EAGER_OLD) == 1, s.count(EAGER_OLD)
s = s.replace(EAGER_OLD, EAGER_NEW)
open(p, "w").write(s); print("patched", p)
