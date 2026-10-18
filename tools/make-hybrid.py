#!/usr/bin/env python3
"""Build a hybrid DeepSeek-V4.1-Flash-1017 snapshot: the EXL3 pack's routed experts for the 40 main layers,
everything else (attention, shared experts, compressor, indexer, Engram, head, MTP/DSpark layers) exactly as the
FP8/FP4 base checkpoint stores it. The directory holds only relative symlinks into the two HF snapshots plus a
merged model.safetensors.index.json and a config.json whose quantization_config is EXL3 with
scope=routed_experts_only. Run the server with DSV41_INDEX_FILTER=<this dir as a container path> so every tensor
is read from the shard the merged index names (both checkpoints share norm/gate/embed names).
usage: make-hybrid.py <base snapshot> <exl3 snapshot> <out snapshot dir>"""
import json, os, re, sys

base, exl3, out = (os.path.abspath(p) for p in sys.argv[1:4])
N_MAIN = 40
ROUTED = re.compile(r"^layers\.(\d+)\.ffn\.experts\.\d+\.")

bi = json.load(open(os.path.join(base, "model.safetensors.index.json")))["weight_map"]
ei = json.load(open(os.path.join(exl3, "model.safetensors.index.json")))["weight_map"]
assert not set(bi.values()) & set(ei.values()), "shard names collide between the two checkpoints"

def main_routed(name):
    m = ROUTED.match(name)
    return m is not None and int(m.group(1)) < N_MAIN

wm = {k: v for k, v in bi.items() if not main_routed(k)}
exl3_part = {k: v for k, v in ei.items() if main_routed(k)}
assert exl3_part, "no routed experts in the EXL3 index"
layers_b = {int(ROUTED.match(k).group(1)) for k in bi if main_routed(k)}
layers_e = {int(ROUTED.match(k).group(1)) for k in exl3_part}
assert layers_b == layers_e == set(range(N_MAIN)), (sorted(layers_b ^ layers_e))
wm.update(exl3_part)

os.makedirs(out, exist_ok=True)
def link(src_dir, name):
    dst = os.path.join(out, name)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.relpath(os.path.join(src_dir, name), out), dst)

for f in sorted({v for k, v in wm.items() if bi.get(k) == v}):
    link(base, f)
for f in sorted(set(exl3_part.values())):
    link(exl3, f)
for f in ("tokenizer.json", "tokenizer_config.json", "encoding"):
    if os.path.exists(os.path.join(base, f)):
        link(base, f)

json.dump({"metadata": {"hybrid": {"base": base, "exl3": exl3, "routed_layers": N_MAIN}},
           "weight_map": dict(sorted(wm.items()))},
          open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=1)

cfg = json.load(open(os.path.join(base, "config.json")))
eq = json.load(open(os.path.join(exl3, "config.json")))["quantization_config"]
cfg["quantization_config"] = {
    "quant_method": "exl3", "version": eq.get("version"), "avg_bits": eq.get("avg_bits", 2.9),
    "codebook": eq.get("codebook", "mul1"), "out_scales": eq.get("out_scales"),
    "scope": "routed_experts_only", "num_main_layers": N_MAIN,
    "non_routed_quantization": json.load(open(os.path.join(base, "config.json")))["quantization_config"],
}
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
kmap = json.load(open(os.path.join(exl3, "exl3_k_map.json")))
json.dump(kmap, open(os.path.join(out, "exl3_k_map.json"), "w"))

files = sorted(set(wm.values()))
print(f"{len(wm)} tensors ({len(exl3_part)} EXL3 routed-expert tensors) in {len(files)} shards "
      f"({sum(f.endswith('-of-00043.safetensors') for f in files)} from the EXL3 pack) -> {out}")
