#!/usr/bin/env python3
"""Apply a layer's Engram when that layer is the first layer of a non-initial pipeline stage
(upstream vllm-project/vllm#62208). The stage-entry branch of DeepseekV41DecoderLayer.forward
takes the incoming pre_mix and residual but skipped the Engram injection that the mid-stage
branch performs, so a partition starting a stage at layer 1 or 14 silently dropped Engram there.
Our default partitions never do; this makes any partition safe.
usage: <models/deepseek_v4_1/nvidia/model.py>"""
import sys
p = sys.argv[1]; s = open(p).read()
if "dsv41 engram at stage entry" in s:
    print("already patched"); sys.exit(0)
old = """            else:
                residual = x
                post_mix, res_mix, x, attn_pre = mhc_pre_delayed_tilelang(
                    residual,
                    self.hc_attn_fn,
"""
new = """            else:
                residual = x
                if self.engram is not None and engram_hashes is not None:  # dsv41 engram at stage entry (#62208)
                    residual = self.engram(
                        residual,
                        engram_hashes[:, self.engram.layer_hash_index],
                        engram_mask,
                    )
                post_mix, res_mix, x, attn_pre = mhc_pre_delayed_tilelang(
                    residual,
                    self.hc_attn_fn,
"""
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, new)); print("patched", p)
