#!/usr/bin/env python3
"""V4.1 attention: pass the indexer head count to SparseAttnIndexer (the SM8x Triton
fallback primes its autotune caches at construction). Anchor-based, idempotent.
usage: patch_attention_v41.py <vllm/models/deepseek_v4_1/attention.py>"""
import sys
path = sys.argv[1]; src = open(path).read()
if "num_heads=self.n_head," in src:
    print("already patched"); sys.exit(0)
old = '''            skip_k_cache_insert=True,
            use_fp4_cache=self.use_fp4_kv,
'''
new = '''            skip_k_cache_insert=True,
            use_fp4_cache=self.use_fp4_kv,
            num_heads=self.n_head,
'''
assert old in src, "anchor missing"
open(path, "w").write(src.replace(old, new, 1)); print("patched", path)
