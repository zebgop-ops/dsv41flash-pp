#!/usr/bin/env python3
"""PP: a projected KV cache group with no local layers keeps the global
UniformTypeKVCacheSpecs, and the tensor builder iterated that dict -> tensors for
layers that live on other ranks (compressor ring buffers of layers 2/8/14 on the
rank holding 24-39). Build tensors only for the group's own layer names.
usage: patch_kv_groups.py <vllm/v1/core/kv_cache_utils.py>"""
import sys
path = sys.argv[1]; src = open(path).read()
if "dsv41: only this worker's layers" in src:
    print("already patched"); sys.exit(0)
old = '''        if isinstance(group_spec, UniformTypeKVCacheSpecs):
            for layer_name, spec in group_spec.kv_cache_specs.items():
                layers_by_spec[spec].append(layer_name)
        elif group.layer_names:
'''
new = '''        if isinstance(group_spec, UniformTypeKVCacheSpecs):
            # dsv41: only this worker's layers (a projected group may be empty
            # here while its spec dict still lists the other ranks' layers).
            local_names = set(group.layer_names)
            for layer_name, spec in group_spec.kv_cache_specs.items():
                if layer_name in local_names:
                    layers_by_spec[spec].append(layer_name)
        elif group.layer_names:
'''
assert src.count(old) == 1, src.count(old)
open(path, "w").write(src.replace(old, new)); print("patched", path)
