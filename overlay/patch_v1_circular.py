#!/usr/bin/env python3
"""V1 runner: CircularBufferSpec groups (V4.1 compressor ring buffers: one block per
request, width padded to 16) must not run the generic token->slot mapping kernel. That
kernel indexes block_table[req, pos // block_size] and reads far past the 16-column row
for pos >= 128 (garbage on Hopper, MMU fault / Xid 31 once the read leaves mapped memory
-- rank 2 at ~2k prompt tokens on the CMP 170HX). The V2 runner already disables it
(slot_mapping_enabled = not CircularBufferSpec) and the compressor builder derives ring
slots from the block table itself. usage: <vllm/v1/worker/gpu_model_runner.py>"""
import sys
path = sys.argv[1]; src = open(path).read()
if "dsv41: circular ring groups" in src:
    print("already patched"); sys.exit(0)
old = '''            if kv_cache_spec_kind == KVCacheSpecKind.MAMBA:
                slot_mapping_modes.append(SlotMappingMode.NONE)
            else:
                slot_mapping_modes.append(SlotMappingMode.TOKEN_TO_KV_SLOT)
'''
new = '''            # dsv41: circular ring groups (V4.1 compressor) keep one block per
            # request and compute their own ring slots; the generic kernel
            # would read past the padded 16-column block-table row.
            _dsv41_layer_spec = (
                kv_cache_spec.first_spec
                if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs)
                else kv_cache_spec
            )
            if kv_cache_spec_kind == KVCacheSpecKind.MAMBA or isinstance(
                _dsv41_layer_spec, CircularBufferSpec
            ):
                slot_mapping_modes.append(SlotMappingMode.NONE)
            else:
                slot_mapping_modes.append(SlotMappingMode.TOKEN_TO_KV_SLOT)
'''
assert src.count(old) == 1, src.count(old)
src = src.replace(old, new, 1)
imp = "from vllm.v1.kv_cache_interface import (\n"
assert imp in src
if "    CircularBufferSpec,\n" not in src.split(imp, 1)[1].split(")", 1)[0]:
    src = src.replace(imp, imp + "    CircularBufferSpec,\n", 1)
open(path, "w").write(src); print("patched", path)
