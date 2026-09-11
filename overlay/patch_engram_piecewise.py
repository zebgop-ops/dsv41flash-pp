#!/usr/bin/env python3
"""Engram hashing inside PIECEWISE CUDA-graph capture.

The model only computes Engram hashes when the forward context carries attention metadata.
PIECEWISE capture runs without it, so piecewise graphs (small prefills, mixed and
non-uniform spec batches) contained no Engram injection at all: replayed logits drifted
1-2 nats from eager and depended on boot/state. The runner now hands the model the static
buffers the hash needs (query_start_loc, the Engram SWA group's slot mapping and block
table); they hold each step's real values at replay, exactly like FULL-graph metadata.
usage: patch_engram_piecewise.py <nvidia/model.py> <nvidia/vl_model.py> <v1/worker/gpu_model_runner.py>"""
import sys
model, vl, runner = sys.argv[1:4]
MARK = "dsv41: engram static meta"
# ---- model.py
s = open(model).read()
if MARK not in s:
    old = '''            if isinstance(attn_metadata, dict) and self.engram_hash.ensure_cache():
                assert self.engram_swa_prefix is not None
                swa_metadata = typing.cast(
                    "DeepseekSparseSWAMetadata", attn_metadata[self.engram_swa_prefix]
                )'''
    new = '''            # dsv41: engram static meta -- under PIECEWISE capture there is no
            # attention metadata; use the runner's static buffers instead.
            _dsv41_static = None
            if not isinstance(attn_metadata, dict) and engram_static_meta is not None:
                _dsv41_static = engram_static_meta
            if (isinstance(attn_metadata, dict) or _dsv41_static is not None) and self.engram_hash.ensure_cache():
                assert self.engram_swa_prefix is not None
                if _dsv41_static is not None:
                    _qsl, _slot, _bt = _dsv41_static
                    swa_metadata = typing.cast("DeepseekSparseSWAMetadata", None)
                    _num_reqs = _qsl.shape[0] - 1
                else:
                    swa_metadata = typing.cast(
                        "DeepseekSparseSWAMetadata", attn_metadata[self.engram_swa_prefix]
                    )
                    _qsl, _slot, _bt = swa_metadata.query_start_loc, swa_metadata.slot_mapping, swa_metadata.block_table
                    _num_reqs = swa_metadata.num_decodes + swa_metadata.num_prefills
                    if engram_static_meta is not None and _dsv41_os.environ.get("DSV41_DEBUG_ENGRAM") == "1":
                        _sq, _ss, _sb = engram_static_meta
                        _T = input_ids.shape[0]
                        _msg = []
                        if not torch.equal(_sq[: _qsl.shape[0]], _qsl): _msg.append(f"qsl static {_sq[:_qsl.shape[0]+1].tolist()} vs meta {_qsl.tolist()}")
                        if not torch.equal(_ss[:_T], _slot[:_T]): _msg.append(f"slot static {_ss[:_T].tolist()} vs meta {_slot[:_T].tolist()}")
                        if _sb.data_ptr() != _bt.data_ptr() or _sb.stride(0) != _bt.stride(0): _msg.append(f"block_table static ptr/stride {_sb.data_ptr()}/{_sb.stride(0)} shape {tuple(_sb.shape)} vs meta {_bt.data_ptr()}/{_bt.stride(0)} shape {tuple(_bt.shape)} row0 static {_sb[0,:3].tolist()} meta {_bt[0,:3].tolist()}")
                        logger.info("DSV41 ENGRAM-META T=%d hashblock=%d %s", _T, self.engram_hash.block_size, "; ".join(_msg) if _msg else "static == meta")'''
    assert s.count(old) == 1, s.count(old); s = s.replace(old, new, 1)
    old = '''                    num_reqs = swa_metadata.num_decodes + swa_metadata.num_prefills
                    lookback_token_ids = input_ids.new_full(
                        (num_reqs, self.engram_hash.lookback_depth), -1
                    )
                engram_hashes = self.engram_hash(
                    input_ids,
                    positions,
                    swa_metadata.query_start_loc,
                    image_mask,
                    lookback_token_ids,
                    image_sentinel_mask(lookback_token_ids),
                    swa_metadata.slot_mapping,
                    swa_metadata.block_table,
                )'''
    new = '''                    num_reqs = _num_reqs
                    lookback_token_ids = input_ids.new_full(
                        (num_reqs, self.engram_hash.lookback_depth), -1
                    )
                engram_hashes = self.engram_hash(
                    input_ids,
                    positions,
                    _qsl,
                    image_mask,
                    lookback_token_ids,
                    image_sentinel_mask(lookback_token_ids),
                    _slot[: input_ids.shape[0]],
                    _bt,
                )'''
    assert s.count(old) == 1, s.count(old); s = s.replace(old, new, 1)
    # forward signatures: inner model and outer causal LM (both carry the same tail)
    old = '''        lookback_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:'''
    assert s.count(old) == 2, s.count(old)
    s = s.replace(old, '''        lookback_token_ids: torch.Tensor | None = None,
        engram_static_meta: tuple | None = None,
    ) -> torch.Tensor | IntermediateTensors:''')
    old = "            lookback_token_ids=lookback_token_ids,\n"
    assert s.count(old) == 1, s.count(old)
    s = s.replace(old, old + "            engram_static_meta=engram_static_meta,\n", 1)
    if "import os as _dsv41_os" not in s:
        s = s.replace("import typing\n", "import typing\nimport os as _dsv41_os\n", 1) if "import typing\n" in s else "import os as _dsv41_os\n" + s
    open(model, "w").write(s); print("patched", model)
else:
    print("already patched", model)
# ---- vl_model.py
s = open(vl).read()
if "engram_static_meta" not in s:
    old = '''        lookback_token_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids=lookback_token_ids,'''
    assert s.count(old) == 1, s.count(old)
    s = s.replace(old, '''        lookback_token_ids: torch.Tensor | None = None,
        engram_static_meta: tuple | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids=lookback_token_ids,
            engram_static_meta=engram_static_meta,''', 1)
    open(vl, "w").write(s); print("patched", vl)
else:
    print("already patched", vl)
# ---- runner
s = open(runner).read()
if MARK not in s:
    old = '''        if self.lookback_token_ids is not None:
            if num_reqs is None:
                num_reqs = self.input_batch.num_reqs
            model_kwargs["lookback_token_ids"] = self._prepare_lookback_token_ids(
                num_reqs
            )
'''
    new = old + '''            # dsv41: engram static meta (see overlay/patch_engram_piecewise.py)
            meta = self._dsv41_engram_static_meta()
            if meta is not None:
                model_kwargs["engram_static_meta"] = meta
'''
    assert s.count(old) == 1, s.count(old); s = s.replace(old, new, 1)
    old = "    def _init_model_kwargs(self, num_reqs: int | None = None):"
    new = '''    def _dsv41_engram_static_meta(self):
        """Static (query_start_loc, slot_mapping, block_table) buffers of the Engram SWA
        group so the model can hash n-grams inside PIECEWISE graph capture."""
        prefix = self.__dict__.get("_dsv41_engram_prefix", "?")
        if prefix == "?":
            prefix = None
            for m in getattr(self, "model", torch.nn.Module()).modules():
                p = getattr(m, "engram_swa_prefix", None)
                if p is not None:
                    prefix = p
                    break
            self.__dict__["_dsv41_engram_prefix"] = prefix
        cfg = getattr(self, "kv_cache_config", None)
        if prefix is None or cfg is None:
            return None
        gid = self.__dict__.get("_dsv41_engram_gid")
        if gid is None:
            # index into input_batch.block_table, which skips encoder-only groups
            bt_idx = 0
            for g in cfg.kv_cache_groups:
                if get_kv_cache_spec_kind(g.kv_cache_spec) == KVCacheSpecKind.ENCODER_ONLY_ATTENTION:
                    continue
                if prefix in g.layer_names:
                    gid = bt_idx
                    break
                bt_idx += 1
            if gid is None:
                return None
            self.__dict__["_dsv41_engram_gid"] = gid
            _bt0 = self.input_batch.block_table[gid]
            logger.info("DSV41 engram static meta: group %d (block_size %d, table %s) for %s",
                        gid, _bt0.block_size, tuple(_bt0.block_table.gpu.shape), prefix)
        bt = self.input_batch.block_table[gid]
        return (self.query_start_loc.gpu, bt.slot_mapping.gpu, bt.block_table.gpu)

''' + old
    assert s.count(old) == 1; s = s.replace(old, new, 1)
    open(runner, "w").write(s); print("patched", runner)
else:
    print("already patched", runner)
