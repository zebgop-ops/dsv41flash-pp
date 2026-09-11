#!/usr/bin/env python3
"""PP cuts inside kv-sharing groups for DeepSeek V4.1 (see hybrid/pp_shadow.py).
Anchor-based, idempotent. usage: patch_pp_shadow.py <vllm/models/deepseek_v4_1/nvidia/model.py>"""
import sys
path = sys.argv[1]; src = open(path).read()
if "PPShadowPlan" in src:
    print("already patched"); sys.exit(0)

def rep(old, new):
    global src
    assert old in src, f"anchor missing: {old[:80]!r}"
    src = src.replace(old, new, 1)

# imports
rep("from vllm.v1.attention.backends.registry import AttentionBackendEnum\n",
    "from vllm.v1.attention.backends.registry import AttentionBackendEnum\n"
    "from hybrid.pp_shadow import (\n"
    "    CAND_KEY as _DSV41_CAND_KEY,\n"
    "    SRC_KEY as _DSV41_SRC_KEY,\n"
    "    PPShadowPlan as _DSV41PPShadowPlan,\n"
    "    ShadowSource as _DSV41ShadowSource,\n"
    "    remap_shadow_weight_name as _dsv41_remap_shadow_weight_name,\n"
    ")\n")

# decoder layer: capture the attention input when a later rank needs it
rep('''        x = self.attn(positions, x, None)
        if self.use_sequence_parallel:
            x = sp_reduce_scatter(x)
''', '''        if getattr(self, "_dsv41_ship_attn_input", False):
            self._dsv41_shipped_x = x
        x = self.attn(positions, x, None)
        if self.use_sequence_parallel:
            x = sp_reduce_scatter(x)
''')

# model __init__: plan + shadow sources BEFORE make_layers (consumer layers look the
# source up in static_forward_context while they are constructed)
rep('''        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
''', '''        # PP cuts inside kv-sharing groups (overlay): shadow kv sources.
        from vllm.distributed.utils import get_pp_indices as _dsv41_get_pp_indices

        _pp_start, _pp_end = _dsv41_get_pp_indices(
            config.num_hidden_layers,
            get_pp_group().rank_in_group,
            get_pp_group().world_size,
        )
        self._pp_plan = _DSV41PPShadowPlan(
            config, config.num_hidden_layers, _pp_start, _pp_end
        )
        self.shadow_sources = nn.ModuleDict()
        for s in self._pp_plan.shadow_ids:
            attn = _select_dsv4_attn_cls(vllm_config)(
                vllm_config,
                prefix=f"{prefix}.layers.{s}.attn",
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=aux_stream_list,
                candidate_block_buffer=self.candidate_block_buffer,
            )
            self.shadow_sources[str(s)] = _DSV41ShadowSource(attn, s)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
''')
rep('''        # The n-gram hash needs a slot-keyed rolling store of compressed ids
        # (chunked prefill / decode lookback); key it off the first local
        # layer's sliding-window KV cache. Only PP ranks owning an engram
        # layer need it.
''', '''        assert (self.start_layer, self.end_layer) == (_pp_start, _pp_end)
        for s in self._pp_plan.owned_ship_ids:
            self.layers[s]._dsv41_ship_attn_input = True

        # The n-gram hash needs a slot-keyed rolling store of compressed ids
        # (chunked prefill / decode lookback); key it off the first local
        # layer's sliding-window KV cache. Only PP ranks owning an engram
        # layer need it.
''')

# make_empty_intermediate_tensors: declare the shipped tensors
rep('''        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.hc_mult, self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                ),
                "pre_mix": torch.zeros(
                    (batch_size, self.hc_mult),
                    dtype=torch.float32,
                    device=device,
                ),
            }
        )
''', '''        tensors = {
            "hidden_states": torch.zeros(
                (batch_size, self.hc_mult, self.config.hidden_size),
                dtype=dtype,
                device=device,
            ),
            "pre_mix": torch.zeros(
                (batch_size, self.hc_mult),
                dtype=torch.float32,
                device=device,
            ),
        }
        plan = self._pp_plan
        # Declare exactly what the previous rank sends (dummy runs read these).
        for s in plan.recv_ids:
            tensors[_DSV41_SRC_KEY.format(s)] = torch.zeros(
                (batch_size, self.config.hidden_size), dtype=dtype, device=device
            )
        if plan.ship_cand or plan.recv_cand:
            tensors[_DSV41_CAND_KEY] = torch.zeros(
                (batch_size, plan.cand_blocks), dtype=torch.int32, device=device
            )
        return IntermediateTensors(tensors)
''')

# forward: run shadows / receive candidates before the local layers
rep('''        residual, post_mix, res_mix = None, None, None
        pre_mix: torch.Tensor | None = None
        if not get_pp_group().is_first_rank:
            assert intermediate_tensors is not None
            pre_mix = intermediate_tensors["pre_mix"]
''', '''        residual, post_mix, res_mix = None, None, None
        pre_mix: torch.Tensor | None = None
        if not get_pp_group().is_first_rank:
            assert intermediate_tensors is not None
            pre_mix = intermediate_tensors["pre_mix"]
            plan = self._pp_plan
            num_tokens = positions.shape[0]
            if plan.recv_cand:
                self.candidate_block_buffer[:num_tokens].copy_(
                    intermediate_tensors[_DSV41_CAND_KEY][:num_tokens]
                )
            for s in plan.shadow_ids:
                self.shadow_sources[str(s)].run(
                    intermediate_tensors[_DSV41_SRC_KEY.format(s)][:num_tokens],
                    positions,
                    plan.shadow_needs[s],
                )
''')

# forward: ship on non-last ranks
rep('''        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "pre_mix": pre_mix}
            )
''', '''        if not get_pp_group().is_last_rank:
            out = {"hidden_states": hidden_states, "pre_mix": pre_mix}
            plan = self._pp_plan
            num_tokens = positions.shape[0]
            for s in plan.ship_ids:
                key = _DSV41_SRC_KEY.format(s)
                if s in plan.owned_ship_ids:
                    out[key] = self.layers[s]._dsv41_shipped_x
                else:
                    assert intermediate_tensors is not None
                    out[key] = intermediate_tensors[key][:num_tokens]
            if plan.ship_cand:
                assert self.candidate_block_buffer is not None
                out[_DSV41_CAND_KEY] = self.candidate_block_buffer[:num_tokens]
            return IntermediateTensors(out)
''')

# loader: shadow weights live under shadow_sources.<s>.
rep('''        for name, loaded_weight in weights:
''', '''        for name, loaded_weight in weights:
            name = _dsv41_remap_shadow_weight_name(name, self._pp_plan.shadow_ids)
''')
open(path, "w").write(src); print("patched", path)
