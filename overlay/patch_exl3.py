#!/usr/bin/env python3
"""Wire the EXL3 overlay (overlay/exl3/dsv41_exl3.py) into the 0909 image for EXL3 packs of
DeepSeek-V4.1-Flash (LibertAIDAI / Mia-AiLab layout: packed experts AND dense linears):
  * quantization registry: "exl3" resolves to Exl3Config (imported from /opt/dsv41/exl3 on PYTHONPATH)
    and is tried first among the config overrides;
  * model: lm_head gets the quant config and the packed head.* names; wo_a.slice.i.* (one packed
    matrix per o-group) is routed to the wo_a parameter with the group as shard id;
  * compressor fused_wkv_wgate and indexer wk take the quant config (the pack quantizes them) and
    the compressor score closure uses the quant method when there is no dense weight;
  * o_proj: the Ampere/ROCm inverse-RoPE path applies the grouped EXL3 wo_a instead of the bf16 einsum.
Every anchor is a no-op for the MXFP4/FP8 checkpoints (dense weights keep their old paths).
usage: patch_exl3.py <overlay vllm root>"""
import sys

V = sys.argv[1]

def patch(rel, pairs, marker):
    p = f"{V}/{rel}"; s = open(p).read()
    if marker in s:
        print("already patched", rel); return
    for old, new in pairs:
        assert s.count(old) == 1, (rel, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w").write(s); print("patched", rel)

# 1. registry
patch("model_executor/layers/quantization/__init__.py", [
    ('QuantizationMethods = Literal[\n    "awq",\n', 'QuantizationMethods = Literal[\n    "exl3",  # dsv41 exl3 overlay\n    "awq",\n'),
    ('    # Update the `method_to_config` with customized quantization methods.\n    method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)\n',
     '    if quantization == "exl3" and "exl3" not in _CUSTOMIZED_METHOD_TO_QUANT_CONFIG:\n'
     '        import dsv41_exl3  # noqa: F401  dsv41 exl3 overlay: registers Exl3Config\n'
     '    # Update the `method_to_config` with customized quantization methods.\n    method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)\n'),
], "dsv41 exl3 overlay")

# 2. override order: an exl3 pack's quantization_config says quant_method=exl3; try it first
patch("config/model.py", [
    ('            overrides = [\n                "auto_gptq",\n', '            overrides = [\n                "exl3",  # dsv41 exl3 overlay\n                "auto_gptq",\n'),
    ('            quantization_methods = [\n                q for q in supported_quantization if q not in overrides\n            ]\n',
     '            if quant_method == "exl3":  # dsv41 exl3 overlay: make sure the class is registered\n'
     '                import dsv41_exl3  # noqa: F401\n'
     '            quantization_methods = [\n                q for q in supported_quantization if q not in overrides\n            ]\n'),
], "dsv41 exl3 overlay")

# 3. model: lm_head quant config + packed head names + wo_a slices
patch("models/deepseek_v4_1/nvidia/model.py", [
    ('            self.lm_head = ParallelLMHead(\n                config.vocab_size,\n                config.hidden_size,\n                prefix=maybe_prefix(prefix, "lm_head"),\n            )\n',
     '            self.lm_head = ParallelLMHead(\n                config.vocab_size,\n                config.hidden_size,\n'
     '                quant_config=vllm_config.quant_config,  # dsv41 exl3: packed head\n                prefix=maybe_prefix(prefix, "lm_head"),\n            )\n'),
    ('        orig_to_new_suffix={\n            "head.weight": "lm_head.weight",\n            "embed.weight": "embed_tokens.weight",\n',
     '        orig_to_new_suffix={\n            "head.weight": "lm_head.weight",\n'
     '            "head.trellis": "lm_head.trellis",  # dsv41 exl3\n            "head.suh": "lm_head.suh",\n            "head.svh": "lm_head.svh",\n            "head.mul1": "lm_head.mul1",\n            "head.mcg": "lm_head.mcg",\n'
     '            "embed.weight": "embed_tokens.weight",\n'),
    ('            if name.startswith(("vision.", "aligner.", "image_")):\n                # Vision weights are loaded by the outer multimodal wrapper.\n                logger.warning_once("Skipping non-text weight: %s", name)\n                continue\n            if pad_shared_expert',
     '            if name.startswith(("vision.", "aligner.", "image_")):\n                # Vision weights are loaded by the outer multimodal wrapper.\n                logger.warning_once("Skipping non-text weight: %s", name)\n                continue\n'
     '            # dsv41 exl3: wo_a is packed per o-group as wo_a.slice.{i}.{trellis,suh,svh,mul1}\n'
     '            _sl = re.search(r"\\.slice\\.(\\d+)\\.(trellis|suh|svh|mul1|mcg)$", name)\n'
     '            if _sl:\n'
     '                _mapped = name[: _sl.start()] + "." + _sl.group(2)\n'
     '                if is_pp_missing_parameter(_mapped, self):\n                    continue\n'
     '                if _mapped not in params_dict:\n'
     '                    _h, _, _l = _mapped.rpartition(".")\n'
     '                    if f"{_h}.base_layer.{_l}" in params_dict:\n                        _mapped = f"{_h}.base_layer.{_l}"\n'
     '                _param = params_dict[_mapped]\n'
     '                _param.weight_loader(_param, loaded_weight, int(_sl.group(1)))\n'
     '                loaded_params.add(_mapped)\n                continue\n'
     '            if pad_shared_expert'),
], "dsv41 exl3")
# the VL wrapper's mapper
patch("models/deepseek_v4_1/nvidia/vl_model.py", [
    ('        orig_to_new_suffix={\n            "head.weight": "language_model.lm_head.weight",\n            "embed.weight": "embed_tokens.weight",\n',
     '        orig_to_new_suffix={\n            "head.weight": "language_model.lm_head.weight",\n'
     '            "head.trellis": "language_model.lm_head.trellis",  # dsv41 exl3\n            "head.suh": "language_model.lm_head.suh",\n            "head.svh": "language_model.lm_head.svh",\n            "head.mul1": "language_model.lm_head.mul1",\n            "head.mcg": "language_model.lm_head.mcg",\n'
     '            "embed.weight": "embed_tokens.weight",\n'),
], "dsv41 exl3")

# 4. compressor + indexer linears carry the quant config; compressor score without a dense weight
patch("models/deepseek_v4_1/compressor.py", [
    ('            quant_config=None,\n            disable_tp=True,\n            prefix=f"{prefix}.fused_wkv_wgate",\n',
     '            quant_config=vllm_config.quant_config,  # dsv41 exl3: packed in EXL3 packs\n            disable_tp=True,\n            prefix=f"{prefix}.fused_wkv_wgate",\n'),
], "dsv41 exl3")
patch("models/deepseek_v4_1/attention.py", [
    ('                bias=False,\n                quant_config=None,\n                prefix=f"{prefix}.wk",\n',
     '                bias=False,\n                quant_config=quant_config,  # dsv41 exl3: packed K=8 in EXL3 packs\n                prefix=f"{prefix}.wk",\n'),
    ('            def compressor_kv_score() -> torch.Tensor:\n                return torch.mm(\n                    hidden_states,\n                    compressor.fused_wkv_wgate.weight.T,\n                    out_dtype=torch.float32,\n                )\n',
     '            def compressor_kv_score() -> torch.Tensor:\n'
     '                _lin = compressor.fused_wkv_wgate\n'
     '                if getattr(_lin, "weight", None) is None:  # dsv41 exl3: packed, no dense weight\n'
     '                    return _lin.quant_method.apply(_lin, hidden_states).float()\n'
     '                return torch.mm(\n                    hidden_states,\n                    _lin.weight.T,\n                    out_dtype=torch.float32,\n                )\n'),
], "dsv41 exl3")

# 5. o_proj: grouped EXL3 wo_a on the inverse-RoPE path (Ampere class inherits the ROCm one)
patch("models/deepseek_v4_1/amd/rocm.py", [
    ('    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:\n        # ROCm BF16 reference wo_a path (inverse RoPE + einsum) + wo_b.\n        z = rocm_inv_rope_einsum(\n',
     '    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:\n'
     '        # ROCm BF16 reference wo_a path (inverse RoPE + einsum) + wo_b.\n'
     '        if getattr(self.wo_a, "weight", None) is None and getattr(self.wo_a, "_dsv4_wo_a_bf16", None) is None:  # dsv41 exl3: packed wo_a, one matrix per o-group\n'
     '            from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _fused_inverse_rope_gptj\n'
     '            o_ref = _fused_inverse_rope_gptj(o, positions, self.rotary_emb.cos_sin_cache, self.rope_head_dim)\n'
     '            o_ref = o_ref.view(o.shape[0], self.n_local_groups, -1)\n'
     '            z = self.wo_a.quant_method.apply(self.wo_a, o_ref)\n'
     '            return self.wo_b(z.flatten(1))\n'
     '        z = rocm_inv_rope_einsum(\n'),
], "dsv41 exl3")
patch("models/deepseek_v4_1/ampere/ampere_sparse.py", [
    ('        wo_a = self.wo_a\n        if not getattr(self, "_dsv41_wo_a_checked", False):\n',
     '        wo_a = self.wo_a\n'
     '        if getattr(wo_a, "weight", None) is None:  # dsv41 exl3: packed wo_a has no dense weight\n'
     '            return super()._o_proj(o, positions)\n'
     '        if not getattr(self, "_dsv41_wo_a_checked", False):\n'),
], "dsv41 exl3")
# 6. one stream for every EXL3 kernel: exl3_gemm and exl3_moe share DevCtx's per-device lock buffer (split-K
#    reductions, group barriers); two of them running concurrently on different streams corrupt it and spin
#    forever (seen as ranks hanging at 100% GPU in the CUDA-graph warmup). No attention aux streams under EXL3;
#    the launcher also sets VLLM_DISABLE_SHARED_EXPERTS_STREAM=1.
patch("models/deepseek_v4_1/nvidia/model.py", [
    ('        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]\n',
     '        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]\n'
     '        if _dsv41_os.environ.get("DSV41_EXL3") == "1":  # dsv41 exl3 streams: exllamav3 kernels share one lock buffer\n'
     '            aux_stream_list = None\n'),
], "dsv41 exl3 streams")
print("done")
