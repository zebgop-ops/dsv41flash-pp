# SPDX-License-Identifier: Apache-2.0
"""EXL3 (ExLlamaV3 trellis) routed experts and dense linears for DeepSeek-V4.1-Flash on the
0909 vLLM image, sm_80, pipeline parallel (no tensor parallel).

Checkpoint ABI (LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw and the Mia-AiLab V4.1 packs):
  quant_method=exl3, codebook=mul1; per matrix: trellis (int16 [in/16, out/16, 16*K]) +
  suh (fp16 [in]) + svh (fp16 [out]) + mul1 (int32 marker). K is per tensor (routed experts
  2 or 3, shared 4/5, attention 5, indexer wk 8, engram wkv 4/5, lm_head 6) and comes from
  exl3_k_map.json next to the weights; wo_a is stored as one matrix per o-group (wo_a.slice.i).

Routed experts run through exllamav3_ext.exl3_moe (the fused trellis MoE kernel, 1.6.x ABI);
experts with more rows than the fused temps take 128-row tiles of the same kernel. Dense,
attention, shared-expert, compressor, indexer, engram-wkv and lm_head matrices are LinearEXL3
(exl3_gemm / reconstruct+hgemm). Ported from Mia'a AI Lab's overlay/exl3.py
(github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks, Apache-2.0) with the fat-kernel
tiers, TP sharding and GB10 staging removed, and the exllamav3 1.6.3 kernel signature.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

logger = init_logger("vllm.dsv41.exl3")

MCG_MARKER_SIGNED_INT32 = -877912083      # 0xCBAC1FED
MUL1_MARKER_SIGNED_INT32 = -2082680531    # 0x83D6B12D
EXL3_SUFFIXES = ("trellis", "suh", "svh", "mcg", "mul1")
SWIGLU_LIMIT_DEFAULT = 10.0
MOE_ACT_SILU = 0                          # exl3_moe_common.cuh; act_limit = DeepSeek's clamped SwiGLU
_K_MAP: dict[str, Any] | None = None
_FUSED_TEMP_CACHE: dict[tuple, tuple[torch.Tensor, ...]] = {}


def temp_rows_fused() -> int:
    """Row capacity per expert of one fused launch (experts above it take row tiles)."""
    return int(os.environ.get("EXL3_TEMP_ROWS_FUSED", "128"))


def fused_moe_enabled() -> bool:
    return os.environ.get("EXL3_FUSED_MOE", "1") != "0"


def pretune_rows() -> list[int]:
    """Row counts every dense EXL3 linear is run at right after load. exllamav3 autotunes exl3_gemm per
    (rows, k, n, K, ...) with timing rounds (event syncs) the first time it sees a shape, which is illegal
    inside CUDA-graph capture; vLLM's eager warmup does not reach every path the capture does (the PP
    shadow layers' indexer projection, for one). Tuning results persist in EXLLAMAV3_TUNE_CACHE."""
    raw = os.environ.get("EXL3_PRETUNE_ROWS", "1,2,3,4,6,8,12,16")
    return [int(x) for x in raw.split(",") if x.strip()]


def pretune_linear(inner, rows: list[int]) -> None:
    if not rows or torch.cuda.is_current_stream_capturing():
        return
    dev = inner.trellis.device
    for r in rows:
        x = torch.zeros((r, int(inner.in_features)), dtype=torch.float16, device=dev)
        inner.forward(x, {}, out_dtype=torch.float32)


# ----------------------------------------------------------------------------- exllamav3 glue
def _install_exllamav3_namespace() -> None:
    """Import LinearEXL3 without running exllamav3/__init__.py (generator, flash-attn, ...)."""
    if "exllamav3.modules.quant.exl3" in sys.modules:
        return
    import exllamav3_ext  # noqa: F401  (compiled for sm_80 in exl3/wheel)

    spec = importlib.util.find_spec("exllamav3")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("exllamav3 package is not on PYTHONPATH (exl3/site)")
    root = Path(list(spec.submodule_search_locations)[0])
    for name, path in (("exllamav3", root), ("exllamav3.modules", root / "modules"), ("exllamav3.model", root / "model")):
        if name in sys.modules:
            continue
        m = types.ModuleType(name)
        m.__file__ = str(path / "__init__.py")
        m.__package__ = name
        m.__path__ = [str(path)]
        sys.modules[name] = m
    if "exllamav3.model.config" not in sys.modules:
        cfg = types.ModuleType("exllamav3.model.config")
        cfg.__file__ = str(root / "model/config.py")
        cfg.__package__ = "exllamav3.model"
        cfg.Config = type("Config", (), {})

        class _InferParams:
            no_reconstruct = False

        class _NullConfig:
            infer_params = _InferParams()

        cfg.InferParams = _InferParams
        cfg.NullConfig = _NullConfig
        sys.modules[cfg.__name__] = cfg


def load_linear_exl3_cls():
    _install_exllamav3_namespace()
    return importlib.import_module("exllamav3.modules.quant.exl3").LinearEXL3


def ext():
    import exllamav3_ext

    return exllamav3_ext


def make_linear_exl3(trellis, suh, svh, marker, *, codebook="mul1", out_dtype=torch.float16):
    """LinearEXL3 over packed tensors that already live on the device (no BF16 expand)."""
    cls = load_linear_exl3_cls()
    kw: dict[str, Any] = dict(
        config=None, in_features=int(suh.numel()), out_features=int(svh.numel()),
        trellis=trellis.contiguous(), suh=suh.contiguous(), svh=svh.contiguous(),
        out_dtype=out_dtype, transformers_fix=True,
    )
    kw["mcg" if codebook == "mcg" else "mul1"] = marker.contiguous()
    return cls(**kw)


# ----------------------------------------------------------------------------- K map
def load_k_map() -> dict[str, Any]:
    global _K_MAP
    if _K_MAP is not None:
        return _K_MAP
    cands = []
    if os.environ.get("DSV41_EXL3_K_MAP"):
        cands.append(Path(os.environ["DSV41_EXL3_K_MAP"]))
    try:
        from vllm.config import get_current_vllm_config

        cands.append(Path(get_current_vllm_config().model_config.model) / "exl3_k_map.json")
    except Exception:  # noqa: BLE001
        pass
    for p in cands:
        if p.is_file():
            _K_MAP = json.loads(p.read_text())
            logger.info("EXL3 K map from %s", p)
            return _K_MAP
    logger.warning("EXL3: no exl3_k_map.json found (%s); falling back to kind-based K", cands)
    _K_MAP = {}
    return _K_MAP


def k_for_prefix(prefix: str) -> int:
    """K for a vLLM module prefix from the pack's K map (falls back to the kind)."""
    p = prefix.replace("-", ".")
    km = load_k_map()
    m = re.search(r"(?:layers|layer)[._](\d+)", p)
    layer = m.group(1) if m else None
    n_main = int(km.get("num_hidden_layers") or 40)
    if p.startswith("mtp") or ".mtp." in f".{p}." or "main_proj" in p or (layer is not None and int(layer) >= n_main):
        return int(km.get("mtp_bits") or 4)
    if "shared_expert" in p and layer is not None:
        k = (km.get("shared") or {}).get(str(int(layer)))
        if isinstance(k, int):
            return k
    if "engram" in p and layer is not None:
        k = (km.get("engram") or {}).get(str(int(layer)))
        if isinstance(k, int):
            return k
    if ".experts" in p or p.endswith("experts") or "routed" in p:
        if layer is not None:
            k = (km.get("routed") or {}).get(str(int(layer)))
            if isinstance(k, int):
                return k
        return 3
    if "indexer.wk" in p:
        return int(km.get("indexer_wk_bits") or 8)
    if "attn" in p or "self_attn" in p:
        return int(km.get("attn_default") or 5)
    if p == "lm_head" or p.endswith("lm_head"):
        return int(km.get("head_bits") or 6)
    return 3


def _suffix_from_mapped_name(weight_name: str) -> str:
    tail = weight_name.rsplit(".", 1)[-1]
    for s in EXL3_SUFFIXES:
        if tail == s or tail.endswith("_" + s):
            return s
    raise ValueError(f"not an EXL3 packed name: {weight_name}")


def _skip_exl3_linear(prefix: str) -> bool:
    """Linears the pack keeps native (BF16/FP16/FP32): vision tower, aligner, router gate,
    indexer weights_proj, DSpark markov/confidence heads, embeddings."""
    p = prefix.replace("-", ".").lower()
    if any(x in p for x in ("vision", "visual", "aligner", "markov_head", "confidence_head", "weights_proj")):
        return True
    if p == "embed" or p.endswith(".embed") or (".embed." in p and "engram" not in p):
        return True
    if p.endswith(".gate") or ".ffn.gate" in p or ".mlp.gate" in p:
        return True
    return False


# ----------------------------------------------------------------------------- config
@register_quantization_config("exl3")
class Exl3Config(QuantizationConfig):
    """Packed EXL3 (mul1 or mcg) for DeepSeek-V4.1-Flash: routed experts + dense linears."""

    def __init__(self, bits: float | int = 2.9, codebook: str = "mul1", **kwargs: Any) -> None:
        super().__init__()
        self.avg_bits = float(bits)
        self.bits = self.avg_bits
        self.codebook = str(codebook).lower()
        self.raw_config = dict(kwargs)
        if self.codebook not in ("mul1", "mcg"):
            raise ValueError(f"EXL3 overlay implements codebook mul1|mcg, got {self.codebook!r}")

    def get_name(self) -> str:
        return "exl3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        skip = {"bits", "avg_bits", "codebook", "quant_method", "tensor_storage"}
        return cls(
            bits=float(config.get("avg_bits", config.get("bits", 2.9))),
            codebook=str(config.get("codebook", "mul1")),
            **{k: v for k, v in config.items() if k not in skip},
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if str((hf_quant_cfg or {}).get("quant_method", "")).lower() == "exl3":
            return "exl3"
        return None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        if isinstance(layer, RoutedExperts):
            return Exl3MoEMethod(layer.moe_config, self, prefix=prefix)
        if isinstance(layer, ParallelLMHead):
            return Exl3LinearMethod(self, prefix=prefix)
        if isinstance(layer, LinearBase) and not _skip_exl3_linear(prefix):
            return Exl3LinearMethod(self, prefix=prefix)
        return None


# ----------------------------------------------------------------------------- routed experts
def build_exl3_fused_state(layer: torch.nn.Module, inners: list[dict[str, Any]]) -> None:
    """Device pointer tables + the shared fp16 temps the fused kernel writes (once per layer)."""
    device = layer.w13_trellis.device
    hidden, inter = int(layer._exl3_hidden_size), int(layer._exl3_intermediate_local)

    def ptrs(which: str, attr: str) -> torch.Tensor:
        return torch.tensor([int(getattr(p[which], attr).data_ptr()) for p in inners], dtype=torch.int64, device=device)

    layer._exl3_ptrs = {f"{w}_{a}": ptrs(w, a) for w in ("gate", "up", "down") for a in ("trellis", "suh", "svh")}
    idx = int(device.index) if device.index is not None else torch.cuda.current_device()
    conc = max(1, int(ext().exl3_moe_max_concurrency(idx)))
    rows = temp_rows_fused()
    key = (str(device), hidden, inter, conc, rows)
    temps = _FUSED_TEMP_CACHE.get(key)
    if temps is None:
        temps = (
            torch.empty((conc, rows, hidden), dtype=torch.float16, device=device),
            torch.empty((conc, rows, hidden), dtype=torch.float16, device=device),
            torch.empty((conc, rows, inter), dtype=torch.float16, device=device),
            torch.empty((conc, rows, inter), dtype=torch.float16, device=device),
        )
        _FUSED_TEMP_CACHE[key] = temps
    layer._exl3_fused_temps = temps
    layer._exl3_fused_concurrency = conc
    g0, u0, d0 = inners[0]["gate"], inners[0]["up"], inners[0]["down"]
    layer._exl3_k_gate, layer._exl3_k_up, layer._exl3_k_down = g0.K, u0.K, d0.K
    layer._exl3_mcg, layer._exl3_mul1 = bool(g0.mcg), bool(g0.mul1)


def moe_num_active() -> int:
    """num_active handed to exl3_moe. -1 = max concurrency (8 groups x 8 SMs = 64 of the 70 SMs must be
    co-resident: the group barriers spin, and anything else holding SMs on another stream can deadlock the
    kernel). N = 1 launches one 32-SM group that walks every active expert: 38 SMs of slack, never deadlocks."""
    return int(os.environ.get("EXL3_MOE_NUM_ACTIVE", "-1"))


def _exl3_moe_launch(xh, out, expert_count, token_sorted, weight_sorted, temps, ptrs, layer, limit, num_active=None):
    """One exl3_moe launch (exllamav3 1.6.x ABI: float K, num_active, det scratch off, 16-row tile)."""
    if num_active is None:
        num_active = moe_num_active()
    cap = int(temps[0].shape[1])
    ext().exl3_moe(
        xh, out, expert_count, token_sorted, weight_sorted,
        temps[0], temps[1], temps[2], temps[3],
        MOE_ACT_SILU,
        float(layer._exl3_k_gate), float(layer._exl3_k_up), float(layer._exl3_k_down),
        ptrs["gate_trellis"], ptrs["gate_suh"], ptrs["gate_svh"],
        ptrs["up_trellis"], ptrs["up_suh"], ptrs["up_svh"],
        ptrs["down_trellis"], ptrs["down_suh"], ptrs["down_svh"],
        bool(layer._exl3_mcg), bool(layer._exl3_mul1),
        bool(layer._exl3_mcg), bool(layer._exl3_mul1),
        bool(layer._exl3_mcg), bool(layer._exl3_mul1),
        float(limit), int(num_active), None, None, 1, cap, 16,
    )


def _exl3_moe_row_tiles(xh, out, counts, token_sorted, weight_sorted, temps, ptrs, layer, limit, max_rows):
    """Prefill with experts above the temp row cap: launch per cap-row slice of the sorted
    assignment buffer (host syncs; decode never gets here)."""
    n_exp = int(counts.shape[0])
    device = counts.device
    tile = int(temps[0].shape[1])
    prefix = torch.zeros(n_exp, dtype=torch.long, device=device)
    if n_exp > 1:
        prefix[1:] = counts[:-1].cumsum(0)
    for t in range((max_rows + tile - 1) // tile):
        row0 = t * tile
        tile_counts = (counts - row0).clamp(min=0, max=tile)
        n_tile = int(tile_counts.sum().item())
        if n_tile == 0:
            continue
        cum = tile_counts.cumsum(0)
        idx = torch.arange(n_tile, device=device)
        e_of = torch.searchsorted(cum, idx, right=True)
        local = idx - (cum[e_of] - tile_counts[e_of])
        src = prefix[e_of] + row0 + local
        ec = torch.zeros(n_exp + 1, dtype=torch.long, device=device)
        ec[:n_exp] = tile_counts
        _exl3_moe_launch(xh, out, ec, token_sorted.index_select(0, src), weight_sorted.index_select(0, src),
                         temps, ptrs, layer, limit)


def apply_exl3_python_loop(x2d, ids, weights, inners, limit, *, only_experts=None, out=None):
    """Unique-expert LinearEXL3 loop (correctness reference / fallback)."""
    tokens, hidden = x2d.shape
    if out is None:
        out = torch.zeros(tokens, hidden, dtype=torch.float32, device=x2d.device)
    for raw in torch.unique(ids).tolist():
        e = int(raw)
        if e < 0 or e >= len(inners) or (only_experts is not None and e not in only_experts):
            continue
        tok, kpos = (ids == e).nonzero(as_tuple=True)
        h = x2d.index_select(0, tok).contiguous().half()
        pack = inners[e]
        g = pack["gate"].forward(h, {}, out_dtype=torch.float32)
        u = pack["up"].forward(h, {}, out_dtype=torch.float32)
        act = F.silu(g.clamp(max=limit)) * u.clamp(min=-limit, max=limit)
        d = pack["down"].forward(act.contiguous().half(), {}, out_dtype=torch.float32)
        out.index_add_(0, tok, d * weights[tok, kpos].unsqueeze(-1).float())
    return out


def apply_exl3_fused_moe(x2d, ids, weights, layer, inners, limit):
    tokens, hidden = x2d.shape
    n_exp = len(inners)
    ptrs, temps = layer._exl3_ptrs, layer._exl3_fused_temps
    flat = ids.reshape(-1).long()
    invalid = (flat < 0) | (flat >= n_exp)
    local = torch.where(invalid, flat.new_full(flat.shape, n_exp), flat)
    topk = int(ids.shape[-1])
    flat_token = torch.arange(tokens, device=x2d.device, dtype=torch.long).repeat_interleave(topk)
    flat_weight = weights.reshape(-1).to(torch.float16)
    order = local.argsort()
    token_sorted, weight_sorted = flat_token[order], flat_weight[order]
    expert_count = torch.zeros(n_exp + 1, dtype=torch.long, device=local.device)
    expert_count.scatter_add_(0, local, torch.ones_like(local))
    out = torch.zeros(tokens, hidden, dtype=torch.float32, device=x2d.device)
    xh = x2d.contiguous().half()
    cap = int(temps[0].shape[1])
    if tokens <= cap:   # decode / small batches: one graph-safe launch, no host sync
        _exl3_moe_launch(xh, out, expert_count, token_sorted, weight_sorted, temps, ptrs, layer, limit)
        return out
    counts = expert_count[:n_exp]
    max_rows = int(counts.max().item())
    if max_rows <= cap:
        _exl3_moe_launch(xh, out, expert_count, token_sorted, weight_sorted, temps, ptrs, layer, limit)
    else:
        _exl3_moe_row_tiles(xh, out, counts, token_sorted, weight_sorted, temps, ptrs, layer, limit, max_rows)
    return out


def apply_exl3_experts(x, topk_ids, topk_weights, layer, *, limit=SWIGLU_LIMIT_DEFAULT):
    inners = getattr(layer, "_exl3_inners", None)
    if not inners:
        raise RuntimeError("EXL3 experts were not built after weight load")
    tokens, hidden = x.shape[-2], x.shape[-1]
    x2d = x.reshape(tokens, hidden)
    ids = topk_ids.reshape(tokens, -1).to(torch.long)
    weights = topk_weights.reshape(tokens, -1)
    if fused_moe_enabled() and getattr(layer, "_exl3_ptrs", None):
        out = apply_exl3_fused_moe(x2d, ids, weights, layer, inners, limit)
    else:
        out = apply_exl3_python_loop(x2d, ids, weights, inners, limit)
    return out.to(dtype=x.dtype)


class Exl3MoEMethod(FusedMoEMethodBase):
    """Packed trellis experts: w13_* stacked [E, 2, ...] so the stock expert mapping
    (experts.w13_ + suffix) lands, w2_* [E, ...]; LinearEXL3 handles after load."""

    def __init__(self, moe, quant_config: Exl3Config, prefix: str = "") -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.prefix = prefix
        self.k = k_for_prefix(prefix)
        self.codebook = quant_config.codebook
        self._logged = False

    def get_fused_moe_quant_config(self, layer) -> FusedMoEQuantConfig | None:
        return None

    def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **extra):
        del params_dtype
        if hidden_size % 16 or intermediate_size_per_partition % 16:
            raise ValueError(f"EXL3 tiles are 16-wide; hidden={hidden_size} inter={intermediate_size_per_partition}")
        k_words = int(self.k) * 16
        in_t, out_t = hidden_size // 16, intermediate_size_per_partition // 16
        attrs = {k: v for k, v in extra.items() if k != "weight_loader"}
        packed = {
            "w13_trellis": torch.empty(num_experts, 2, in_t, out_t, k_words, dtype=torch.int16),
            "w13_suh": torch.empty(num_experts, 2, hidden_size, dtype=torch.float16),
            "w13_svh": torch.empty(num_experts, 2, intermediate_size_per_partition, dtype=torch.float16),
            "w13_mul1": torch.empty(num_experts, 2, 1, dtype=torch.int32),
            "w2_trellis": torch.empty(num_experts, out_t, in_t, k_words, dtype=torch.int16),
            "w2_suh": torch.empty(num_experts, intermediate_size_per_partition, dtype=torch.float16),
            "w2_svh": torch.empty(num_experts, hidden_size, dtype=torch.float16),
            "w2_mul1": torch.empty(num_experts, 1, dtype=torch.int32),
        }
        for name, t in packed.items():
            p = Parameter(t, requires_grad=False)
            layer.register_parameter(name, p)
            set_weight_attrs(p, attrs)
            p.weight_loader = self._load_exl3
            p._exl3_owner = layer
        layer.w13_mcg = layer.w13_mul1   # stock mappings that look for .mcg hit the marker too
        layer.w2_mcg = layer.w2_mul1
        if hasattr(layer, "w13_weight") or hasattr(layer, "w2_weight"):
            raise RuntimeError("EXL3 create_weights must not allocate dense expert weights")
        layer._exl3_hidden_size = hidden_size
        layer._exl3_intermediate_local = intermediate_size_per_partition
        layer._exl3_k_words = k_words

    def _load_exl3(self, param, loaded_weight, weight_name, shard_id="w1", expert_id=0, return_success=False):
        owner = getattr(param, "_exl3_owner", None)
        if owner is not None and hasattr(owner, "_map_global_expert_id_to_local_expert_id"):
            local = owner._map_global_expert_id_to_local_expert_id(expert_id)
            if local == -1:
                return False if return_success else None
            expert_id = local
        loaded = loaded_weight.detach()
        if shard_id in ("w1", "w3"):
            dest = param.data[expert_id, 0 if shard_id == "w1" else 1]
        elif shard_id == "w2":
            dest = param.data[expert_id]
        else:
            raise ValueError(f"unknown EXL3 shard_id={shard_id}")
        if tuple(dest.shape) != tuple(loaded.shape):
            if dest.numel() != loaded.numel():
                raise RuntimeError(f"EXL3 expert load shape mismatch {weight_name} {shard_id} e{expert_id}: "
                                   f"dest {tuple(dest.shape)} vs loaded {tuple(loaded.shape)} (K map wrong?)")
            loaded = loaded.reshape(dest.shape)
        dest.copy_(loaded)
        return True if return_success else None

    def process_weights_after_loading(self, layer) -> None:
        if not hasattr(layer, "w13_trellis"):
            return
        want = MUL1_MARKER_SIGNED_INT32 if self.codebook == "mul1" else MCG_MARKER_SIGNED_INT32
        if not torch.all(layer.w13_mul1.reshape(-1) == want) or not torch.all(layer.w2_mul1.reshape(-1) == want):
            raise RuntimeError(f"EXL3 {self.codebook} marker is not {want}: packed ABI mismatch in {self.prefix}")
        n_exp = int(layer.w13_trellis.shape[0])
        cb = self.codebook
        inners = []
        for e in range(n_exp):
            inners.append({
                "gate": make_linear_exl3(layer.w13_trellis[e, 0], layer.w13_suh[e, 0], layer.w13_svh[e, 0], layer.w13_mul1[e, 0], codebook=cb),
                "up": make_linear_exl3(layer.w13_trellis[e, 1], layer.w13_suh[e, 1], layer.w13_svh[e, 1], layer.w13_mul1[e, 1], codebook=cb),
                "down": make_linear_exl3(layer.w2_trellis[e], layer.w2_suh[e], layer.w2_svh[e], layer.w2_mul1[e], codebook=cb),
            })
        layer._exl3_inners = inners
        layer._exl3_codebook = cb
        layer._exl3_ptrs = None
        mode = "python_loop"
        if fused_moe_enabled():
            build_exl3_fused_state(layer, inners)
            mode = f"exl3_moe(concurrency={layer._exl3_fused_concurrency}, rows={temp_rows_fused()})"
        if not self._logged:
            logger.info("EXL3 %s experts: K=%s experts=%s hidden=%s inter=%s %s (%s)", cb, self.k, n_exp,
                        layer._exl3_hidden_size, layer._exl3_intermediate_local, mode, self.prefix)
            self._logged = True

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None, **kwargs):
        del shared_experts, shared_experts_input, kwargs
        limit = getattr(layer, "swiglu_limit", None) or getattr(self.moe, "swiglu_limit", None) or SWIGLU_LIMIT_DEFAULT
        return apply_exl3_experts(x, topk_ids, topk_weights, layer, limit=float(limit))


# ----------------------------------------------------------------------------- dense linears
class Exl3LinearMethod(LinearMethodBase):
    """Packed EXL3 for attention / shared-expert / compressor / indexer / engram-wkv / lm_head."""

    def __init__(self, quant_config: Exl3Config, prefix: str = "") -> None:
        super().__init__()
        self.quant_config = quant_config
        self.prefix = prefix
        self.k = k_for_prefix(prefix)
        self.codebook = quant_config.codebook

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype, **extra):
        del params_dtype
        attrs = {k: v for k, v in extra.items() if k != "weight_loader"}
        parts = [int(x) for x in output_partition_sizes]
        out_size = int(sum(parts))
        if input_size_per_partition % 16 or out_size % 16:
            raise ValueError(f"EXL3 tiles are 16-wide; in={input_size_per_partition} out={out_size} prefix={self.prefix}")
        k_words = int(self.k) * 16
        packed = {
            "trellis": torch.empty(input_size_per_partition // 16, out_size // 16, k_words, dtype=torch.int16),
            "suh": torch.empty(input_size_per_partition, dtype=torch.float16),
            "svh": torch.empty(out_size, dtype=torch.float16),
            "mul1": torch.empty((), dtype=torch.int32),
        }
        for name, t in packed.items():
            p = Parameter(t, requires_grad=False)
            layer.register_parameter(name, p)
            set_weight_attrs(p, attrs)
            p.weight_loader = self._load
            p._exl3_suffix = name
            p._exl3_owner = layer
        layer.mcg = layer.mul1
        layer._exl3_input_size = input_size_per_partition
        layer._exl3_output_size = out_size
        layer._exl3_output_partition_sizes = parts
        layer._exl3_fused_stash = {}
        layer._exl3_k = self.k
        layer._exl3_codebook = self.codebook
        layer._exl3_prefix = self.prefix

    def _load(self, param, loaded_weight, loaded_shard_id=None, **kwargs):
        weight_name = kwargs.get("weight_name")
        shard_id = None
        if isinstance(loaded_shard_id, int):
            shard_id = loaded_shard_id
        elif isinstance(loaded_shard_id, str):
            if loaded_shard_id.isdigit():
                shard_id = int(loaded_shard_id)
            else:
                weight_name = loaded_shard_id
        name = getattr(param, "_exl3_suffix", None)
        if not name:
            try:
                name = _suffix_from_mapped_name(str(weight_name or ""))
            except ValueError:
                name = "trellis" if param.dim() == 3 else ("suh" if param.dim() == 1 else "mul1")
        loaded = loaded_weight.detach()
        owner = getattr(param, "_exl3_owner", None)
        if name == "mul1":
            loaded = loaded.reshape(()).to(dtype=torch.int32)
        parts = list(getattr(owner, "_exl3_output_partition_sizes", None) or [])
        fused = owner is not None and (shard_id is not None or len(parts) > 1)
        if fused:
            # Merged linears (fused_wqa_wkv, gate_up_proj, fused_wkv_wgate) and the grouped wo_a
            # (one packed matrix per o-group, wo_a.slice.i) keep one LinearEXL3 per shard; the
            # full-size Parameter from create_weights is never read -> drop its storage.
            if loaded.device != param.device:
                loaded = loaded.to(param.device, dtype=param.dtype)
            owner._exl3_fused_stash.setdefault(int(shard_id or 0), {})[name] = loaded
            if param.dim() > 0 and param.numel() > 0:
                param.data = torch.empty(0, dtype=param.dtype, device=param.device)
            return
        if param.dim() == 0:
            param.data.copy_(loaded.reshape(()))
            return
        if name == "trellis" and tuple(param.shape) != tuple(loaded.shape):
            # K from the map differs from the checkpoint (layer 29 shared is mixed): adopt the packed shape.
            param.set_(loaded.to(param.device, dtype=param.dtype))
            return
        if tuple(param.shape) != tuple(loaded.shape):
            raise RuntimeError(f"EXL3 linear load shape mismatch {weight_name}: dest {tuple(param.shape)} != "
                               f"loaded {tuple(loaded.shape)} (prefix={self.prefix} K={self.k} suffix={name})")
        param.data.copy_(loaded)

    def process_weights_after_loading(self, layer) -> None:
        stash = getattr(layer, "_exl3_fused_stash", None) or {}
        parts = list(getattr(layer, "_exl3_output_partition_sizes", None) or [])
        if stash:
            n = len(parts) if len(parts) > 1 else (max(stash) + 1)
            inners = []
            for i in range(n):
                sh = stash.get(i)
                if not sh or not {"trellis", "suh", "svh"} <= set(sh):
                    raise RuntimeError(f"EXL3 fused linear {self.prefix} missing packed shard {i}: have {sorted((sh or {}).keys())}")
                inners.append(make_linear_exl3(sh["trellis"], sh["suh"], sh["svh"], sh.get("mul1", sh.get("mcg")), codebook=self.codebook))
            layer._exl3_inners = inners
            layer._exl3_inner = None
            layer._exl3_k = int(stash[0]["trellis"].shape[-1]) // 16
            for h in inners:
                pretune_linear(h, pretune_rows())
            return
        layer._exl3_inner = make_linear_exl3(layer.trellis, layer.suh, layer.svh, layer.mul1, codebook=self.codebook)
        layer._exl3_inners = None
        layer._exl3_k = int(layer.trellis.shape[-1]) // 16
        pretune_linear(layer._exl3_inner, pretune_rows())

    def apply(self, layer, x, bias=None):
        inners = getattr(layer, "_exl3_inners", None)
        inner = getattr(layer, "_exl3_inner", None)
        if inner is None and not inners:
            self.process_weights_after_loading(layer)
            inners, inner = getattr(layer, "_exl3_inners", None), layer._exl3_inner
        xh = x.contiguous().half()
        odt = torch.float32   # exllamav3 kernels emit fp16/fp32 only
        if inners and getattr(layer, "is_bmm", False) and xh.dim() == 3 and int(xh.shape[1]) == len(inners):
            # grouped wo_a: [T, G, in] -> group g through inner g -> [T, G, out_g]
            out = torch.stack([h.forward(xh[:, g, :].contiguous(), {}, out_dtype=odt) for g, h in enumerate(inners)], dim=1)
        elif inners:
            out = torch.cat([h.forward(xh, {}, out_dtype=odt) for h in inners], dim=-1)
        else:
            out = inner.forward(xh, {}, out_dtype=odt)
        out = out.to(dtype=x.dtype)
        if bias is not None:
            out = out + bias
        return out
