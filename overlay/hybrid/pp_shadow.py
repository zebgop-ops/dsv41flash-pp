# SPDX-License-Identifier: Apache-2.0
"""Pipeline-parallel cuts inside a DeepSeek V4.1 kv-sharing group.

Consumers of a kv-source layer (compress_ratio > 0) read the source's
compressed-KV cache and indexer K cache through ``static_forward_context`` on
the same rank, so upstream only allows PP cuts at kv-source boundaries
(layers 2, 8, 14, 20). The decoder group 20-39 (~138 GiB of experts) does not
fit one 64 GiB card, so a receiving rank instantiates a *shadow* of the source
layer's attention module under the source's own prefix: the KV-cache manager
allocates the compressed cache / compressor state / indexer K cache there, and
every step the shadow recomputes the compressor latent from the source layer's
attention input, which the owning rank ships in IntermediateTensors
(bf16 [T, hidden]). Layer 20's candidate blocks ([T, candidate_topk_blocks]
int32) are shipped the same way for the hierarchical indexers at 24/28/32/36.
Cuts must still sit on index-source boundaries (consumers of an index source
reuse its top-k buffer on the same rank).
"""
from __future__ import annotations

import torch
from torch import nn

from vllm.distributed import get_pp_group
from vllm.distributed.utils import get_pp_indices
from vllm.logger import init_logger

logger = init_logger(__name__)

SRC_KEY = "dsv41_src{}"
CAND_KEY = "dsv41_cand"


def _rank_ranges(num_layers: int) -> list[tuple[int, int]]:
    pp = get_pp_group().world_size
    return [get_pp_indices(num_layers, r, pp) for r in range(pp)]


def shadow_needs_for(config, start: int, end: int) -> dict[int, bool]:
    """{source layer -> needs_topk} for the sources of [start, end) that live on
    an earlier rank. kv sources provide the compressed-KV + indexer K caches;
    index sources additionally publish top-k indices (and candidate blocks)."""
    kv_sources = tuple(getattr(config, "kv_source_layer_ids", ()) or ())
    idx_sources = tuple(getattr(config, "index_source_layer_ids", ()) or ())
    ratios = list(getattr(config, "compress_ratios", []) or [])
    needed: dict[int, bool] = {}
    for layer in range(start, end):
        if layer >= len(ratios) or ratios[layer] == 0:
            continue
        kv_src = max(s for s in kv_sources if s <= layer)
        if kv_src < start:
            needed.setdefault(kv_src, False)
        idx_src = max(s for s in idx_sources if s <= layer)
        if idx_src < start:
            needed[idx_src] = True
    return needed


def shadow_ids_for(config, start: int, end: int) -> list[int]:
    return sorted(shadow_needs_for(config, start, end))


def index_sources_ok(config, start: int) -> bool:
    """A cut must not split an index group (top-k buffers are rank-local)."""
    ratios = list(getattr(config, "compress_ratios", []) or [])
    idx = tuple(getattr(config, "index_source_layer_ids", ()) or ())
    if start == 0 or start >= len(ratios) or ratios[start] == 0:
        return True
    return start in idx


class PPShadowPlan:
    """Per-rank plan: which sources to shadow, which inputs to ship."""

    def __init__(self, config, num_layers: int, start: int, end: int) -> None:
        self.start, self.end = start, end
        pp = get_pp_group()
        self.rank = pp.rank_in_group
        ranges = _rank_ranges(num_layers)
        self.shadow_needs = shadow_needs_for(config, start, end) if self.rank > 0 else {}
        self.shadow_ids = sorted(self.shadow_needs)
        # sources some later rank needs -> this rank must emit them (own or forward)
        later_needs: set[int] = set()
        for r in range(self.rank + 1, len(ranges)):
            later_needs.update(shadow_ids_for(config, *ranges[r]))
        # only sources already produced by this or an earlier rank can be shipped
        self.ship_ids = sorted(s for s in later_needs if s < end)
        self.owned_ship_ids = [s for s in self.ship_ids if start <= s < end]
        # what the previous rank hands us (= what our forward reads before its layers)
        self.recv_ids = sorted(set(self.shadow_ids) | {s for s in self.ship_ids if s < start})
        cand_src = getattr(config, "candidate_source_layer_id", -1)
        self.cand_blocks = getattr(config, "candidate_topk_blocks", 0) or 0
        has_cand = cand_src >= 0 and self.cand_blocks > 0
        # later ranks hold layers after the candidate source -> ship candidates
        self.ship_cand = has_cand and cand_src < end and any(
            ranges[r][1] > cand_src + 1 for r in range(self.rank + 1, len(ranges))
        )
        self.recv_cand = has_cand and start > cand_src
        logger.info(
            "PP shadow plan rank %d layers [%d, %d): shadow sources %s, ship %s "
            "(owned %s), candidates ship=%s recv=%s",
            self.rank, start, end, self.shadow_needs, self.ship_ids,
            self.owned_ship_ids, self.ship_cand, self.recv_cand,
        )

    @property
    def active(self) -> bool:
        return bool(self.shadow_ids or self.ship_ids or self.ship_cand or self.recv_cand)


class ShadowSource(nn.Module):
    """A kv-source layer's attention module, re-instantiated on a later rank
    under the source's prefix, driven by the shipped attention input."""

    def __init__(self, attn: nn.Module, layer_id: int) -> None:
        super().__init__()
        self.attn = attn
        self.layer_id = layer_id

    @torch.no_grad()
    def run(self, x: torch.Tensor, positions: torch.Tensor, need_topk: bool) -> None:
        """Recompute what consumers read from this source: the compressed-KV
        cache and indexer K cache (always), plus the top-k indices / candidate
        blocks when a local layer uses this source's indexer."""
        attn = self.attn
        qr_kv, kv_score, indexer_weights = attn._run_parallel_input_projections(x)
        qr, qr_scale, _kv = attn._split_qkv_and_norm(qr_kv)
        latent = attn.compressor(kv_score, positions) if attn.compressor is not None else None
        index_q = index_q_scale = index_weights_out = None
        if attn.indexer is not None:
            # produces the K cache rows (owns_k) and the quantized index query
            index_q, index_q_scale, index_weights_out = attn.indexer(
                qr, latent, indexer_weights, positions, attn.indexer_rotary_emb, qr_scale
            )
        if attn.compressor is not None:
            attn.compressor.insert_cache(latent, positions, attn.rotary_emb)
        if need_topk and attn.indexer is not None and index_q is not None:
            q_quant = (index_q, index_q_scale) if index_q_scale is not None else index_q
            attn.indexer.indexer_op(x, q_quant, None, index_weights_out)


def remap_shadow_weight_name(name: str, shadow_ids: list[int]) -> str:
    for s in shadow_ids:
        pfx = f"layers.{s}.attn."
        if name.startswith(pfx):
            return f"shadow_sources.{s}.attn." + name[len(pfx):]
    return name
