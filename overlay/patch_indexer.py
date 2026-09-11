#!/usr/bin/env python3
"""SM8x: Triton MQA-logits fallback for the sparse attention indexer (DeepGEMM is
Hopper+), torch prefill top-k fallback (0005a), row-chunked prefill scoring (0006),
and the radix persistent_topk gate. Anchor-based, idempotent.
usage: patch_indexer.py <vllm/model_executor/layers/sparse_attn_indexer.py>"""
import sys
path = sys.argv[1]; src = open(path).read()
if "fp8_mqa_logits_triton" in src:
    print("already patched"); sys.exit(0)

def rep(old, new, count=1):
    global src
    assert old in src, f"anchor missing: {old[:70]!r}"
    src = src.replace(old, new, count)

# --- imports ----------------------------------------------------------------
rep("    has_deep_gemm,\n", "    has_deep_gemm,\n    is_deep_gemm_supported,\n")
rep("from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton\n",
    "from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton\n"
    "from vllm.v1.attention.ops.mqa_logits_triton import (\n"
    "    fp8_mqa_logits_triton,\n"
    "    fp8_paged_mqa_logits_triton,\n"
    ")\n")

# --- helpers (torch top-k fallback, row chunk knob) ----------------------------
HELPERS = '''
import os as _os

# Row-chunked prefill scoring on the Triton fallback: caps the [rows, N] fp32 logits
# transient (the indexer's largest allocation at long context). 0 = off.
_DSV4_LOGITS_ROW_CHUNK = int(_os.environ.get("DSV4_LOGITS_ROW_CHUNK", "64"))


def _top_k_per_row_prefill_torch(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
) -> None:
    """torch.topk fallback with ``top_k_per_row_prefill``'s output contract
    (vLLM PR #49897 for SM12x; the CUDA kernel's histogram path can emit
    uninitialized indices on SM8x -> Xid 31 above ~128k tokens)."""
    num_cols = logits.shape[1]
    ks = cu_seqlen_ks.to(torch.long)[:, None]
    cols = torch.arange(num_cols, device=logits.device)[None, :]
    valid = (cols >= ks) & (cols < cu_seqlen_ke.to(torch.long)[:, None])
    logits.masked_fill_(~valid, float("-inf"))
    k = min(topk_tokens, num_cols)
    top_values, top_cols = logits.topk(k, dim=-1)
    relative = (top_cols - ks).to(torch.int32)
    pad_sentinel = torch.iinfo(torch.int32).max
    relative = torch.where(
        top_values.isfinite(), relative, relative.new_full((), pad_sentinel)
    )
    relative, _ = relative.sort(dim=-1)
    relative = torch.where(
        relative == pad_sentinel, relative.new_full((), -1), relative
    )
    topk_indices[:, :k] = relative
    if k < topk_tokens:
        topk_indices[:, k:] = -1


def _prefill_topk_needs_torch_fallback() -> bool:
    forced = _os.environ.get("VLLM_DSV4_PREFILL_TOPK_TORCH")
    if forced is not None:
        return forced == "1"
    if not current_platform.is_cuda():
        return False
    return current_platform.is_device_capability_family(120) or (
        current_platform.has_device_capability(80)
        and not current_platform.has_device_capability(90)
    )


def _prefill_topk(logits, ks, ke, topk_indices, num_rows, topk_tokens) -> None:
    if _prefill_topk_needs_torch_fallback():
        _top_k_per_row_prefill_torch(logits, ks, ke, topk_indices, topk_tokens)
    else:
        ops.top_k_per_row_prefill(
            logits, ks, ke, topk_indices, num_rows,
            logits.stride(0), logits.stride(1), topk_tokens,
        )


def _prefill_candidates(logits, ks, ke, chunk_candidates, candidate_block_size,
                        candidate_write) -> None:
    if chunk_candidates is None:
        return
    if candidate_write:
        _select_candidate_blocks(
            logits, ks, ke, chunk_candidates.shape[1], candidate_block_size,
            chunk_candidates,
        )
    else:
        _apply_candidate_mask(
            logits, ks, ke, chunk_candidates, candidate_block_size,
        )

'''
rep("def _merge_dcp_topk_global(", HELPERS + "\ndef _merge_dcp_topk_global(")

# --- prefill: Triton fallback (row-chunked) ---------------------------------------
rep('''                else:
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        clean_logits=False,
                    )
                num_rows = logits.shape[0]
                if candidate_blocks is not None:
                    # Two-level selection (v4.1): the candidate source
                    # publishes its top blocks; later indexers mask their
                    # scores to them. Both before the row top-k.
                    chunk_candidates = candidate_blocks[
                        chunk.token_start : chunk.token_end
                    ]
                    if candidate_write:
                        _select_candidate_blocks(
                            logits,
                            cu_seqlen_ks,
                            cu_seqlen_ke,
                            chunk_candidates.shape[1],
                            candidate_block_size,
                            chunk_candidates,
                        )
                    else:
                        _apply_candidate_mask(
                            logits,
                            cu_seqlen_ks,
                            cu_seqlen_ke,
                            chunk_candidates,
                            candidate_block_size,
                        )
                ops.top_k_per_row_prefill(
                    logits,
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
''', '''                elif is_deep_gemm_supported():
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        clean_logits=False,
                    )
                else:
                    # SM8x Triton fallback (DeepGEMM unavailable), row-chunked:
                    # each row's candidates/top-k read only its own [ks, ke),
                    # so blocking over rows is exact and caps the fp32 logits
                    # transient at rows x N x 4 bytes.
                    assert q_scale_slice is None, "FP4 indexer Q unsupported on SM8x"
                    _tot = q_slice_cast.shape[0]
                    _step = _DSV4_LOGITS_ROW_CHUNK if _DSV4_LOGITS_ROW_CHUNK > 0 else _tot
                    _cands = (
                        candidate_blocks[chunk.token_start : chunk.token_end]
                        if candidate_blocks is not None
                        else None
                    )
                    for _r0 in range(0, _tot, _step):
                        _r1 = min(_r0 + _step, _tot)
                        _logits = fp8_mqa_logits_triton(
                            q_slice_cast[_r0:_r1],
                            (k_quant_cast, k_scale_cast),
                            weights[chunk.token_start + _r0 : chunk.token_start + _r1],
                            cu_seqlen_ks[_r0:_r1],
                            cu_seqlen_ke[_r0:_r1],
                            clean_logits=False,
                        )
                        _prefill_candidates(
                            _logits, cu_seqlen_ks[_r0:_r1], cu_seqlen_ke[_r0:_r1],
                            None if _cands is None else _cands[_r0:_r1],
                            candidate_block_size, candidate_write,
                        )
                        _prefill_topk(
                            _logits, cu_seqlen_ks[_r0:_r1], cu_seqlen_ke[_r0:_r1],
                            topk_indices[_r0:_r1], _r1 - _r0, topk_tokens,
                        )
                        del _logits
                    logits = None
                if logits is not None:
                    num_rows = logits.shape[0]
                    _prefill_candidates(
                        logits, cu_seqlen_ks, cu_seqlen_ke,
                        candidate_blocks[chunk.token_start : chunk.token_end]
                        if candidate_blocks is not None else None,
                        candidate_block_size, candidate_write,
                    )
                    _prefill_topk(
                        logits, cu_seqlen_ks, cu_seqlen_ke, topk_indices,
                        num_rows, topk_tokens,
                    )
''')

# --- decode: Triton paged fallback --------------------------------------------------
rep('''        else:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
                indices=decode_metadata.indices,
            )
''', '''        elif is_deep_gemm_supported():
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
                indices=decode_metadata.indices,
            )
        else:
            # SM8x Triton fallback. The kernel scores [B, next_n] rows against
            # the last context_lens[b] positions of each request; size the
            # buffer to the active batch max rather than the model max.
            assert padded_q_scale is None, "FP4 indexer Q unsupported on SM8x"
            _ctx = seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
            logits = fp8_paged_mqa_logits_triton(
                padded_q_quant_cast,
                kv_cache,
                weights[:num_padded_tokens],
                _ctx,
                decode_metadata.block_table,
                max_model_len=attn_metadata_narrowed.max_seq_len,
                clean_logits=False,
            )
''')

# --- persistent_topk: sm8x radix kernel returns wrong indices (DSv4 lesson) --------
rep('''        use_persistent_topk = current_platform.is_cuda() and topk_tokens in (
            512,
            1024,
            2048,
        )
''', '''        use_persistent_topk = (
            current_platform.is_cuda()
            and current_platform.has_device_capability(90)
            and topk_tokens in (512, 1024, 2048)
        )
''')

# --- constructor: no hard DeepGEMM requirement on SM8x; prime autotune caches --------
rep('''        if current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM support in "
                "the current vLLM environment."
            )
''', '''        if current_platform.is_cuda() and not is_deep_gemm_supported():
            logger.warning_once(
                "DeepGEMM not supported on this platform; using the Triton "
                "fallback for the sparse attention indexer."
            )
            if use_fp4_cache:
                raise RuntimeError(
                    "The Triton indexer fallback needs the FP8 indexer K cache "
                    "(set --attention-config.indexer_kv_dtype=fp8)."
                )
            # Prime the autotune caches now: memory profiling captures graphs
            # before any warmup hook runs and autotuning is illegal under capture.
            from vllm.v1.attention.ops.mqa_logits_triton import (
                warmup_fp8_mqa_logits_triton,
                warmup_fp8_paged_mqa_logits_triton,
            )

            _nh = num_heads if num_heads is not None else 32
            device = topk_indices_buffer.device
            warmup_fp8_mqa_logits_triton(_nh, head_dim, device)
            for _bs in sorted({64, 128, 256, get_current_vllm_config().cache_config.block_size}):
                warmup_fp8_paged_mqa_logits_triton(_nh, head_dim, _bs, device)
''')
rep('''        use_fp4_cache: bool = False,
        compress_ratio''', '''        use_fp4_cache: bool = False,
        num_heads: int | None = None,
        compress_ratio''')

open(path, "w").write(src); print("patched", path)
