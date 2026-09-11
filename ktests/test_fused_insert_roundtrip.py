"""CUDA fused Q-RoPE / KV-RoPE+UE8M0-insert op (sm80 build) vs the Triton inverse RoPE and
the V4.1 cache dequant: both must agree on the RoPE convention."""
import torch, pytest
from test_rocm_triton_attn_dsv4 import _make_dsv4_rotary, HEAD_DIM, ROPE_HEAD_DIM, NOPE_HEAD_DIM, _ROTARY_CACHE_LEN
import vllm._custom_ops  # noqa
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _fused_inverse_rope_gptj
from vllm.models.deepseek_v4_1.common.ops.cache_utils import dequantize_and_gather_k_cache
dev = torch.device("cuda")

@torch.inference_mode()
def test_q_and_kv_roundtrip(default_vllm_config):
    torch.manual_seed(0)
    rot = _make_dsv4_rotary(dev)
    T, H, BS, NB = 40, 64, 32, 8
    q = torch.randn(T, H, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    kv = torch.randn(T, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    positions = torch.randint(0, _ROTARY_CACHE_LEN, (T,), dtype=torch.int64, device=dev)
    cache = torch.zeros(NB, BS * 584, dtype=torch.uint8, device=dev)
    slots = torch.arange(T, dtype=torch.int64, device=dev)  # consecutive slots in blocks 0..
    op = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
    q_out = op(q, kv, cache, slots, positions, rot.cos_sin_cache, H, 1e-20, BS, False)
    assert q_out.shape == (T, H, HEAD_DIM), q_out.shape
    # Q side: inverse RoPE (as used before wo_a) must give q back
    q_back = _fused_inverse_rope_gptj(q_out.contiguous(), positions, rot.cos_sin_cache, ROPE_HEAD_DIM)
    nope_exact = torch.equal(q_back[..., :NOPE_HEAD_DIM], q[..., :NOPE_HEAD_DIM])
    rope_err = (q_back[..., NOPE_HEAD_DIM:].float() - q[..., NOPE_HEAD_DIM:].float()).abs().max().item()
    print(f"Q: nope passthrough exact={nope_exact}, rope lanes max abs err after inverse={rope_err:.4f}")
    # Also compare q_out rope lanes with the native forward rotation
    q_ref, _ = rot.forward_native(positions, q.clone(), None)
    fwd_err = (q_out[..., NOPE_HEAD_DIM:].float() - q_ref[..., NOPE_HEAD_DIM:].to(torch.bfloat16).float()).abs().max().item()
    print(f"Q: op rope lanes vs native forward rope max abs err={fwd_err:.4f}")
    # K side: dequantize inserted rows and compare with native RoPE of kv
    out = torch.empty(1, T, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    bt = torch.arange((T + BS - 1) // BS, dtype=torch.int32, device=dev).unsqueeze(0)
    dequantize_and_gather_k_cache(out, cache, seq_lens=torch.tensor([T], dtype=torch.int32, device=dev), gather_lens=None,
                                  block_table=bt, block_size=BS, offset=0, use_fnuz=False)
    kv_ref, _ = rot.forward_native(positions, kv.clone().unsqueeze(1), None); kv_ref = kv_ref.squeeze(1).float()
    got = out[0].float()
    rope_k_err = (got[:, NOPE_HEAD_DIM:] - kv_ref[:, NOPE_HEAD_DIM:]).abs().max().item()
    nope_rel = ((got[:, :NOPE_HEAD_DIM] - kv_ref[:, :NOPE_HEAD_DIM]).abs() / (kv_ref[:, :NOPE_HEAD_DIM].abs() + 1e-2)).mean().item()
    print(f"K: rope lanes (bf16 in cache) max abs err={rope_k_err:.4f}; nope lanes (fp8 ue8m0) mean rel err={nope_rel:.4f}")
    assert nope_exact and rope_err < 0.05 and fwd_err < 0.05 and rope_k_err < 0.05 and nope_rel < 0.08
