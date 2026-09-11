"""Does the fused V4 KV insert kernel skip PAD (-1) slots? Guard tensors around the cache
detect out-of-bounds writes; the cache itself must stay untouched for a -1 slot."""
import torch, pytest
from test_rocm_triton_attn_dsv4 import _make_dsv4_rotary, HEAD_DIM, _ROTARY_CACHE_LEN
import vllm._custom_ops  # noqa
dev = torch.device("cuda")

@torch.inference_mode()
def test_pad_slot_is_skipped(default_vllm_config):
    torch.manual_seed(0)
    rot = _make_dsv4_rotary(dev)
    T, H, BS, NB = 4, 64, 32, 2
    q = torch.randn(T, H, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    kv = torch.randn(T, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    positions = torch.arange(T, dtype=torch.int64, device=dev)
    # guard | cache | guard in one allocation
    row = BS * 584
    big = torch.full((3 * NB * row,), 0x5A, dtype=torch.uint8, device=dev)
    cache = big[NB * row: 2 * NB * row].view(NB, row)
    cache.fill_(0x11)
    before = big.clone()
    slots = torch.tensor([-1, -1, -1, -1], dtype=torch.int64, device=dev)
    op = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
    op(q, kv, cache, slots, positions, rot.cos_sin_cache, H, 1e-20, BS, False)
    torch.cuda.synchronize()
    diff = (big != before).nonzero().flatten()
    print("bytes changed with all-PAD slots:", diff.numel(), "first idx:", diff[:4].tolist(), "(cache spans", NB*row, "to", 2*NB*row, ")")
    assert diff.numel() == 0, "kernel wrote despite PAD slot"
    # mixed: one real slot, one PAD
    slots = torch.tensor([0, -1, 5, -1], dtype=torch.int64, device=dev)
    before = big.clone()
    op(q, kv, cache, slots, positions, rot.cos_sin_cache, H, 1e-20, BS, False)
    torch.cuda.synchronize()
    diff = (big != before).nonzero().flatten()
    lo, hi = diff.min().item(), diff.max().item()
    print("mixed: changed bytes", diff.numel(), "range", lo, hi, "inside cache:", lo >= NB*row and hi < 2*NB*row)
    assert lo >= NB * row and hi < 2 * NB * row
