"""Round-trip the patched V4.1 cache kernels on SM8x against torch references."""
import torch, pytest
from vllm.models.deepseek_v4_1.common.ops.cache_utils import quantize_and_insert_k_cache, dequantize_and_gather_k_cache
dev = torch.device("cuda")

def test_quantize_dequantize_roundtrip():
    torch.manual_seed(0)
    num_blocks, block_size, T = 64, 64, 200
    k = (torch.randn(T, 512, device=dev) * 3).to(torch.bfloat16)
    cache = torch.zeros(num_blocks, block_size, 656, dtype=torch.uint8, device=dev)  # 576 data + 80 (pad/scales) per token? use kernel's expected width
    # discover expected width from the kernel docstring: 448 fp8 + 128 bf16 (=256B) + 8 scale bytes -> 584? try 584 first
    cache = torch.zeros(num_blocks, block_size * 584, dtype=torch.uint8, device=dev)
    slots = torch.randperm(num_blocks * block_size, device=dev)[:T].to(torch.int32)
    quantize_and_insert_k_cache(k, cache, slots, use_fnuz=False)
    out = torch.empty(1, T, 512, dtype=torch.bfloat16, device=dev)
    # gather back the same tokens through a fake block table of one request
    # (use the kernel's own reference semantics: dequant of what we wrote)
    seq_lens = torch.tensor([T], dtype=torch.int32, device=dev)
    # build a block table by writing tokens at consecutive slots instead
    slots2 = torch.arange(T, device=dev, dtype=torch.int32)
    cache.zero_(); quantize_and_insert_k_cache(k, cache, slots2, use_fnuz=False)
    block_table = torch.arange((T + block_size - 1)//block_size, device=dev, dtype=torch.int32).unsqueeze(0)
    dequantize_and_gather_k_cache(out, cache, seq_lens=seq_lens, gather_lens=None, block_table=block_table, block_size=block_size, offset=0, use_fnuz=False)
    got = out[0].float(); ref = k.float()
    # first 448 dims fp8-block-quantized (UE8M0 per 64), last 64 bf16 exact
    rel = ((got - ref).abs() / (ref.abs() + 1e-2))
    assert torch.equal(got[:, 448:], ref[:, 448:]), "rope part must be exact"
    assert rel[:, :448].mean() < 0.06, rel[:, :448].mean()
