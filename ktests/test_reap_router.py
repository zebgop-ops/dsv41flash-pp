"""The DSv4 Triton top-k router must match the reference Gate for ANY expert count.

patch_reap_router.py lets vLLM's Triton `dsv4_topk` handle expert counts outside the
(256, 384) table it shipped with, which is what makes the REAP-272E checkpoint routable.
This checks the kernel against a torch transcription of the checkpoint's reference
`Gate.forward` (inference/model.py): scores = sqrt(softplus(logits)), the correction bias
steers selection only, weights come from the unbiased scores, renormalized and scaled.

run: pytest -q ktests/test_reap_router.py   (inside the image, one GPU)
"""
import pytest
import torch

from vllm.model_executor.layers.fused_moe.router.dsv4_topk import can_use_dsv4_topk, dsv4_topk

TOPK = 6


def reference_gate(logits: torch.Tensor, bias: torch.Tensor, scale: float, topk: int = TOPK):
    scores = torch.nn.functional.softplus(logits.float()).sqrt()
    idx = (scores + bias).topk(topk, dim=-1)[1]
    w = scores.gather(1, idx)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return w * scale, idx


@pytest.mark.parametrize("num_experts", [272, 256, 384, 128, 160])
@pytest.mark.parametrize("num_tokens", [1, 7, 2048])
def test_matches_reference(num_experts, num_tokens):
    torch.manual_seed(num_experts * 1000 + num_tokens)
    dev = "cuda"
    logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=dev).contiguous()
    bias = (0.1 * torch.randn(num_experts, dtype=torch.float32, device=dev)).contiguous()
    scale = 1.5

    assert can_use_dsv4_topk(logits, bias, TOPK, True, torch.int32), (
        f"the Triton router refuses {num_experts} experts; patch_reap_router.py not applied?")
    w, ids = dsv4_topk(logits, bias, torch.int32, scale)
    rw, rids = reference_gate(logits, bias, scale)

    # the kernel emits its top-k in descending biased-score order, as the reference does
    assert torch.equal(ids.to(torch.int64), rids), "selected experts differ"
    torch.testing.assert_close(w, rw, rtol=1e-5, atol=1e-5)


def test_selection_is_bias_steered_but_weights_are_not():
    """A large bias on one expert must pull it into the top-k without inflating its weight."""
    dev = "cuda"
    torch.manual_seed(0)
    n = 272
    logits = torch.randn(3, n, dtype=torch.float32, device=dev).contiguous()
    bias = torch.zeros(n, dtype=torch.float32, device=dev)
    victim = 200
    bias[victim] = 50.0
    bias = bias.contiguous()
    w, ids = dsv4_topk(logits, bias, torch.int32, 1.0)
    assert (ids == victim).any(dim=-1).all(), "biased expert was not selected"
    rw, rids = reference_gate(logits, bias, 1.0)
    torch.testing.assert_close(w, rw, rtol=1e-5, atol=1e-5)
