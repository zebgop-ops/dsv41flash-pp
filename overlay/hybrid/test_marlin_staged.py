"""Staged Marlin MXFP4 MoE prepare must equal vLLM's pure prepare bit for bit."""
import sys, torch
sys.path.insert(0, "/opt/dsv41")
import vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 as m4
from hybrid.marlin_staged import prepare_marlin_mxfp4_moe_staged
pure = m4.prepare_moe_mxfp4_layer_for_marlin
dev = torch.device("cuda", 0); torch.manual_seed(0)
E, N, K = 4, 256, 512
class L(torch.nn.Module):
    pass
lay = L(); lay.params_dtype = torch.bfloat16
w13 = torch.randint(0, 256, (E, 2 * N, K // 2), dtype=torch.uint8, device=dev)
w2 = torch.randint(0, 256, (E, K, N // 2), dtype=torch.uint8, device=dev)
s13 = torch.randint(100, 140, (E, 2 * N, K // 32), dtype=torch.uint8, device=dev)
s2 = torch.randint(100, 140, (E, K, N // 32), dtype=torch.uint8, device=dev)
for n, t in (("w13_weight", w13), ("w2_weight", w2), ("w13_weight_scale", s13), ("w2_weight_scale", s2)):
    lay.register_parameter(n, torch.nn.Parameter(t.clone(), requires_grad=False))
ref = pure(lay, w13.clone(), w2.clone(), s13.clone(), s2.clone(), None, None)
prepare_marlin_mxfp4_moe_staged(lay)
got = (lay.w13_weight, lay.w2_weight, lay.w13_weight_scale, lay.w2_weight_scale)
for name, a, b in zip(("w13", "w2", "s13", "s2"), got, ref[:4]):
    print(name, tuple(a.shape), a.dtype, "equal:", torch.equal(a, b))
    assert torch.equal(a, b)
print("PASS")
