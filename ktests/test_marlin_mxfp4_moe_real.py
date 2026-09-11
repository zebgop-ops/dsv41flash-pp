"""Marlin MXFP4 MoE (as vLLM runs the GPU layers) on 8 real V4.1 experts vs a torch reference."""
import json, torch
from safetensors import safe_open
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_moe_mxfp4_layer_for_marlin
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.fused_moe.activation import MoEActivation, ApplyMoEActivationConfig
from vllm.scalar_type import scalar_types
S = "/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277"
idx = json.load(open(S + "/model.safetensors.index.json"))["weight_map"]
dev = torch.device("cuda"); L, E, K = 0, 8, 2
def get(n):
    with safe_open(S + "/" + idx[n], "pt") as f: return f.get_tensor(n).to(dev)
w1 = torch.stack([get(f"layers.{L}.ffn.experts.{e}.w1.weight").view(torch.uint8) for e in range(E)])
w3 = torch.stack([get(f"layers.{L}.ffn.experts.{e}.w3.weight").view(torch.uint8) for e in range(E)])
w2 = torch.stack([get(f"layers.{L}.ffn.experts.{e}.w2.weight").view(torch.uint8) for e in range(E)])
s1 = torch.stack([get(f"layers.{L}.ffn.experts.{e}.w1.scale").view(torch.uint8) for e in range(E)])
s3 = torch.stack([get(f"layers.{L}.ffn.experts.{e}.w3.scale").view(torch.uint8) for e in range(E)])
s2 = torch.stack([get(f"layers.{L}.ffn.experts.{e}.w2.scale").view(torch.uint8) for e in range(E)])
print("w1", tuple(w1.shape), "s1", tuple(s1.shape), "w2", tuple(w2.shape), "s2", tuple(s2.shape))
N, H = w1.shape[1], w1.shape[2] * 2
E2M1 = torch.tensor([0,.5,1,1.5,2,3,4,6,-0,-.5,-1,-1.5,-2,-3,-4,-6], device=dev)
def deq(p, s):  # [E, n, k/2] u8, [E, n, k/32] e8m0 -> [E, n, k] f32
    nib = torch.stack([p & 0xF, p >> 4], -1).reshape(p.shape[0], p.shape[1], -1).long()
    sc = (s.to(torch.int32) << 23).view(torch.float32).repeat_interleave(32, -1)
    return E2M1[nib] * sc
g, u, d = deq(w1, s1), deq(w3, s3), deq(w2, s2)
class Lyr(torch.nn.Module): pass
layer = Lyr(); layer.params_dtype = torch.bfloat16
w13 = torch.cat([w1, w3], 1).contiguous(); s13 = torch.cat([s1, s3], 1).contiguous()
mw13, mw2, ms13, ms2, _, _ = prepare_moe_mxfp4_layer_for_marlin(layer, w13, w2, s13, s2, None, None)
torch.manual_seed(0); M = 16
x = (torch.randn(M, H, device=dev) / 3).to(torch.bfloat16)
ids = torch.stack([torch.randperm(E, device=dev)[:K] for _ in range(M)]).to(torch.int32)
wts = (torch.rand(M, K, device=dev) / K)
for limit in (None, 10.0):
    y = fused_marlin_moe(x, mw13, mw2, None, None, ms13, ms2, wts, ids, quant_type_id=scalar_types.float4_e2m1f.id,
                         global_num_experts=E, activation=MoEActivation.SILU,
                         activation_config=ApplyMoEActivationConfig(clamp_limit=limit) if limit else None)
    ref = torch.zeros(M, H, device=dev)
    for t in range(M):
        for j in range(K):
            e = ids[t, j].item(); a = x[t].float() @ g[e].T; b = x[t].float() @ u[e].T
            if limit: b = b.clamp(-limit, limit); a = a.clamp(max=limit)
            ref[t] += wts[t, j] * ((torch.nn.functional.silu(a) * b) @ d[e].T)
    err = (y.float() - ref).abs(); print(f"limit={limit}: max abs {err.max():.4f} ref absmax {ref.abs().max():.3f} mean rel {(err/(ref.abs()+1e-2)).mean():.4f}")
print("done")
