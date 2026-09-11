"""Dense MXFP8 [32,32]-block weight through the Marlin path vs torch dequant, on a real V4.1 tensor."""
import sys, torch
from safetensors import safe_open
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_mxfp8_layer_for_marlin, apply_mxfp8_marlin_linear
import json; S = "/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277"
name = sys.argv[1] if len(sys.argv) > 1 else "layers.0.attn.wq_b"
dev = torch.device("cuda")
idx = json.load(open(S + "/model.safetensors.index.json"))["weight_map"]
with safe_open(S + "/" + idx[name + ".weight"], "pt") as f:
    w = f.get_tensor(name + ".weight").to(dev); s = f.get_tensor(name + ".scale").to(dev)
N, K = w.shape; print(name, "weight", tuple(w.shape), w.dtype, "scale", tuple(s.shape), s.dtype)
s_u8 = s.view(torch.uint8)
s_rows = s_u8.repeat_interleave(32, dim=0)              # [N, K/32]  (loader behaviour)
scale_f = (s_rows.to(torch.int32) << 23).view(torch.float32)  # e8m0 -> float
w_deq = w.to(torch.float32) * scale_f.repeat_interleave(32, dim=1)
class L(torch.nn.Module): pass
layer = L(); layer.weight = torch.nn.Parameter(w.clone(), requires_grad=False); layer.weight_scale = torch.nn.Parameter(s_rows.clone(), requires_grad=False)
layer.output_size_per_partition = N; layer.input_size_per_partition = K
prepare_mxfp8_layer_for_marlin(layer)
torch.manual_seed(0); x = (torch.randn(64, K, device=dev) / 4).to(torch.bfloat16)
y = apply_mxfp8_marlin_linear(input=x, weight=layer.weight, weight_scale=layer.weight_scale, workspace=layer.workspace, size_n=N, size_k=K)
ref = x.float() @ w_deq.T
err = (y.float() - ref).abs(); print(f"marlin vs torch: max abs {err.max():.4f}, mean rel {(err / (ref.abs() + 1e-3)).mean():.4f}, ref absmax {ref.abs().max():.3f}")
assert (err / (ref.abs() + 1e-2)).mean() < 0.05
print("PASS")
