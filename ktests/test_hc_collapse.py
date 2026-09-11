import torch
from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton
dev = torch.device("cuda"); torch.manual_seed(0)
T, HC, H = 33, 4, 5120
h = (torch.randn(T, HC, H, device=dev) * 0.7).to(torch.bfloat16)
pre = torch.softmax(torch.randn(T, HC, device=dev), -1)
out = hc_collapse_triton(h, pre)
ref = (pre.unsqueeze(-1) * h.float()).sum(1)
err = (out.float() - ref).abs().max().item()
print("hc_collapse: out", tuple(out.shape), out.dtype, "max abs err vs weighted-sum reference", round(err, 4), "ref absmax", round(ref.abs().max().item(), 3))
