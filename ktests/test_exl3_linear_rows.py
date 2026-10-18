#!/usr/bin/env python3
"""LinearEXL3 (exl3_gemm, autotuned per row count) vs a reconstructed-weight fp32 matmul, for one packed
matrix of the pack at every row count the serving graphs capture. A row count whose relative error is far
above the others means the autotuner picked a kernel configuration that is wrong on this GPU.
usage: test_exl3_linear_rows.py <snapshot dir> <tensor prefix, e.g. layers.20.attn.wq_b> [rows...]"""
import json, sys, torch
sys.path.insert(0, "/opt/dsv41/exl3")
import dsv41_exl3 as m
from safetensors import safe_open

snap, name = sys.argv[1].rstrip("/") + "/", sys.argv[2]
rows = [int(r) for r in sys.argv[3:]] or [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32, 64, 144, 145, 512]
idx = json.load(open(snap + "model.safetensors.index.json"))["weight_map"]
dev = torch.device("cuda")
t = {}
for suf in ("trellis", "suh", "svh", "mul1"):
    with safe_open(snap + idx[f"{name}.{suf}"], "pt") as f:
        t[suf] = f.get_tensor(f"{name}.{suf}").to(dev)
inner = m.make_linear_exl3(t["trellis"], t["suh"], t["svh"], t["mul1"])
K, n_in, n_out = inner.K, inner.in_features, inner.out_features
print(f"{name}: K={K} in={n_in} out={n_out} codebook mul1")
ext = m.ext()
# reference weight: reconstruct() gives the had-domain weight; LinearEXL3.reconstruct_hgemm path applies the
# Hadamards + sign vectors in-kernel, so use its own high-row path (reconstruct + hgemm) as the reference at
# one large row count and compare the row-subsets.
torch.manual_seed(0)
X = (torch.randn(1024, n_in, device=dev) * 0.5).half()
ref = inner.forward(X.contiguous(), {"reconstruct": True}, out_dtype=torch.float32)
print(f"reference via reconstruct+hgemm over 1024 rows; |ref| mean {ref.abs().mean():.4f}")
for r in rows:
    x = X[:r].contiguous()
    y = inner.forward(x, {}, out_dtype=torch.float32)
    err = (y - ref[:r]).norm() / ref[:r].norm()
    worst = (y - ref[:r]).abs().max().item()
    print(f"rows {r:4d}: rel err {err.item():.5f}  max abs {worst:.4f}  {'GEMV/GEMM' if r <= 144 else 'reconstruct'}")
