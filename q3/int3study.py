#!/usr/bin/env python3
"""3-bit feasibility study on real DeepSeek-V4.1 experts (CPU, no server needed).

For a layer and a few experts: dequantize the checkpoint's MXFP4 experts (the reference the model
was trained to), re-quantize to INT3 (asymmetric, per-group along K) by round-to-nearest and by GPTQ
(Hessian from Gaussian inputs unless an activation file is given), and report
  * relative Frobenius weight error per matrix,
  * relative output error of the whole expert (clamped SwiGLU, w2) on fresh inputs,
  * bits/weight and GiB per layer (384 experts) for each scheme.
usage: int3study.py <snapshot dir> <layer> <n_experts> [group sizes, e.g. 32,64,128] [acts.pt]
"""
import json, math, struct, sys, time, torch, torch.nn.functional as F
from safetensors import safe_open

snap, layer, n_exp = sys.argv[1].rstrip("/") + "/", int(sys.argv[2]), int(sys.argv[3])
groups = [int(g) for g in (sys.argv[4] if len(sys.argv) > 4 else "32,64,128").split(",")]
acts = torch.load(sys.argv[5]) if len(sys.argv) > 5 else None
torch.manual_seed(0); torch.set_num_threads(8)
E, H, I, LIM = 384, 5120, 2304, 10.0
idx = json.load(open(snap + "model.safetensors.index.json"))["weight_map"]
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])

def deq(name):
    with safe_open(snap + idx[name + ".weight"], "pt") as f:
        p = f.get_tensor(name + ".weight"); s = f.get_tensor(name + ".scale")
    p = p.view(torch.uint8); nib = torch.stack([p & 0xF, p >> 4], -1).reshape(p.shape[0], -1).long()
    sc = (s.view(torch.uint8).to(torch.int32) << 23).view(torch.float32).repeat_interleave(32, 1)
    return E2M1[nib] * sc

def rtn3(w, G, levels=7):
    """asymmetric INT3 (levels=7) or INT4 (levels=15), per group of G along K; returns dequantized weights"""
    R, K = w.shape; wg = w.reshape(R, K // G, G)
    mn, mx = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True)
    scale = (mx - mn).clamp(min=1e-8) / levels
    zero = torch.round(-mn / scale)
    q = torch.clamp(torch.round(wg / scale) + zero, 0, levels)
    return ((q - zero) * scale).reshape(R, K)

def gptq3(w, Hm, G, blocksize=128, percdamp=0.01):
    """GPTQ (Frantar et al.) with per-group asymmetric INT3 grids fitted on the error-compensated columns."""
    W = w.clone().float(); R, K = W.shape
    Hh = Hm.clone(); dead = torch.diag(Hh) == 0; Hh[dead, dead] = 1; W[:, dead] = 0
    Hh += torch.eye(K) * percdamp * torch.mean(torch.diag(Hh))
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hh)), upper=True)
    Q = torch.zeros_like(W)
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K); W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % G == 0:  # new group: fit the grid on the (already compensated) group columns
                g = W[:, col:col + G] if i == 0 else torch.cat([W1[:, i:], W[:, i2:col + G]], 1)[:, :G]
                mn, mx = g.amin(1, keepdim=True), g.amax(1, keepdim=True)
                scale = (mx - mn).clamp(min=1e-8) / 7; zero = torch.round(-mn / scale)
            wcol = W1[:, i:i + 1]; d = Hinv1[i, i]
            q = (torch.clamp(torch.round(wcol / scale) + zero, 0, 7) - zero) * scale
            Q1[:, i:i + 1] = q; err = (wcol - q) / d
            W1[:, i:] -= err @ Hinv1[i:i + 1, i:]; Err1[:, i:i + 1] = err
        Q[:, i1:i2] = Q1; W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q

def expert_out(x, w1, w3, w2):
    g = (x @ w1.t()).clamp(max=LIM); u = (x @ w3.t()).clamp(-LIM, LIM)
    return (F.silu(g) * u) @ w2.t()

def rel(a, b): return ((a - b).norm() / b.norm()).item()

def bpw(G): return 3 + 19 / G          # 3-bit codes + fp16 scale + 3-bit zero per group
def gib_per_layer(b): return E * (2 * I * H + H * I) * b / 8 / 2**30

print(f"layer {layer}, {n_exp} experts, groups {groups}; MXFP4 reference = 4.25 bpw = {gib_per_layer(4.25):.2f} GiB/layer")
for G in groups: print(f"  INT3 g{G}: {bpw(G):.3f} bpw = {gib_per_layer(bpw(G)):.2f} GiB/layer; 10 layers = {10*gib_per_layer(bpw(G)):.1f} GiB")
Xcal = acts["x"].float() if acts is not None else torch.randn(16384, H) * 0.05  # calibration inputs (n >> K, else the Hessian is rank-deficient)
Xtst = torch.randn(512, H) * 0.05                                                 # fresh test inputs
Hx = (Xcal.t() @ Xcal) * 2 / Xcal.shape[0]
res = {}
for e in range(n_exp):
    t0 = time.time()
    w1, w3, w2 = (deq(f"layers.{layer}.ffn.experts.{e}.{n}") for n in ("w1", "w3", "w2"))
    ref = expert_out(Xtst, w1, w3, w2)
    # Hessian for w2's input = the hidden activations of this expert on the calibration set
    hcal = F.silu((Xcal @ w1.t()).clamp(max=LIM)) * (Xcal @ w3.t()).clamp(-LIM, LIM)
    Hh = (hcal.t() @ hcal) * 2 / hcal.shape[0]
    q1, q3, q2 = rtn3(w1, 32, 15), rtn3(w3, 32, 15), rtn3(w2, 32, 15)   # INT4 g32 reference (4.59 bpw)
    r = res.setdefault((32, "int4"), {"w1": [], "w3": [], "w2": [], "out": []})
    r["w1"].append(rel(q1, w1)); r["w3"].append(rel(q3, w3)); r["w2"].append(rel(q2, w2)); r["out"].append(rel(expert_out(Xtst, q1, q3, q2), ref))
    for G in groups:
        for method in ("rtn", "gptq"):
            if method == "rtn": q1, q3, q2 = rtn3(w1, G), rtn3(w3, G), rtn3(w2, G)
            else: q1, q3, q2 = gptq3(w1, Hx, G), gptq3(w3, Hx, G), gptq3(w2, Hh, G)
            out = expert_out(Xtst, q1, q3, q2)
            r = res.setdefault((G, method), {"w1": [], "w3": [], "w2": [], "out": []})
            r["w1"].append(rel(q1, w1)); r["w3"].append(rel(q3, w3)); r["w2"].append(rel(q2, w2)); r["out"].append(rel(out, ref))
    print(f"  expert {e} done in {time.time()-t0:.0f}s", flush=True)
print(f"\n{'scheme':<14}{'bpw':>6}{'GiB/L':>7} | rel weight err w1 / w3 / w2 | expert output rel err (mean, max over experts)")
for (G, m), r in sorted(res.items()):
    f = lambda k: sum(r[k]) / len(r[k])
    b = 4 + 20 / G if m == "int4" else bpw(G)
    print(f"{'INT4' if m == 'int4' else 'INT3'} g{G:<4}{m:<5}{b:>6.2f}{gib_per_layer(b):>7.2f} | {f('w1'):.3f} / {f('w3'):.3f} / {f('w2'):.3f}        | {f('out'):.3f}  (max {max(r['out']):.3f})")
