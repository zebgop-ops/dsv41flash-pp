"""Compare two DSV41_DEBUG_DUMP directories layer by layer (attn/ffn in+out, per-layer h/residual).
usage: diff_dumps.py <dirA> <dirB>"""
import glob, os, sys, torch, torch.nn.functional as F
A, B = sys.argv[1], sys.argv[2]
def cmp(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    if a.numel() != b.numel(): return f"shape {a.numel()} vs {b.numel()}"
    err = (a - b).abs().max().item(); cos = F.cosine_similarity(a, b, dim=0).item() if a.norm() > 0 else float('nan')
    return f"maxabs {err:.4g} cos {cos:.6f}"
for f in sorted(glob.glob(os.path.join(A, "L*_r*_attn.pt")) + glob.glob(os.path.join(A, "L*_r*_ffn.pt"))):
    name = os.path.basename(f); g = os.path.join(B, name)
    if not os.path.exists(g): print(name, "missing in B"); continue
    da, db = torch.load(f), torch.load(g)
    same_in = torch.equal(da["x"], db["x"]); same_out = torch.equal(da["out"], db["out"])
    flag = "" if (same_in and same_out) else "  <-- DIFF"
    print(f"{name:20s} in: {'identical' if same_in else cmp(da['x'], db['x'])}   out: {'identical' if same_out else cmp(da['out'], db['out'])}{flag}")
    if "input_ids" in da and da["input_ids"] is not None and db.get("input_ids") is not None and not torch.equal(da["input_ids"], db["input_ids"]):
        print("   input_ids differ:", da["input_ids"].tolist()[:12], "vs", db["input_ids"].tolist()[:12])
