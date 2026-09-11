"""Compare graph-replay per-layer dumps (G*_L*_r*.npy, bf16 bits) with eager dumps (L*_r*.pt).
usage: diff_gdump.py <gdump_dir> <eager_dir> [call_offset=0]"""
import glob, os, re, sys, numpy as np, torch, torch.nn.functional as F
G, E = sys.argv[1], sys.argv[2]; off = int(sys.argv[3]) if len(sys.argv) > 3 else 0
files = {}; files_e = {}
for f in glob.glob(os.path.join(E, "G*_*_r*.npy")):
    m = re.match(r"G(\d+)_([A-Za-z0-9]+)_r(\d+)\.npy", os.path.basename(f))
    files_e.setdefault((m.group(2), int(m.group(3))), []).append((int(m.group(1)), f))
for f in glob.glob(os.path.join(G, "G*_*_r*.npy")):
    m = re.match(r"G(\d+)_([A-Za-z0-9]+)_r(\d+)\.npy", os.path.basename(f))
    n, l, r = int(m.group(1)), m.group(2), int(m.group(3)); files.setdefault((l, r), []).append((n, f))
def load(f):
    a = np.load(f)
    return torch.from_numpy(a).view(torch.bfloat16).float() if a.dtype == np.int16 else torch.from_numpy(a).float()
for (l, r) in sorted(files):
    seq = sorted(files[(l, r)])
    if off >= len(seq) or -off > len(seq): continue
    n, f = seq[off]
    g = load(f)
    if os.path.isdir(E) and glob.glob(os.path.join(E, "G*_%s_r%d.npy" % (l, r))):
        eseq = sorted(files_e[(l, r)]) if (l, r) in files_e else []
        if off >= len(eseq) or -off > len(eseq): continue
        e = load(eseq[off][1]); T = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    else:
        ef = os.path.join(E, f"{l}_r{r}.pt")
        if not os.path.exists(ef): print(f"{l} r{r}: no eager file"); continue
        e = torch.load(ef)["hidden_states"].float(); T = e.shape[0]
    if g.ndim == 1 or e.ndim == 1 or g.shape[1] != e.shape[1]:
        print(f"{l:10s} r{r}: shape mismatch graph {tuple(g.shape)} vs eager {tuple(e.shape)}"); continue
    gg = g[:T].reshape(T, -1); ee = e[:T].reshape(T, -1)
    err = (gg - ee).abs().max().item(); cos = F.cosine_similarity(gg.flatten(), ee.flatten(), dim=0).item()
    print(f"{l:10s} r{r} (call G{n:03d}): max abs {err:.4g}  cos {cos:.6f}  eager |h| {ee.abs().mean():.4f}  graph |h| {gg.abs().mean():.4f}{'  <-- DIFF' if cos < 0.9999 else ''}")
