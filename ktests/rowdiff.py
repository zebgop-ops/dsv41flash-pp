import numpy as np, torch, glob, sys
R = sys.argv[1] if len(sys.argv) > 1 else "/r"
def load(d, key, r):
    fs = sorted(glob.glob(f"{R}/{d}/G*_{key}_r{r}.npy")); a = np.load(fs[0])
    return torch.from_numpy(a).view(torch.bfloat16).float() if a.dtype == np.int16 else torch.from_numpy(a)
for key, r in (("L00", 0), ("L01", 0), ("EG0hash", 0)):
    e, f, a = load("gd-eager", key, r), load("gd-pw-fresh", key, r), load("gd-pw-after", key, r)
    print(key, "row max|after-eager| rows0..7:", [round((a[i].float()-e[i].float()).abs().max().item(), 3) for i in range(8)])
    print(key, "row max|fresh-eager| rows0..7:", [round((f[i].float()-e[i].float()).abs().max().item(), 3) for i in range(8)])
h_e, h_a, h_f = load("gd-eager", "EG0hash", 0), load("gd-pw-after", "EG0hash", 0), load("gd-pw-fresh", "EG0hash", 0)
print("EG0hash[:6,:4] eager:", h_e[:6, :4].tolist())
print("EG0hash[:6,:4] fresh:", h_f[:6, :4].tolist())
print("EG0hash[:6,:4] after:", h_a[:6, :4].tolist())
