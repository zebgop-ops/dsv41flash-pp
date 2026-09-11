import numpy as np, glob, sys, torch
for f in sorted(glob.glob(f"{sys.argv[1]}/G*_A00in_r0.npy")):
    a = np.load(f); t = torch.from_numpy(a).view(torch.bfloat16).float()
    nz = int((t.abs().sum(1) > 0).sum()); print(f.split('/')[-1], "nonzero rows:", nz)
