"""Run one seed of the regime-T experiment with the PyTorch port and compare to the stored NumPy result.
Only meaningful AFTER test_parity.py passes.
Usage: python3 run_torch.py [seed=0] [device=cpu]
Conditions: solo, dml, challenge. Results are close to, not identical to, results/raw_T_random_<seed>.json
(float32 summation order differs, and grokking is chaotic), so compare the pattern, not the digits."""
import sys, json, time
import numpy as np
import torch

import eco
import eco_torch as et
from regimes import build_cfg

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"
cfg = build_cfg("T")
a, b, y = eco.all_pairs()
X = eco.onehot(a, b)
sp = eco.make_split(seed, "random", cfg["fa"], cfg["fb"], cfg["fu"], cfg["fv"])
T = lambda x, dt=None: et.to_t(x, dt, dev)
XA, yA = T(X[sp["S_A"]], torch.float32), T(y[sp["S_A"]].astype(np.int64))
XB, yB = T(X[sp["S_B"]], torch.float32), T(y[sp["S_B"]].astype(np.int64))
XU, Xt, yt = T(X[sp["U"]], torch.float32), T(X[sp["Test"]], torch.float32), T(y[sp["Test"]].astype(np.int64))
ref = json.load(open(f"results/raw_T_random_{seed}.json"))

for cond, inter in [("solo", None), ("dml", "dml"), ("challenge", "challenge")]:
    c = {**cfg, **cfg["cond_over"].get(cond, {})}
    mA = et.TMLP(c["H"], 1, c["act"], c["init_scale"], device=dev)
    mB = et.TMLP(c["H"], 2, c["act"], c["init_scale"], device=dev)
    t0 = time.time()
    et.train_models([mA, mB], [(XA, yA), (XB, yB)], XU if inter else None, c, inter)
    got = (et.acc(mA, Xt, yt) + et.acc(mB, Xt, yt)) / 2
    exp = (ref[cond]["A"]["test"] + ref[cond]["B"]["test"]) / 2
    print(f"{cond:10s} torch {100*got:5.1f}%   numpy {100*exp:5.1f}%   ({time.time()-t0:.0f}s)")
