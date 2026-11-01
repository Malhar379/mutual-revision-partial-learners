"""Parity check: PyTorch port vs NumPy reference. RUN THIS FIRST on a machine with PyTorch.

Both backends start from identical weights and data, train for a few hundred full-batch steps, and
we compare final logits and predictions on the Test pool. Small float32 drift is expected; a large
difference means a bug in the port. Usage: python3 test_parity.py
"""
import json, os
import numpy as np
import torch

import eco
import eco_torch as et

torch.set_num_threads(1)
F = json.load(open("results/frozen_config.json"))
CFG = {**F, "fa": 0.30, "fb": 0.30, "fu": 0.20, "fv": 0.08,
       "lam": 3.0, "warm": 10, "ramp": 20, "self_refresh": 50, "wclip": 10, "log_every": 10**9}
STEPS = int(os.environ.get("PARITY_STEPS", 150))  # if a borderline MISMATCH appears, retry with PARITY_STEPS=50
LOGIT_TOL = 5e-2      # max abs logit difference allowed after STEPS steps of Adam
PRED_AGREE = 0.98     # fraction of Test predictions that must match


def run(inter, seed=0):
    cfg = {**CFG, "steps": STEPS}
    a, b, y = eco.all_pairs()
    X = eco.onehot(a, b)
    sp = eco.make_split(seed, "random", cfg["fa"], cfg["fb"], cfg["fu"], cfg["fv"])
    XA, yA, XB, yB = X[sp["S_A"]], y[sp["S_A"]], X[sp["S_B"]], y[sp["S_B"]]
    XU, yU, Xt = X[sp["U"]], y[sp["U"]], X[sp["Test"]]
    use_u = inter is not None
    # NumPy reference
    mA, mB = (eco.MLP(cfg["H"], 1, cfg["act"], cfg["init_scale"]), eco.MLP(cfg["H"], 2, cfg["act"], cfg["init_scale"]))
    eco.train_models([mA, mB], [(XA, yA), (XB, yB)], XU if use_u else None, cfg, inter, None, yU=yU)
    ref = [m.forward(Xt)[0] for m in (mA, mB)]
    # PyTorch port
    tA, tB = (et.TMLP(cfg["H"], 1, cfg["act"], cfg["init_scale"]), et.TMLP(cfg["H"], 2, cfg["act"], cfg["init_scale"]))
    L = lambda X_, y_: (et.to_t(X_, torch.float32), et.to_t(y_.astype(np.int64)))
    et.train_models([tA, tB], [L(XA, yA), L(XB, yB)], et.to_t(XU, torch.float32) if use_u else None, cfg, inter,
                    yU=et.to_t(yU.astype(np.int64)))
    with torch.no_grad():
        out = [m(et.to_t(Xt, torch.float32))[0].numpy() for m in (tA, tB)]
    d = max(float(np.abs(r - o).max()) for r, o in zip(ref, out))
    agree = min(float((r.argmax(1) == o.argmax(1)).mean()) for r, o in zip(ref, out))
    return d, agree


if __name__ == "__main__":
    ok = True
    for inter in [None, "dml", "challenge", "agree", "self", "oracle"]:
        d, ag = run(inter)
        good = d < LOGIT_TOL and ag >= PRED_AGREE
        ok &= good
        print(f"{str(inter):10s} max|logit diff| = {d:.2e}   prediction agreement = {100*ag:.1f}%   {'OK' if good else 'MISMATCH'}")
    print("\nPARITY OK" if ok else "\nPARITY FAILED: do not trust the port until this passes")
