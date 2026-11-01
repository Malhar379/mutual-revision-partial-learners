"""Tune ONLY solo / pooled_small / pooled_big on Val (per PREREG). No interaction conditions here."""
import sys, time, json, itertools
import numpy as np
from eco import *

base = dict(H=128, Hbig=512, act="square", init_scale=0.5, lr=1e-3, wd=1.0, steps=3000,
            fa=0.18, fb=0.18, fu=0.24, fv=0.10, log_every=500,
            warm=0, ramp=1, lam=0, wclip=10, self_refresh=100)

def run(over, seed=0, mode="random", conds=("solo", "pooled_small")):
    cfg = {**base, **over}
    t = time.time()
    r = run_seed(seed, mode, cfg, list(conds))
    s = {}
    if "solo" in r: s["soloA_val"] = r["solo"]["A"]["val"]; s["soloB_val"] = r["solo"]["B"]["val"]
    if "pooled_small" in r: s["pool_s_val"] = r["pooled_small"]["M"]["val"]
    if "pooled_big" in r: s["pool_b_val"] = r["pooled_big"]["M"]["val"]
    s["sec"] = round(time.time() - t, 1)
    return s

if __name__ == "__main__":
    grid = [dict(act=a, lr=lr, wd=wd) for a in ("square", "relu") for lr in (1e-3, 3e-3) for wd in (0.3, 1.0)]
    for g in grid:
        print(g, run(g), flush=True)
