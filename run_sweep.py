"""Labeled-fraction sweep. Usage: python3 run_sweep.py. Resumable. Seeds 10-12 (not used for tuning or main eval).
Uses the regime-T distillation settings unchanged at every fraction (not re-tuned). U fixed at 20% of pairs."""
import sys, json, os
from multiprocessing import Pool
from regimes import build_cfg
from eco import run_seed

FRACS = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34]
SEEDS = [10, 11, 12]
CONDS = ["solo", "pooled_small", "dml", "agree_only", "challenge"]

def job(a):
    f, seed = a
    path = f"results/sweep_{f:.2f}_{seed}.json"
    if os.path.exists(path):
        return path
    cfg = {**build_cfg("T"), "fa": f, "fb": f, "fu": 0.20, "fv": 0.02}
    r = run_seed(seed, "random", cfg, CONDS)
    json.dump(r, open(path + ".tmp", "w"))
    os.replace(path + ".tmp", path)
    return path

if __name__ == "__main__":
    jobs = [(f, s) for s in SEEDS for f in FRACS]
    with Pool(2) as p:
        for r in p.imap_unordered(job, jobs):
            print("done", r, flush=True)
