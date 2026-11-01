"""Extra control: two models with the SAME labeled data interact (dml_same) vs not (solo_same).
Usage: python3 run_extra.py <regime> <n_seeds>. Random split only. Writes results/extra_<regime>_<seed>.json"""
import sys, json, os
from multiprocessing import Pool
from regimes import build_cfg
from eco import run_seed

def job(a):
    reg, seed = a
    path = f"results/extra_{reg}_{seed}.json"
    if os.path.exists(path):
        return path
    r = run_seed(seed, "random", build_cfg(reg), ["solo_same", "dml_same"])
    json.dump(r, open(path + ".tmp", "w"))
    os.replace(path + ".tmp", path)
    return path

if __name__ == "__main__":
    reg, n = sys.argv[1], int(sys.argv[2])
    with Pool(2) as p:
        for r in p.imap_unordered(job, [(reg, s) for s in range(n)]):
            print("done", r, flush=True)
