"""Control: solo models trained 4x longer. Usage: python3 run_long.py <regime> <n_seeds>. Writes results/long_<regime>_<seed>.json"""
import sys, json, os
from multiprocessing import Pool
from regimes import build_cfg
from eco import run_seed

def job(a):
    reg, seed = a
    path = f"results/long_{reg}_{seed}.json"
    if os.path.exists(path):
        return path
    r = run_seed(seed, "random", build_cfg(reg), ["solo_long"])
    json.dump(r, open(path + ".tmp", "w"))
    os.replace(path + ".tmp", path)
    return path

if __name__ == "__main__":
    reg, n = sys.argv[1], int(sys.argv[2])
    with Pool(2) as p:
        for r in p.imap_unordered(job, [(reg, s) for s in range(n)]):
            print("done", r, flush=True)
