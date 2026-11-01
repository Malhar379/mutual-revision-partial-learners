"""Usage: python3 run_all.py <regime P|T> <mode random|region> <n_seeds>. Resumable: skips finished seeds."""
import sys, json, os
from multiprocessing import Pool
from regimes import build_cfg
from eco import run_seed

def job(a):
    reg, mode, seed = a
    path = f"results/raw_{reg}_{mode}_{seed}.json"
    if os.path.exists(path):
        return path
    r = run_seed(seed, mode, build_cfg(reg))
    json.dump(r, open(path + ".tmp", "w"))
    os.replace(path + ".tmp", path)
    return path

if __name__ == "__main__":
    reg, mode, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with Pool(2) as p:
        for r in p.imap_unordered(job, [(reg, mode, s) for s in range(n)]):
            print("done", r, flush=True)
