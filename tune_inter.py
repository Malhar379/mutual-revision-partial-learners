"""Pick distillation weight / warm-up per interaction condition on Val, tuning seed 100 (not an eval seed)."""
import json, itertools, sys
from multiprocessing import Pool
from eco import run_seed

from regimes import REGIMES
REG = sys.argv[1]
F = {**json.load(open("results/frozen_config.json")), **REGIMES[REG]}
CONDS = ["self_teacher", "dml", "agree_only", "challenge"]
GRID = [(lam, warm) for lam in (1.0, 3.0) for warm in (300, 1500)]

def job(args):
    c, lam, warm = args
    cfg = {**F, "lam": lam, "warm": warm}
    r = run_seed(100, "random", cfg, [c])[c]
    return c, lam, warm, (r["A"]["val"] + r["B"]["val"]) / 2

if __name__ == "__main__":
    jobs = [(c, l, w) for c in CONDS for (l, w) in GRID]
    with Pool(2) as p:
        res = p.map(job, jobs)
    best = {}
    for c, l, w, v in res:
        print(c, l, w, round(v, 3))
        if c not in best or v > best[c][2]:
            best[c] = (l, w, v)
    json.dump({c: dict(lam=b[0], warm=b[1], val=b[2]) for c, b in best.items()}, open(f"results/inter_config_{REG}.json", "w"), indent=1)
    print(best)
