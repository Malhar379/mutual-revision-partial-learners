import json
from multiprocessing import Pool
from eco import run_seed
F = json.load(open("results/frozen_config.json"))
def job(f):
    cfg = {**F, "fa": f, "fb": f, "fu": 0.20, "fv": 0.08}
    r = run_seed(100, "random", cfg, ["solo", "pooled_small"])
    return f, r["solo"]["A"]["val"], r["solo"]["B"]["val"], r["pooled_small"]["M"]["val"]
if __name__ == "__main__":
    with Pool(2) as p:
        for f, a, b, m in p.map(job, [0.22, 0.26, 0.30]):
            print(f"frac/model={f:.2f} soloA={a:.3f} soloB={b:.3f} pooled_small={m:.3f}", flush=True)
