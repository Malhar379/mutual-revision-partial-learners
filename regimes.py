import json
# P = pre-registered primary regime. T = post-hoc "near-threshold" regime added after seeing P's null (see README).
REGIMES = {"P": dict(fa=0.18, fb=0.18, fu=0.24, fv=0.10),
           "T": dict(fa=0.30, fb=0.30, fu=0.20, fv=0.08)}
def build_cfg(regime):
    F = json.load(open("results/frozen_config.json"))
    ic = json.load(open(f"results/inter_config_{regime}.json"))
    cond_over = {c: dict(lam=v["lam"], warm=v["warm"]) for c, v in ic.items()}
    return {**F, **REGIMES[regime], "cond_over": cond_over, "lam": 0.0, "warm": 0}
