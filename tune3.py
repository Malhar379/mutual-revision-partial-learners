from tune import run
base = dict(act="square", lr=3e-3, wd=1.0, steps=6000)
for bo in [dict(lr=1e-3, wd=1.0), dict(lr=3e-3, wd=0.5), dict(lr=1e-3, wd=0.5, steps=10000), dict(lr=3e-3, wd=1.0, steps=12000)]:
    print("big_over", bo, run({**base, "big_over": bo}, conds=("pooled_big",)), flush=True)
