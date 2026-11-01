import time
from tune import run
for g in [dict(act="square", lr=3e-3, wd=1.0, steps=6000),
          dict(act="square", lr=3e-3, wd=2.0, steps=6000),
          dict(act="square", lr=1e-2, wd=1.0, steps=6000),
          dict(act="square", lr=1e-2, wd=2.0, steps=6000)]:
    print(g, run(g, conds=("solo", "pooled_small", "pooled_big")), flush=True)
