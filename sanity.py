import json
from eco import run_seed
F = json.load(open("results/frozen_config.json"))
cfg = {**F, "lam": 1.0, "warm": 300}
r = run_seed(100, "random", cfg, ["oracle"])["oracle"]
print("oracle peer: A val", r["A"]["val"], "B val", r["B"]["val"], "A test", r["A"]["test"])
