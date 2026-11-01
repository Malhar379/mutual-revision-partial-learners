"""Leakage check: pools must be disjoint in both modes, and region mode must respect regions."""
from eco import make_split, all_pairs, HALF
a, b, y = all_pairs()
for mode in ("random", "region"):
    for reg in ({"fa": .18, "fb": .18, "fu": .24, "fv": .10}, {"fa": .30, "fb": .30, "fu": .20, "fv": .08}):
        for seed in range(6):
            sp = make_split(seed, mode, **reg)
            names = list(sp)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    assert not set(sp[names[i]]) & set(sp[names[j]]), (mode, seed, names[i], names[j])
            assert len(sp["Test"]) > 300
            if mode == "region":
                assert (a[sp["S_A"]] < HALF).all() and (a[sp["S_B"]] >= HALF).all()
print("splits OK: all pools disjoint, regions respected")
