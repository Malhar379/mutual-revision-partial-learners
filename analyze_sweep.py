"""Summarize results/sweep_*.json -> results/sweep.md and results/fig_sweep.png"""
import glob, json, re, collections
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # scripts read results/ relative to the repo root, wherever they are launched from
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = collections.defaultdict(list)
for f in glob.glob("results/sweep_*.json"):
    fr, seed = re.findall(r"sweep_([\d.]+)_(\d+)\.json", f)[0]
    runs[float(fr)].append(json.load(open(f)))
fracs = sorted(runs)
pm = lambda r, c: r[c]["M"]["test"] if "M" in r[c] else (r[c]["A"]["test"] + r[c]["B"]["test"]) / 2
S = [("solo", "solo", "#9a9992"), ("pooled_small", "pooled labels (small)", "#e87ba4"),
     ("dml", "mutual, uniform", "#2a78d6"), ("agree_only", "mutual, agreement-weighted", "#eb6834"),
     ("challenge", "mutual, challenge-weighted", "#1baf7a")]

md = ["# Labeled-fraction sweep (post-hoc, seeds 10-12; regime-T distillation settings, not re-tuned)", "",
      "Mean test accuracy at the final step; per-seed range in brackets. n seeds per row shown.", "",
      "| labeled / model | n | " + " | ".join(l for _, l, _ in S) + " |", "|---|---|" + "---|" * len(S)]
fig, ax = plt.subplots(figsize=(8, 5.0), dpi=150)
for c, lab, col in S:
    ys = []
    for f in fracs:
        v = np.array([pm(r, c) for r in runs[f]]) * 100
        ys.append(v)
    ax.plot([100 * f for f in fracs], [v.mean() for v in ys], "-o", color=col, lw=2, ms=4, label=lab)
    for f, v in zip(fracs, ys):
        ax.scatter([100 * f] * len(v), v, s=8, color=col, alpha=0.35, zorder=1)
for f in fracs:
    cells = []
    for c, _, _ in S:
        v = np.array([pm(r, c) for r in runs[f]]) * 100
        cells.append(f"{v.mean():.1f}% [{v.min():.0f}-{v.max():.0f}]")
    md.append(f"| {100*f:.0f}% | {len(runs[f])} | " + " | ".join(cells) + " |")
ax.set_xlabel("labeled data per model (% of all 53x53 pairs)", fontsize=9, color="#52514e")
ax.set_ylabel("test accuracy (%) at final step", fontsize=9, color="#52514e")
ax.set_ylim(-3, 105)
ax.grid(color="#e7e6e2", lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
ax.set_title("Where does interaction help? Dots are individual seeds", fontsize=10, loc="left")
fig.tight_layout()
fig.savefig("results/fig_sweep.png")
open("results/sweep.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))