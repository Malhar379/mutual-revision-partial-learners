# Mutual Revision Between Partial Learners

**Do two models that learned from different data end up better by revising their predictions toward each other?**

📄 [Full technical writeup (PDF)](report/technical_note.pdf)

# Research Question

Two small models each learn from a *different* slice of labeled data, and neither has enough to solve the task alone. If they can also see a shared pool of **unlabeled** inputs and revise their predictions toward each other's, do they learn something that neither learns alone, or that two non-interacting models do not learn together?

## Approach

- **Task:** modular addition `(a + b) mod 53`. It has a known data-amount threshold for generalization (grokking), so "one model has too little data, two together have enough" can be constructed and checked.
- **Models:** two small MLPs (A, B), written in NumPy. A and B get disjoint labeled slices (`S_A`, `S_B`) and share an unlabeled pool `U`. Nothing in Val or Test is ever used for training.
- **Interaction:** mutual distillation on `U` (each model learns from the other's predictions), uniform or weighted toward agreement or toward disagreement ("challenge").
- **Controls:** each model alone, a non-interacting ensemble, a model that distills from a lagged copy of itself, pooled labels (small and 4x wider model), a peer with the same labeled data, and solo training for 4x longer.
- **Pre-registered:** hypotheses (H1 to H5), conditions and the decision rule are in [PREREG.md](PREREG.md), written before any run. A difference counts only if its 95% bootstrap CI over seeds excludes 0 **and** it is at least 3 accuracy points.

## Result

1. **Pre-registered setting (18% labeled per model): a null.** Every model, alone or interacting, scores 0% on test (4 seeds). Pooled labels reach 75% (small) and 93% (wider). Two models that know nothing cannot teach each other anything.
2. **Near the threshold (30% labeled per model): a large effect, added after seeing the null, so exploratory.** Solo reaches 32.7%, a non-interacting ensemble 43.1%, mutual distillation 99.7%. The gap over solo is +67.0 points (95% CI 62.3 to 71.5). Pooled labels reach 100%, so interaction recovers essentially all of what pooling gives, but never exceeds it.
3. **It needs different data histories.** With the *same* labeled slice, mutual distillation gives 21.3% (27.5% without it). With different slices, 99.7%.
4. **It is not extra training time.** A solo model trained 4x longer ends at 22.1%.
5. **It is a band, not a point.** Sweeping the labeled fraction, nothing happens below about 22%, the jump to near 100% comes between roughly 24% and 28%, and above that pooling already suffices.
6. **Weighting is unresolved.** Agreement- and challenge-weighting are indistinguishable from each other. At 26% labeled both are about 11 to 12 points above uniform, with 3 seeds and overlapping ranges.

## What This Does and Doesn't Establish

In one toy task, two partial learners with different data can push each other across a generalization threshold that neither crosses alone, using only each other's predictions on unlabeled inputs. It does not show this is new: the mechanism resembles co-training and deep mutual learning. It does not test latent-space interaction, other architectures, language models or different *skills*, and modular addition's sharp threshold probably flatters the effect. The positive result sits in a setting chosen after the pre-registered one failed, and the interaction settings were tuned there. This is a pre-registered null plus an exploratory band, reported as such.

## Repository Structure

```
├── PREREG.md             # hypotheses and decision rules, written before any run
├── eco.py                # NumPy core: data splits, models, interaction losses
├── eco_torch.py          # PyTorch port, checked against the NumPy code
├── regimes.py            # regime definitions
├── tune*.py              # tuning on Val only
├── run_all.py            # main runs (regimes P and T, random and region splits)
├── run_extra.py          # same-data control
├── run_long.py           # solo trained 4x longer
├── run_sweep.py          # labeled-fraction sweep
├── run_torch.py          # one full seed through the PyTorch port
├── analyze.py            # tables, bootstrap CIs, figures -> results/summary.md
├── analyze_sweep.py      # sweep table and figure -> results/sweep.md
├── test_splits.py        # leakage check
├── test_parity.py        # PyTorch vs NumPy parity check
├── results/              # raw runs, summary.md, sweep.md, figures
└── requirements.txt
```

## Running This Project

```bash
git clone https://github.com/Malhar379/mutual-revision-partial-learners.git
cd mutual-revision-partial-learners
pip install numpy matplotlib          # torch only for test_parity.py and run_torch.py

python3 test_splits.py                                      # leakage check
python3 run_all.py T random 6                               # main near-threshold result
python3 run_all.py T region 6; python3 run_all.py P random 4  # region split, pre-registered null
python3 run_extra.py T 6; python3 run_long.py T 6           # same-data and 4x-longer-solo controls
python3 run_sweep.py                                        # labeled-fraction sweep (~30 min)
python3 analyze.py && python3 analyze_sweep.py              # tables, CIs, figures -> results/
```

Each full seed takes a few minutes on CPU. The tuned settings are already saved in `results/`, so the tuning scripts (`tune*.py`) don't need to be rerun.

## Limitations

- One toy task with a sharp threshold, small MLPs, 4 to 6 seeds per setting, so the intervals are crude. The sweep has 3 seeds per point.
- The effect appears only in a narrow band of data amounts, and the interaction hyperparameters were tuned in the post-hoc setting.
- Interaction is logit-level only. Latent-space exchange, weight updates and compositional skills are untested.
- Both models solve the same task, so this tests different data histories, not different skills.