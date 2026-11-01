# Pre-registration (written before any experiment was run)

**Question.** Do two small models with *different data histories* that interact
(reinforce, share, challenge) end up better on held-out inputs than the controls
that do not interact, and can they approach or beat a larger model trained in
isolation on the same pooled labels?

This is a toy test of one idea ("metaemergence": capability that comes from
interaction between systems, not from scaling a single one). It is not a test of
any broader theory of intelligence.

## Setup (fixed in advance)

- Task: modular addition, `y = (a + b) mod 53`, on all 53x53 pairs. Chosen because
  it has a known generalization threshold in the amount of training data
  (grokking), so "one model has too little data, two together have enough"
  is a regime we can construct and check.
- Disjoint pools of pairs: `S_A`, `S_B` (labeled, one per model), `U` (unlabeled,
  inputs only, shared by both models for interaction), `Val` (hyperparameter
  tuning only), `Test` (never touched until final reporting).
- Models: 2-layer MLP on one-hot inputs, written in NumPy. "Small" = hidden H_s,
  "big" = hidden H_b (about 4x).
- Two split modes:
  - `random`: S_A and S_B are random disjoint subsets.
  - `region`: S_A only has pairs with a < 27, S_B only a >= 27. Test is reported
    per region, which lets us check whether interaction damages each model's own
    domain ("retention") and whether it gains the other's ("transfer").

## Conditions

| id | description | what it controls for |
|---|---|---|
| solo | each model trained on its own labels only | baseline |
| ensemble | average of solo A and solo B predictions, no interaction | "just combine two models" |
| pooled_small | one small model on S_A + S_B labels | data pooling (upper-ish bound for small) |
| pooled_big | one big model on S_A + S_B labels | the "vastly better isolated model" |
| self_teacher | each model also distills from a lagged copy of itself on U | gains from merely seeing U inputs |
| dml | A and B distill from each other on U, uniform weight | reinforce / share (deep mutual learning style) |
| agree_only | as dml, weight highest where A and B agree | reinforcement without challenge |
| challenge | as dml, weight highest where A and B disagree | challenge-driven revision |

Interaction conditions use own-label loss plus a distillation loss on U, with the
distillation weight ramped up after a warm-up. Nothing in Test or Val is ever used
for training or distillation.

## Hypotheses and decision rules

Metric: test accuracy at the **final training step** (no early stopping on Test),
mean over seeds, paired across conditions within a seed, 95% bootstrap CI over seeds.
A difference counts as real only if the CI excludes 0 **and** the mean gap is at
least 3 accuracy points.

- **H1 (interaction helps).** dml and challenge beat solo, ensemble, and self_teacher.
- **H2 (challenge matters).** challenge beats dml and agree_only.
- **H3 (the strong claim).** The best interaction condition matches or beats pooled_big.
  My prior is low (~10%). Pooled labels are close to the best information a model
  can have, so beating them would be surprising.
- **H4 (diversity cost).** Interaction reduces A/B representational diversity
  (linear CKA up, prediction disagreement down) compared with solo. I expect yes;
  the question is how much, and whether the gain survives it.
- **H5 (retention, region mode).** Interaction does not reduce each model's accuracy
  on its own region by more than 3 points.

## Allowed tuning, and when it freezes

Training hyperparameters (learning rate, weight decay, steps, activation, labeled
fractions) are tuned **only on solo / pooled_small / pooled_big, using Val**, to find
a regime where solo models under-generalize and pooled models generalize (otherwise
there is nothing for interaction to recover). The chosen values are written to
`results/frozen_config.json` before any interaction condition is run. Interaction
hyperparameters (distillation weight, temperature, warm-up) are a small fixed grid
chosen on Val and then frozen; the same grid is given to dml, agree_only, challenge.

## What a null result means

If H1 fails, the interaction mechanism used here (logit-level distillation on
unlabeled inputs) gives no benefit over its controls in this task, which says
nothing about richer channels (latent-space interaction, weight updates mid-debate).
If H1 holds and H3 fails, interaction recovers part of what pooling gives but does
not beat it, which is consistent with known co-distillation results. Both outcomes
get written up as they come out.

## Known limits (stated up front)

Modular addition is a toy. Both models solve the same task, so this tests
different data histories, not different skills. Logit-level exchange on shared
unlabeled inputs is a much narrower channel than the latent-space interaction
discussed in the motivating idea. A compositional-skills task and a richer
channel are follow-ups, not part of this run.
