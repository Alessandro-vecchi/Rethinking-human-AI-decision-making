# M5-PLAN — L2D-Mozannar arm (Option B), PHASE-1 result

Companion to `tasks/M5-l2d-mozannar.md`. This records the PHASE-1 vendor + extract + **verification**
outcome that gates PHASE-2 implementation.

## 1. Vendored upstream

- Repo: `github.com/clinicalml/learn-to-defer` → `third_party/mozannar2020/`.
- **Pinned SHA: `e84f3ee719fe9a9637883eb22ae622717c786bb1`** (promotes GROUND_TRUTH §8 Mozannar-SHA [U]→[V]).
- Notebook-based (no packaged API). The `L_CE^α` reference implementation lives in
  `cifar/cifar10_defer_ours.ipynb`: `reject_CrossEntropyLoss(outputs, m, labels, m2, n_classes)`
  + the `m`/`m2` cost assignment in `train_reject(...)`.
- torch compat: project `.venv` has **torch 2.2.2** (≥2.x). The loss is plain torch ops
  (index + `log2` + weighted sum); no port needed. (Resolves HANDOFF §7 / GROUND_TRUTH §8 torch-compat [OPEN].)

## 2. Surrogate verification (the correctness gate) — VERIFIED ✓

Upstream `train_reject` sets, per sample: `if m==y: m=1, m2=alpha  else: m=0, m2=1`, and
`reject_CrossEntropyLoss = −m·log2(softmax_⊥) − m2·log2(softmax_y)`. Substituting:

`L_CE^α = −(α·1{m=y} + 1{m≠y})·log softmax_y(g) − 1{m=y}·log softmax_⊥(g)`

This **exactly matches paper eq (10)** (arXiv:2006.01862 §4); `α=1` ⇒ `L_CE` (eq 7). 

> **Note:** the upstream notebook *docstring* for `m2` ("`α·I_{m≠y} + I_{m=y}`") is mislabeled —
> it swaps the indicator. The **code** (and our port) follow eq (10): α multiplies `1{m=y}` on the
> classifier term. We cite the code, not the docstring.

Immaterial deviation: upstream uses `log2(softmax)`; we compute natural `log_softmax` (numerically
stable). These differ only by the constant `1/ln2`, which rescales the loss/lr and never changes the
minimizer or the argmax deferral rule.

Test-time rule (upstream `metrics_print` + paper eq 6): `predicted = argmax over K+1 outputs`; defer
iff that argmax is the reject index ⊥ (i.e. `g_⊥ ≥ max_{y∈Y} g_y`); else predict `argmax_{y∈Y} g_y`.

## 3. Option-B construction (frozen classifier) + consistency

Class logits are FROZEN from the M2 backbone, reconstructed from `score`:
`g_1=log(score)`, `g_0=log(1−score)` ⇒ `softmax(g_0,g_1)=(1−score,score)`, `exp(g_0)+exp(g_1)=1`.
Trainable scalar rejector `g_⊥(x)=MLP(emb)` ⇒ `Z=1+exp(g_⊥)`, `softmax_⊥=σ(g_⊥)`,
`softmax_y=p_y/(1+exp(g_⊥))`. Defer iff `g_⊥ ≥ log(max(score,1−score))`.

`L_CE^α` is convex in **g**, hence convex in `g_⊥` alone (affine coordinate). At **α=1**, eq (9)/Prop 2
give the minimizer over `g_⊥` (given fixed near-Bayes class logits) = the Bayes rejector
`r^B = 1{max_y η_y ≤ P(Y=M|x)}`. The M2 backbone approximates the Bayes classifier (trained `argmax η`),
so freezing it and learning only `g_⊥` is a **consistent rejector surrogate at α=1**; α≠1 deliberately
shifts the operating point. → proceed (no STOP). Documented as the Option-B deviation.

## 4. Modeling decisions (settled with user)

- Sweep knob = **α** over `configs/arms.yaml l2d_mozannar.defer_cost_sweep [0.1,0.2,0.5,1.0,2.0,5.0]`
  (small α → more deferral; large α → automate-everything → AI-alone 0.8285). α is a reweighting
  hyperparameter, analog to Okati's `b`, not a literal cost.
- Expert term = **expected correctness `q(x)=P(m=y|x)` = urn vote-share matching y_debiased =
  `1 − expected_human_loss`** (soft, deterministic; Rao-Blackwell of eq 10's indicator, matches the
  population loss eq 8). Rejector fit ONCE per α; rater seeds 0–4 enter only at the realized
  test-time human draw (mirrors L2D-Okati Option B).

## 5. PHASE-2 (tests-first → code)

Tests (`tests/arms/test_l2d_mozannar.py`): correctness gate (`l_ce_alpha`==eq10; α=1==`L_CE`;
m=y vs m≠y branches; soft q==hard); finite-diff gradient check on g_⊥; logit reconstruction;
frozen-classifier consistency at α=1 (recovers `r^B`); single-batch overfit; deferral
monotone-decreasing in α; automate-everything==0.8285; cost accounting (`_assert_cost_unit`);
frozen-split hash; binary schema; determinism.

Module (`src/haidc/arms/l2d_mozannar.py`): `l_ce_alpha`, `fit_rejector_mozannar` (MLP→scalar g_⊥),
`mozannar_defer_mask`, `run_alpha_sweep`, `operating_points` (keyed on `alpha`), `run`,
`_wire_defaults`, `main`; register in `run_all.py`. Reuse okati helpers
(`ai_label_from_score`, `expected_human_loss`, `_draw_humans_per_seed`, `embedding_feature_cols`,
`validate_embeddings_frame`, `_assert_binary/_assert_cost_unit/_assert_frozen_split/_assert_scores_match`)
and the shared `eval/` metrics + `data/sampler`.

Outputs: `results/l2d_mozannar_predictions.parquet`
`[GalaxyID,y_debiased,ai_label,defer,human,decision,human_queries,alpha,seed,policy="mozannar"]`;
`results/l2d_mozannar_operating_points.csv`
`[alpha,policy,accuracy_mean,accuracy_lo,accuracy_hi,cost_mean,deferral_fraction,n_seeds]`.
