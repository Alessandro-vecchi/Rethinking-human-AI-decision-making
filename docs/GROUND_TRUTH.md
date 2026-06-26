# GROUND_TRUTH.md

Every method/dataset fact an agent states must trace to this file or to a citation it has
checked. Entries are marked **[V]** verified against the source in this repo / a checked web
source, or **[U]** unverified — treat **[U]** as an open question, not a fact. Do not promote a
**[U]** to a fact without checking. Do not add facts here you have not verified.

Verification basis: the four papers are in the project as image-scan archives; the claims below
were read directly from the page images / bundled OCR on 2026-06-13, except where noted.

---

## 1. The four papers (identifiers)

| Key | Title | Venue / ID | Role in this project |
|---|---|---|---|
| **HCT-1** | The Hybrid Confirmation Tree: A robust strategy for hybrid intelligence | arXiv:2602.02375v1, 2 Feb 2026, cs.HC. Berger, Analytis, Andersen, Lorenzen, Satopää, Kurvers | **[V]** Originating HCT paper. Defines the rule + analytic derivations. |
| **HCT-2** | Beyond AI advice—independent aggregation boosts human–AI accuracy | arXiv:2603.29866v1, 31 Mar 2026, cs.HC. Berger, Analytis, Satopää, Kurvers | **[V]** Follow-up: HCT vs AI-as-advisor, 10 datasets, signal-detection-theory analysis. |
| **Okati** | Differentiable Learning Under Triage | NeurIPS 2021, arXiv:2103.08902v4. Okati, De, Gómez-Rodríguez (MPI-SWS / IIT Bombay) | **[V]** L2D arm (MGR line). Provides the Galaxy Zoo dataset + repo. |
| **Mozannar** | Consistent Estimators for Learning to Defer to an Expert | ICML 2020 (PMLR v119:7076–7087), arXiv:2006.01862. Mozannar, Sontag (MIT CSAIL) | **[V]** Parallel L2D arm. **Not** a Gómez-Rodríguez paper; the two L2D lines diverged after 2020. |

Framing-only (cite, do **not** implement as an arm): **[V]** De Toni, Okati, Thejaswi,
Straitouri & Gómez-Rodríguez, NeurIPS 2024 — prediction sets / greedy algorithm. It is the
current "provably" paper from MGR's group but operates on **multiclass prediction sets** — a
different protocol. Out of scope as a comparison arm.

> Correction to the original brief: the brief noted uncertainty over the two HCT arXiv IDs. Both
> are real and correctly attributed (HCT-1 = 2602.02375, HCT-2 = 2603.29866). HCT-1 is dated
> "February 3, 2026" and HCT-2 "April 1, 2026" on their title pages — note the latter date if it
> matters for citation hygiene, but the cs.HC arXiv stamps are authoritative.

---

## 2. HCT — mechanics and cost model  **[V]** (HCT-1 abstract, §intro p.2; HCT-2 abstract)

The rule, applied per instance:
1. Elicit one human judgment `h1` and one AI judgment, **independently**.
2. If they agree → that label is the decision. (1 human query.)
3. If they disagree → a second human `h2` breaks the tie; `h2` is the decision. (2 human queries.)
4. A human always approves the final decision (human-in-the-loop is preserved by construction).

**Cost model (human-query-cost, the project's x-axis).** Per instance: 1 on agreement, 2 on
disagreement. Expected human cost = **1 + P(human–AI disagree)**, therefore **bounded in [1, 2]**.

**Operating points.** Human labels are fixed (drawn from the rater pool, §4). The only continuous
knob in *this instantiation* is the **AI binarization threshold**: sweeping it changes the AI's
(TPR, FPR), which changes the disagreement rate, which moves *both* accuracy and expected cost.
This traces a **coarse** locus of points, qualitatively unlike L2D's smooth deferral-cost curve.

> Refinement to the brief's hard constraint: HCT is **not** literally "two points (cost 1 / cost
> 2)." Each AI threshold gives one (accuracy, expected-cost) point with expected cost in [1, 2];
> the set of points is small/coarse, not smooth. Plot it as markers, not a continuous line, and
> state the asymmetry in the caption. (CLAUDE.md invariant #4.)

**Claims HCT makes about itself** (context, not targets to reproduce): accuracy up to ~10 pts over
3-person majority vote while using 28–44% fewer human choices; more flexible TPR/FPR navigation
than hierarchy/polyarchy heuristics. HCT-1's own datasets are skin-cancer, deepfake, geopolitical
forecasting, criminal rearrest — **[V]** **HCT never used Galaxy Zoo.** Applying the HCT rule to
Galaxy Zoo is part of this project's novelty, not a replication.

HCT is a **human-subjects / decision-rule paper (cs.HC), not a trainable algorithm.** "Implementing
HCT" = applying the decision rule over multi-rater labels + a binarized AI classifier. There is no
HCT model to train and no HCT loss.

---

## 3. Okati 2021 — method  **[V]** (arXiv:2103.08902v4 §2–§4)

- Setup (§2): find a triage policy `π(x) ∈ {0,1}` (0 = model predicts, 1 = defer to human) and a
  model `m(x)`, minimizing `L(π,m) = E[(1−π)·ℓ(m(x),y) + π·ℓ(h,y)]` subject to `E[π(x)] ≤ b`,
  where `b` caps the deferral fraction. (Eq. 1–2.)
- **Theorem 3:** for any fixed `m`, the optimal triage policy is a **deterministic threshold rule**
  on the per-instance gap `E_y[ℓ(m(x),y)] − E_h[ℓ(h,y)]`. (This is the L2D-Okati deferral mechanism.)
- Practical algorithm (Alg. 1): alternating `TrainModel` / `TrainTriage`, with a parametric
  `π̂_γ(x)` fit at the end to approximate the optimal policy at test time (needed because the
  oracle policy depends on labels/human predictions unavailable at test time). Converges to a local
  minimum (Prop. 6).
- The **deferral budget `b`** is the sweep parameter → produces L2D-Okati's **smooth** Pareto curve.
- Okati's own baselines (Fig. 4): confidence-based triage [Bansal 2021], score-based [Raghu 2019a],
  **surrogate-based [Mozannar & Sontag 2020]**, full automation. (So Okati already benchmarks
  against Mozannar's surrogate — useful sanity reference, but our framing is HCT-vs-L2D.)

---

## 4. Galaxy Zoo dataset  **[V]** (Okati §"Experimental setup", p.9, fns 8–9)

- Binary: labels ∈ {**early type**, **spiral**}. Satisfies HCT's binary constraint.
- `|D| = 10,000` images. **[V]** A randomly chosen subset of the original 61,577 (Okati **fn 8**).
- **30+ human labels per image. [V]** Okati's text calls them "human experts."
- Per-image label distribution `P(h|x) ∝ n_x(h)` (number of raters choosing label `h`); the
  **ground-truth label is the rater majority**: `y = argmax_h P(h|x)`. **[V]**
- Features: galaxy images → **deep residual network (He et al. 2015)** representation. **[V]**
- Pixel-map source: Kaggle `galaxy-zoo-the-galaxy-challenge` (Okati **fn 9**). **[V]**

> Corrections to the brief:
> - It is **footnote 8** (10k subset) and **footnote 9** (Kaggle source), **not** footnote 7.
> - The raters are **Galaxy Zoo citizen-science volunteers, not domain experts.** Okati's own
>   wording ("human experts") is loose; the report must state the non-expert nature explicitly,
>   because the human-accuracy ceiling and the h2 tiebreak quality both depend on it. This is a
>   correction *to the source's terminology*, to be stated as such.

**Why this dataset and not others** — see §5.

**Data risk #1 (the single most important verification before any arm runs):** HCT's `h1`/`h2`
draw needs, per image, either the raw individual rater votes or the per-label vote counts
`(n_early, n_spiral)` with `n_early + n_spiral ≥ 2`. Drawing two distinct labels without
replacement is then a hypergeometric draw of 2 from the `N ≥ 30` votes — counts suffice; raw
per-rater identities are not required. **[U] It is unverified whether Okati's released repo ships
the per-image counts or only the final majority label `y`.** If only `y` is shipped, the h1/h2
draw is impossible from the inherited data and the data agent must locate the counts (Kaggle
solutions file / Galaxy Zoo DR) before HCT can run. Resolve this in M1; do not assume.

---

## 5. Rejected datasets — do not revisit  **[V]** (brief + paper constraints)

- **CIFAR-10H, ImageNet-16H:** multiclass. HCT undefined. (Note: CIFAR-10H appears in Mozannar's
  *own* experiments — that is why Mozannar has no binary multi-rater split to inherit; see §6.)
- **HAM10000:** no released per-rater labels → no `h2` draw.
- **Brinker:** 200-image eval set → destroys statistical power.
- **Good Judgment Open:** probabilistic forecasting with resolution lag, not instance-level
  classification with fixed labels.

Galaxy Zoo is the only candidate that is binary, has 30+ real per-image labels, and ships with one
original author's codebase (Okati), eliminating one porting cost.

---

## 6. Mozannar & Sontag 2020 — method + repo  **[V]** (arXiv:2006.01862; repo checked 2026-06-13)

- Consistent **convex surrogate** for the defer objective, generalizing cross-entropy; denoted
  `L_CE^α`. Learns a classifier + a rejector jointly via a reduction to cost-sensitive learning.
- Deferral cost parameter → smooth Pareto curve (analogous role to Okati's `b`, different mechanism).
- Official repo: **`github.com/clinicalml/learn-to-defer`** **[V]** — Jupyter-notebook based;
  their experiments include **CIFAR-10H** (multiclass) and synthetic experts.
- Consequence **[V]**: there is **no binary real-multi-rater dataset in their repo to inherit a
  split from**. For Galaxy Zoo we use a **fresh seeded split: 70/15/15, seed = 0**, documented in
  the repo and recorded in `docs/DECISIONS.md`. The shared split (CLAUDE.md invariant #2) overrides
  any split logic inside the vendored notebooks.

**Porting note [U]:** the surrogate-loss + rejector training is inside notebooks, not a packaged
API. The Mozannar arm requires *extracting* `L_CE^α` and the rejector training loop into
`src/haidc/arms/`, then feeding it the shared backbone features + frozen split. Estimate this as
the highest-effort arm. Verify the loss implementation against Eq. in §3 of arXiv:2006.01862 before
trusting any number.

---

## 7. Cross-arm invariants (restated; these are what make the comparison valid)

- **Same AI backbone** for: AI-alone baseline, the HCT "AI" judgment, and the L2D classifier
  component, wherever feasible. If an L2D method must co-train its classifier with its rejector
  (Okati, Mozannar), document the deviation and report whether the co-trained classifier differs
  materially from the shared backbone — a confound to disclose, not hide.
- **Same frozen test split** (seed=0, 70/15/15) for every reported number.
- **Same human-label model** `P(h|x)` from the 30+ rater pool feeds: the single-human baseline,
  HCT's `h1`/`h2`, and the "human" term in both L2D losses.
- **Same metric definitions** (accuracy, TPR/FPR, expected human-query-cost) computed by one
  shared module in `src/haidc/eval/`. No arm computes its own metrics.

## 8. Open questions log (resolve, then move the answer up with a [V] and a citation)
- [U] Does Okati's repo ship per-image rater counts for Galaxy Zoo? (Data risk #1, §4.)
- [U] Exact storage format of Okati's Galaxy Zoo features/labels in the repo (ResNet embeddings? raw images?).
- [U] Mozannar surrogate exact form to re-derive in code (§6 porting note).
- [U] Pinned commit SHAs for both vendored repos (record in DECISIONS.md once cloned).
- [U] Whether to reuse Okati's exact ResNet features for the shared backbone or train a fresh one
      (decide in M2; whichever is chosen must be used by all arms).
