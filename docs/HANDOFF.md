# HANDOFF.md — Full context transfer from claude.ai research sessions

> **Purpose:** This file captures every verified finding, decision, and open item from the
> claude.ai planning sessions (June 2025–June 2026). It supersedes `[U]` items in
> `GROUND_TRUTH.md` where noted. Read this before starting any milestone.
>
> **Convention:** `[V]` = verified against source. `[R]` = resolved (was `[U]`). `[OPEN]` = still
> unresolved.

---

## 1. Project identity (unchanged from scaffold)

Head-to-head comparison of HCT vs L2D on Galaxy Zoo binary (spiral vs early-type). Contribution
is the comparison itself (unpublished). Pareto frontier (accuracy vs expected human-query cost)
is the visualization, not the novelty.

Arms: HCT (Berger 2026), L2D-Okati (NeurIPS 2021), L2D-Mozannar (ICML 2020), AI-alone,
single-human-alone. AI-as-advisor demoted to discussion-only (Galaxy Zoo has no "shown the AI"
labels).

Framing: cost–accuracy trade-off characterization, NOT a horse race. HCT underperforms AI-alone
on accuracy across all six datasets in Berger Fig. 3. Frame HCT as auditable, transparent,
training-free human–AI integration. L2D may dominate on accuracy but requires co-training.

---

## 2. Corrections to the original brief (all [V])

These were identified in the first session and baked into GROUND_TRUTH.md, but restated here
because Claude Code may see the brief in the system prompt:

1. Galaxy Zoo 10k subset + Kaggle source are Okati **footnotes 8–9**, NOT footnote 7.
2. Raters are Galaxy Zoo **citizen-science volunteers, NOT domain experts**. Okati's own text
   says "human experts" — that is loose terminology. The report must state the non-expert
   nature explicitly because the human-accuracy ceiling and h2 tiebreak quality depend on it.
3. HCT is NOT literally two operating points. Expected human cost = 1 + P(human–AI disagree),
   **bounded in [1,2]**, traced as a coarse marker locus over the AI binarization threshold.
   Still qualitatively distinct from L2D's smooth curve — state the asymmetry, but accurately.

---

## 3. Okati 2021 repo findings [V] (github.com/Networks-Learning/differentiable-learning-under-triage)

Cloned and inspected. Key findings:

### 3a. Backbone
- `torchvision.models.resnet50()` — **scratch init, NO ImageNet pretrained weights** (the
  `pretrained` kwarg is absent in the notebook; that torchvision API version defaults to False).
- Head: `nn.Sequential(nn.Linear(2048, 2), nn.LogSoftmax(dim=-1))`, trained with `NLLLoss`.
- Input: 3×224×224, ImageNet mean/std normalization (from `prepare_data.py`).
- **Sanity target:** Okati Fig. 4(b), Galaxy Zoo, triage level b=0 (full automation) →
  misclassification P(y≠ŷ) ≈ 0.17, i.e. AI test accuracy ≈ 0.83.

### 3b. Data pipeline — CRITICAL
- `prepare_data.py` consumes Kaggle `training_solutions_rev1.csv` **aggregate vote fractions**
  (Class1.1=smooth, Class1.2=features/disk).
- Ground-truth label: `Y = argmax(fraction)`, i.e. crowd majority.
- Simulated human: **ONE draw per image** from Bernoulli(P(spiral|x)) where P comes from the
  Kaggle aggregate fraction. NOT a real per-rater draw.
- Output artifact: `galaxy_data.pkl` containing (features, Y, h_pred) — **no per-rater counts**.
- The 10k subset is `sorted(filenames)[:10000]` — lexicographic sort on 6-digit GalaxyIDs
  (100008–999967), which equals numeric sort.
- Split: **UNSEEDED** `random.shuffle` in `prepare_data.py` (60/20/20). We discard this and
  use a fresh **70/15/15, seed=0** split.

### 3c. No per-rater labels in repo [R]
- **The repo ships NO per-image rater counts for Galaxy Zoo.** This was GROUND_TRUTH §4 Data
  Risk #1. It is now resolved: the counts do not exist in the repo. HCT's h1/h2 draw requires
  external data (the GZ2 catalog). This motivated the entire Option A vs Option B investigation.

---

## 4. Option A vs Option B — the central data decision

### Option A: Simulated single-draw expert (Okati's own pipeline)
- Draw h ~ Bernoulli(P(spiral|x)) using Kaggle aggregate fractions.
- Pro: zero additional data work; matches Okati's and Mozannar's own experimental regime.
- Con: **biased in favor of L2D arms** — this is exactly L2D's home field. A reviewer would
  flag that HCT is tested on a protocol it was not designed for.

### Option B: Real per-rater labels from GZ2 catalog
- Join Kaggle GalaxyID → GZ2 dr7objid → pull raw vote counts → build per-image urns →
  draw h1/h2 without replacement.
- Pro: gives HCT a legitimate test with real human–human error correlation.
- Con: requires recovering the GalaxyID→dr7objid mapping (Kaggle anonymized its IDs).

**Decision: Option B**, justified on validity grounds. The per-rater pipeline is reframed as a
small technical contribution.

---

## 5. The GalaxyID→dr7objid crosswalk — FULLY RESOLVED

This was the hardest data-engineering problem in the project. Full resolution below.

### 5a. Exact-ID join: DEAD [V]
- Hypothesis: Kaggle GalaxyID == GZ2 asset_id (from gz2_filename_mapping.csv).
- Test: `verify_galaxyid_assetid.py` — membership 100% (IDs overlap numerically), but
  **Task-01 content correlation r ≈ 0** (smooth r=0.001, features r=0.000, artifact r=-0.009).
- **VERDICT: REJECT.** Kaggle anonymized/renumbered its IDs. The integers happen to overlap
  in range but map to different galaxies. No CSV join via asset_id exists.
- Script is in repo: `verify_galaxyid_assetid.py`. Hardened with `dtype={'GalaxyID':'int64'}`
  etc. to prevent float64 corruption on large objids.

### 5b. Content-based crosswalk: WORKS [V]
- Recover GalaxyID→dr7objid by matching 37-dim vote-signature vectors.
- Both datasets describe the same GZ2 vote distributions. L1-normalize each task block on
  both sides (undoes Kaggle's path scaling), then nearest-neighbor match via cKDTree.
- Accept only **mutual nearest neighbors** (A's NN is B, AND B's NN is A).
- Script: `match_galaxyid_to_dr7objid.py` (v3, debiased flavor, multi-table support).

### 5c. Table/flavor findings [V]
- **Hart et al. 2016 (`gz2_hart16.csv.gz`): WRONG TABLE for matching.** Hart reprocessed the
  GZ2 vote reduction ~2 years after the Kaggle competition. All three flavors (raw fraction,
  weighted_fraction, debiased) give ~37–41% mutual-NN with ~0.30 median distance. None
  collapses to ~0 for any flavor → reduction-version mismatch, not a column choice.
- **Willett et al. 2013 tables (`zoo2MainSpecz`, `zoo2MainPhotoz`): CORRECT.** The Kaggle
  competition was built on the contemporaneous Willett reduction. Debiased flavor of Willett
  matches Kaggle at median per-dim difference 0.0000 (essentially perfect).
- Matching against both Willett tables (specz + photoz) gives **5,904 mutual-NN pairs (59.0%)**
  at perfect precision (median per-dim diff = 0.0001).

### 5d. Coverage analysis [V]
- 59% coverage is the hard plateau with the two main Willett tables.
- The ~27% "absent" galaxies are genuinely not in `zoo2MainSpecz` + `zoo2MainPhotoz`. They
  are likely from the GZ2 Stripe-82-normal subsample (`dr10_gz2_stripe82_normal`), which uses
  a separate debiasing and lives in a different table.
- The ~14% remaining are ambiguous signatures (degenerate vote profiles, e.g. unanimous-smooth
  ellipticals) that cannot be uniquely matched. Correctly rejected by the mutual-NN criterion.
- **The matched subset is morphologically representative:** binary-hardness 0.52 ≈ overall
  0.48. Not skewed toward easy or hard galaxies.
- 5,904 galaxies → 70/15/15 split → ~885 test instances. Ample statistical power.

### 5e. Fallback options (not exercised)
- **Stripe-82 table:** could recover some of the 27%, but GZ2 notes S82 debiasing is
  "slightly different" — matches may not be reliable. Gamble, not recommended.
- **Image perceptual-hash route:** download `images_gz2.zip` (~3 GB), hash both image sets,
  match directly. Would give ~100% coverage but is heavy. Only worth it if a reviewer demands
  full coverage.
- **Option A fallback:** run A with the L2D-home-field bias declared in limitations, use
  matched subset as corroboration. This is the floor if B fails entirely.

### 5f. Current recommendation (accepted)
Run Option B on the 5,904 mutual-matched subset. Disclose in the writeup:
> "The HCT arm's real per-rater labels are recovered for 59% of the 10k via a
> debiased-vote-fraction crosswalk to Willett et al. 2013; the unmatched remainder is largely a
> GZ2 subsample (likely Stripe 82) absent from the main catalog tables, not a
> morphology-selected drop."

---

## 6. Urn construction for HCT [OPEN — next deliverable]

For each matched dr7objid, the urn builder must:
1. Read `t01_smooth_or_features_a01_smooth_count` and `t01_smooth_or_features_a02_features_or_disk_count`
   from the Willett table (raw counts, NOT debiased fractions — keep matching vs urn-building separate).
2. Drop galaxies whose Task-01 mode is `star_or_artifact` (mirrors Okati's Class1.3 filter).
3. Form binary urn `{smooth_count, features_count}` per image.
4. Draw h1, h2 without replacement (hypergeometric from N ≥ 30 votes; 2 draws).

**Important nuance:** Two draws without replacement from a single fixed urn are
*hypergeometrically (slightly negatively) correlated* within that urn. The positive
human–human correlation Berger cares about lives in the *between-image* variance of urn
composition (hard galaxies → near-50/50 urns → frequent h1/h2 disagreement). The report must
describe this correctly so a reviewer doesn't read "without-replacement draw" as "negatively
correlated humans, contradicting Berger."

**Structural limits of the urn approach (disclose in report):**
- h1 ⊥ h2 | x is structurally enforced (conditional independence given the image).
- h2 is not a "second human" in any real sense — it's a second draw from the same distribution.
- The opinion-leader α reconstruction (Berger's ρ analysis) is NOT supported — that requires
  identity-linked per-classification rows, not aggregate counts.
- The "30+ labels per image" property is not fully exploited — we use only counts, not per-rater IDs.

---

## 7. Mozannar & Sontag 2020 [partially verified]

- Repo: `github.com/clinicalml/learn-to-defer` — Jupyter-notebook-based.
- Surrogate: consistent convex surrogate `L_CE^α` for the defer objective. Learns classifier +
  rejector jointly via cost-sensitive reduction.
- Deferral cost parameter → smooth Pareto curve (analogous to Okati's `b`, different mechanism).
- Their experiments: CIFAR-10H (multiclass, violates our binary constraint), hate-speech (3-class,
  synthetic expert), chest X-rays (synthetic experts). **Zero binary real-multi-rater datasets.**
- [OPEN] The surrogate loss lives in notebooks; must be extracted and verified against
  arXiv:2006.01862 §3 before any number is trusted. This is estimated as the highest-effort arm.
- [OPEN] Confirm repo runs on PyTorch ≥2.x.

---

## 8. HCT mechanism [V] (arXiv:2602.02375, Fig. 1 + Introduction)

- Human h1 and AI decide independently on binary label.
- Agree → accept label, cost = 1.
- Disagree → sample h2 (second human), take h2's call, cost = 2.
- At least one human always approves the decision.
- Expected cost = 1 + P(human–AI disagree | threshold), continuous in [1, 2].
- The AI binarization threshold is the ONLY sweep parameter → coarse marker locus on Pareto plot.
- **HCT is a decision-rule paper (cs.HC), not a trainable algorithm.** "Implementing HCT" =
  applying the decision rule over multi-rater labels + a binarized AI classifier. There is no HCT
  model to train, no HCT loss function.
- HCT has NEVER been applied to Galaxy Zoo — this is part of the project's novelty.

---

## 9. Validity threat — ground truth is crowd consensus [V]

The ground-truth label y = argmax_h P(h|x) is the crowd majority, NOT an objective truth.
Consequences:
- A single sampled human's accuracy equals the majority vote share by construction.
- HCT's tie-breaker h2, drawn from the same P(h|x) that defines y, is correct ~50% on near-split
  images by construction. HCT cannot beat consensus where the label itself is a coin flip.
- This likely compresses inter-arm gaps and weakens claims about HCT's complementarity in
  Berger's sense.
- **Report must disclose this prominently:** frame accuracy as "agreement with crowd consensus,"
  not "agreement with truth."

---

## 10. Cross-arm invariants (from scaffold, unchanged)

1. **Same AI backbone** for all arms. If L2D co-trains its classifier with its rejector (Okati,
   Mozannar), document the deviation and report whether the co-trained classifier differs
   materially from the shared backbone.
2. **Same frozen test split:** seed=0, 70/15/15, on the matched ~5,904 subset.
3. **Same human-label model** P(h|x) feeds: single-human baseline, HCT h1/h2, L2D "human" term.
4. **Same metric definitions** computed by one shared module. No arm computes its own metrics.

---

## 11. Files the user has on disk (available to Claude Code)

### CSVs (in `csvs/` or similar local directory)
- `training_solutions_rev1.csv` — Kaggle Galaxy Zoo Challenge, 61,578 rows, aggregate vote fractions
- `gz2_filename_mapping.csv` — Zenodo 3565489, 355,990 rows, columns: objid, sample, asset_id
- `gz2_hart16.csv.gz` — Hart et al. 2016, 239,695 rows (NOT useful for matching — wrong reduction)
- `zoo2MainSpecz.csv.gz` — Willett 2013 spectroscopic-z main sample (CORRECT for matching)
- `zoo2MainPhotoz.csv.gz` — Willett 2013 photometric-z main sample (CORRECT for matching)

### Scripts (created during sessions, may need to be placed in repo)
- `verify_galaxyid_assetid.py` — the exact-ID join gate (verdict: REJECT)
- `match_galaxyid_to_dr7objid.py` (v3) — the content-based crosswalk (verdict: 5,904 mutual pairs)
- `train_c0.py` — AI-alone backbone training script (Okati-faithful ResNet-50, draft, not yet run)
- `galaxyid_to_dr7objid.csv` — output crosswalk file (filter on `mutual==True` for the 5,904 pairs)

### Cloned repos
- Okati 2021: `github.com/Networks-Learning/differentiable-learning-under-triage` — vendor in
  `third_party/okati2021/`
- Mozannar 2020: `github.com/clinicalml/learn-to-defer` — vendor in `third_party/mozannar2020/`

### Project PDFs (in project knowledge)
- `HCT_paper.pdf` — Berger et al. 2026 (arXiv:2602.02375), the originating HCT paper
- `Berger_HCT.pdf` — Berger et al. 2026 (arXiv:2603.29866), the AI-as-advisor follow-up
- `Okati2021.pdf` — Okati, De & Gómez-Rodríguez, NeurIPS 2021 (arXiv:2103.08902)
- `mozannar2020.pdf` — Mozannar & Sontag, ICML 2020 (arXiv:2006.01862)

---

## 12. What has NOT been done yet

1. **Urn builder** — script to extract raw vote counts from Willett tables for the 5,904 matched
   galaxies and construct binary urns for h1/h2 draws. Next deliverable.
2. **Backbone training** — `train_c0.py` exists as a draft but has not been executed. Must land
   near Okati's ~0.83 accuracy sanity target.
3. **Okati L2D arm** — needs to be wired to the shared backbone and shared split. The upstream
   notebook uses its own unseeded split; replace with ours.
4. **Mozannar L2D arm** — highest-effort arm. Extract `L_CE^α` from notebooks, verify against
   the paper, wire to shared backbone and split.
5. **HCT arm** — decision rule implementation over urns + binarized AI. Sweep the threshold.
6. **Evaluation module** — shared metrics, bootstrap CIs, Pareto plot.
7. **Report** — 2–3 page technical report.

---

## 13. Papers to cite but NOT implement

- De Toni, Okati, Thejaswi, Straitouri & Gómez-Rodríguez (NeurIPS 2024): prediction sets +
  greedy algorithm. Multiclass — different protocol. Cite for framing only.
- Mozannar & Sontag are MIT CSAIL, NOT affiliated with Gómez-Rodríguez. The two L2D lines
  diverged after 2020.

---

## 14. Rejected datasets — do NOT revisit

- CIFAR-10H, ImageNet-16H: multiclass. HCT undefined.
- HAM10000: no released per-rater labels for h2 draw.
- Brinker: 200-image eval set destroys statistical power.
- Good Judgment Open: probabilistic forecasting with resolution lag.

---

## 15. Non-negotiable conventions

- Binary classification only.
- Tests before implementation. An arm is not "done" until its tests existed first and pass.
- Use upstream code as baseline. Do not reimplement from scratch unless original is broken.
- State the HCT/L2D curve-shape asymmetry in every plot and caption.
- No invented citations. If uncertain, say so and verify against the source.
- Deterministic seeds everywhere. Bootstrap CIs on all reported numbers.
- No sycophancy. No filler. State uncertainty explicitly.
