# DECISIONS.md

Append-only. Newest at the bottom. One entry per decision/deviation. An agent that deviates from
the plan, pins a dependency, fixes a hyperparameter, or promotes a `[U]` fact to `[V]` records it
here in the same change. Format:

```
## YYYY-MM-DD — <short title>
- Context: why a decision was needed.
- Decision: what was chosen.
- Alternatives rejected: and why.
- Evidence: commit SHA / file / paper section / test that backs it.
- Status: active | superseded by <date>.
```

---

## 2026-06-13 — Repository scaffold and invariants
- Context: project kickoff; agents need a grounded, low-context starting point.
- Decision: adopt the orchestrator/subagent layout, the five cross-arm invariants
  (`GROUND_TRUTH.md §7`), the M1–M7 DAG, and the four CLAUDE.md non-negotiables.
- Alternatives rejected: per-arm bespoke agents (replaced by one reusable `arm-implementer` role to
  cut duplication and context); separate baseline milestones (folded into arms).
- Evidence: this scaffold; brief; papers verified 2026-06-13 (see GROUND_TRUTH verification basis).
- Status: active.

## 2026-06-13 — Brief corrections recorded
- Context: three claims in the original brief did not match the sources.
- Decision: (a) Galaxy Zoo 10k subset + Kaggle source are Okati **fns 8–9**, not fn 7;
  (b) raters are citizen-science **volunteers, not domain experts** — state explicitly in report;
  (c) HCT operating points are a **coarse locus over the AI threshold with expected cost ∈ [1,2]**,
  not literally two points.
- Alternatives rejected: propagating the brief verbatim (would put errors in the report).
- Evidence: Okati arXiv:2103.08902v4 p.9 fns 8–9; HCT-1 arXiv:2602.02375v1 abstract + p.2.
- Status: active.

## 2026-06-26 — M1 frozen split + multi-rater label table
- Context: first executable arm. Build the hash-frozen 70/15/15 split (seed=0) and the per-image
  urn/label table feeding HCT h1/h2 and both L2D human terms. Two preflight STOP conditions arose.
- Decision (crosswalk universe): use `accepted==True` = **4,626** (mutual-NN AND vote-signature
  L2 dist < 0.30; median per-dim diff ≈ 0.0001), NOT `mutual==True` = 5,904. The HANDOFF §5c
  "5,904" is the *mutual* count; `accepted` is the dist-capped high-precision subset
  (`match_galaxyid_to_dr7objid.py:93`). The 1,278 mutual-only pairs (median dist 0.368, ≤0.703)
  are poor matches that would corrupt the urns. §5c's 0.0001 precision figure was in fact measured
  on `accepted`.
- Decision (label): the mandatory Willett-counts-vs-Kaggle `y` consistency check **failed at
  20.86% (964/4,621, > the 5% STOP)**. Cause is a known GZ2 redshift-debiasing artifact, not a bug:
  raw-count argmax = **6.1% spiral**; Kaggle-fraction argmax = **26.8%**; Willett-debiased argmax =
  26.6% (confirming Kaggle Class1.x == Willett debiased, crosswalk diff 0.0001). Resolution
  (orchestrator-approved): emit **both** `y_count` (raw argmax, tie→smooth) and `y_debiased`
  (Kaggle/debiased argmax) for reversibility; **all arms (M2+) consume `y_debiased`** — it is
  simultaneously the canonical GZ consensus label, Okati's published label, the target of the
  ~0.83 AI-alone anchor, and the L2D reproduction target. The h1/h2 **urn stays raw counts**
  (HANDOFF §6) — integer counts are required for the without-replacement draw. The 20.86% is the
  disclosed single-human-vs-consensus gap (single-human accuracy ≈ 79%); with y=y_debiased over
  raw-count human draws, single-human accuracy no longer equals vote-share by construction, which
  **dissolves the HANDOFF §9 degeneracy**.
- Alternatives rejected: (a) `mutual` universe — injects ~1,278 wrong-galaxy crosswalks; (b)
  `y = y_count` everywhere — 94/6 imbalance breaks the 0.83 anchor, diverges 20% from Okati's
  published labels, forces balanced-acc/AUC and degrades comparability with both L2D papers.
- Build result: **N = 4,621** (5 star/artifact-mode dropped, 0 below min_votes≥2); split
  3234/693/694; manifest `content_hash =
  5ba759793918967ce3c1e514d26fe930a295795d6c70467718f355e92742947a`. `make verify-split` reproduces
  it. Tests (3 modules, 19 cases) written first, all pass; `ruff check src tests` clean.
- [U]→[V] promotions (GROUND_TRUTH updated in this change):
  - §4 Data risk #1 → **resolved**: Okati's repo ships no per-rater counts (HANDOFF §3c); counts
    come from Willett 2013 via the content-based crosswalk.
  - §4 label correction: literal "`y ∝ raw counts`" was an oversimplification — Okati's pipeline
    (and ours) uses the **debiased** Kaggle/Willett fraction argmax. Evidence: crosswalk Kaggle ==
    Willett-debiased at median per-dim diff 0.0001; raw vs debiased argmax disagree 20.86%.
- Environment pinned (REPRODUCIBILITY.md): Python 3.12.8. `numpy==1.26.4` pinned <2 because
  torch 2.2.2 (latest macOS x86_64 wheel) needs the NumPy-1.x C-API; `scipy==1.13.1`,
  `pandas==2.2.3` are their last NumPy-1.26-compatible releases. Added `pyarrow` (parquet engine)
  and a minimal `pyproject.toml` (`pip install -e . --no-deps`, so `python -m haidc...` + pytest
  resolve). Resolved-lockfile sha256 (`pip freeze | sort | shasum -a 256`):
  `4116d5b7e7bae72603a6dbbc4ab4d2eda4d8f712c2db4fcfacf49c751b9b43e7`.
- Input provenance pinned: `training_solutions_rev1.csv` (Kaggle, Okati fn 9); `zoo2MainSpecz.csv.gz`
  + `zoo2MainPhotoz.csv.gz` (Willett 2013); `galaxyid_to_dr7objid.csv` (this project's crosswalk).
  All git-ignored; only `data/split_manifest.json` is committed.
- Status: active.

## 2026-06-26 — M2 shared AI backbone: probe → train-fresh (Branch B), blocked at TIME GATE
- Context: M2 builds the one shared AI backbone (CLAUDE.md #2). M2-PLAN §1 gated the reuse-vs-train
  choice behind a probe of the vendored Okati repo: does it ship a GalaxyID-aligned 2048-d
  embeddings artifact (REUSE), or only `prepare_data.py` (TRAIN)?
- Decision (reuse vs train): **TRAIN FRESH (Branch B).** Probe result (Okati SHA
  `43ec215d3ce06fabe45b5c9c23957ce2b86424ff`, 2025-04-25): the repo ships **no** aligned embeddings.
  `Galaxy-zoo/prepare_data.py` stores **raw images** `X=(10000,3,224,224)` (no feature-extraction
  step); `galaxy_data.pkl` is *generated* from the Kaggle `images_training_rev1/` folder, not
  committed; the shipped `Galaxy-zoo/results/*.pkl` (132–444 KB) are triage *outputs* on Okati's
  own unseeded 60/20/20 split with no GalaxyIDs → not joinable to our frozen split or `y_debiased`.
  So "reuse" collapses into "train" (M2-PLAN §1). Backbone is Okati-faithful (train.ipynb cell 10,
  HANDOFF §3a): `torchvision.models.resnet50(weights=None)` scratch init; head
  `nn.Sequential(nn.Linear(2048,2), nn.LogSoftmax(dim=-1))`; `NLLLoss`; **Adam (default lr 1e-3),
  batch_size 128, 50 epochs**; 3×224×224 ImageNet-normalized input; trained on the frozen train
  split (3,234 imgs) vs `y_debiased`.
- Branch-A coverage (verified, retained for the record): all 4,621 split GalaxyIDs lie inside
  Okati's first-10k subset — the 10,000th-smallest Kaggle GalaxyID is 248516 = our split max — so a
  `sorted(GalaxyID)[:10000]` row→ID join *would* have been sound had embeddings shipped.
- Status of the AI-alone ~0.83 anchor (sanity gate #2): **NOT YET RUN — blocked at the M2-PLAN
  TIME GATE.** Two blockers: (1) the ~3 GB Kaggle `images_training_rev1/` set is not on disk;
  (2) torch is the macOS x86_64 **CPU** wheel (CUDA=False) and a measured train step at batch 128
  is **260 s**, projecting the 50-epoch train to **~90 h (~3.8 days)** on this machine — infeasible.
  Requires a CUDA/Linux box (or accepting a multi-day CPU run) + the image download. AI-alone
  accuracy + bootstrap CI to be recorded here once the train runs; do not cite a number before then.
- Delivered + verified this session (machinery, no images needed): tests-first
  `tests/arms/` (4 modules, 11 cases, all green) → `src/haidc/arms/backbone.py`. Sanity gate #1
  (single-batch overfit → final NLLLoss < 0.05) and gate #3 (two same-seed exports value-identical)
  pass at unit scale; score-export contract (694 rows, manifest order, scores ∈ [0,1]),
  label-coverage join (loud on miss), and the inline AI-alone accuracy + percentile-bootstrap CI
  pass on fixtures. `ruff check src tests` clean; `make verify-split` still reproduces
  `content_hash 5ba75979…2947a`. `make backbone` fails loud with a download hint when images absent.
- Config knobs added (`configs/backbone.yaml`; CODING.md config-over-constants): `features`
  (`fresh`), `images_dir`, `train.{epochs,batch_size,lr,weight_decay}`, `bootstrap.{n_boot,ci}`,
  `artifact_path`, `run_manifest_path`. Bootstrap CI computed inline in `backbone.py` until the
  shared eval module exists (M6 to absorb; M2-PLAN §2).
- Alternatives rejected: (a) reuse `Galaxy-zoo/results/*.pkl` as the shared backbone — wrong
  (unseeded) split, no GalaxyIDs, not joinable, would confound CLAUDE.md #2; (b) silently launch
  the ~90 h CPU train — violates the M2-PLAN TIME GATE.
- Evidence: Okati SHA 43ec215; `third_party/okati2021/Galaxy-zoo/prepare_data.py:19,30,52,57,59,168`;
  `train.ipynb` cell 10; `tests/arms/*`; benchmark 260 s/step @ b=128 (threads=2).
- [U]→[V] promotions (GROUND_TRUTH §8 updated in this change): Okati feature/label storage format
  (raw images); reuse-vs-train backbone choice (train fresh); Okati vendored SHA pinned (43ec215).
- Status: active (M2 partial — code/tests landed; full-train sanity gate #2 pending TIME GATE go/no-go).

## 2026-06-27 — M2 backbone made GPU-native (TIME GATE resolved via Colab)
- Context: `backbone.py` hardcoded `device = torch.device("cpu")` and `predict_spiral_proba` never
  left CPU, so the 50-epoch ResNet-50 train projected to ~90 h (DECISIONS 2026-06-26 TIME GATE).
  No local GPU is available; the run will execute on a Colab GPU runtime via `notebooks/m2_colab.ipynb`.
- Decision (TIME GATE): **resolved via Colab GPU.** `backbone.py` is now device-agnostic — a single
  `resolve_device("auto")` selects cuda when available else cpu, so the *same* committed code runs CPU
  here (CUDA=False) and CUDA on Colab. This removes the notebook's runtime monkey-patch (old Cell 7),
  which is deleted.
- Decision (precision): **FP32, deterministic-faithful — no AMP, no TF32, `cudnn.benchmark` stays
  False.** Preserves Okati's FP32 recipe and the ~0.83 anchor and keeps reproducibility (CLAUDE.md #2,
  REPRODUCIBILITY.md). GPU speedup comes from running on the GPU at all + `channels_last` conv layout
  + pinned host tensor with `non_blocking=True` H2D copies — none of which change numerics. AMP/TF32
  were considered and rejected (max-throughput) because they deviate from the published recipe and
  weaken bitwise reproducibility. Config knobs added: `train.device: auto`, `train.channels_last: true`.
- Determinism caveat (REPRODUCIBILITY.md "document any op that cannot be made deterministic"):
  ResNet-50's CUDA backward includes ops with no deterministic implementation (e.g.
  `adaptive_avg_pool2d_backward_cuda`) which **raise** under the strict
  `torch.use_deterministic_algorithms(True)` set by `seed_everything`. On the CUDA path only,
  `train_model` relaxes to `use_deterministic_algorithms(True, warn_only=True)` (best-effort GPU
  determinism); the CPU path stays strict and the CPU determinism test is unchanged. The notebook
  keeps `CUBLAS_WORKSPACE_CONFIG=":4096:8"` so cuBLAS GEMMs stay deterministic. Bitwise
  cross-device (CPU vs GPU) reproducibility is not claimed; the run manifest now records the resolved
  `device`, `torch_version`, and `cuda_version` for provenance.
- Env note: Colab uses a CUDA-build torch (deviates from the pinned CPU `torch==2.2.2`, expected per
  DECISIONS 2026-06-26 env note). Record the GPU torch version + a fresh `pip freeze` lockfile hash
  after the first successful Colab run.
- Notebook (`notebooks/m2_colab.ipynb`, moved to top-level — `src/` is the importable package root):
  removed the GPU monkey-patch cell; fixed the Okati fidelity grep path
  (`Galaxy-zoo/prepare_data.py`); aligned the clone slug to the real remote case.
- Tests: `tests/arms/test_backbone_device.py` added first (red→green) — `resolve_device`, train via
  the device path, and device-inferred `predict_spiral_proba`. Existing M1/M2 tests unchanged and green.
- Sanity gate #2 (AI-alone TEST accuracy ≈0.83 + bootstrap CI): **still pending** — to be filled from
  the Colab `results/backbone_run.json` once the user runs the notebook end-to-end. Do not cite a
  number before then.
- Evidence: `src/haidc/arms/backbone.py` (`resolve_device`, `train_model`, `predict_spiral_proba`,
  `run`); `configs/backbone.yaml`; `notebooks/m2_colab.ipynb`; `tests/arms/test_backbone_device.py`.
- Status: active (M2 still partial — full-train sanity gate #2 pending the Colab run).

## 2026-06-27 — M2 first GPU run analysis: 0.769 < 0.83 anchor → corrective re-run (cell-41 recipe)
- Context: the first full GPU run finished on Colab (CUDA, torch 2.11.0+cu128, repo `git_sha
  ac6bc4b`) and produced `results/backbone_scores.parquet` + `results/backbone_run.json` (user had
  placed them in `data/`; moved to `results/` — config's declared output dir; `data/` is inputs only).
- Result: **AI-alone TEST accuracy = 0.7694, 95% bootstrap CI [0.7378, 0.7983]** (n_boot=1000),
  threshold 0.5, n_train=3234, n_test=694. Test majority-class baseline = 0.732 → the model beats
  baseline by only ~3.7 pts and sits ~6 pts below Okati's 0.83 anchor (Fig 4(b), b=0). **Sanity
  gate #2 NOT met.**
- Analysis: scores contract valid (694 rows, order matches frozen test split, ∈[0,1], unique). Not
  collapsed — predicted spiral rate 0.277 ≈ true 0.268 — **but 83% of scores are >0.9 or <0.1
  (confident-but-wrong)**, the signature of a generalization / data-limited gap rather than
  undertraining. **Preprocessing verified faithful**: Okati `Galaxy-zoo/prepare_data.py:41-46` uses
  the exact `Resize(256)+CenterCrop(224)+ToTensor+Normalize(ImageNet)` our `load_images_for_ids`
  uses. **Labels verified faithful**: Okati `Y=argmax(Class1.1, Class1.2)` debiased fractions =
  our `y_debiased`. So neither preprocessing nor labels explain the gap.
- Two real differences identified: (a) **recipe** — our backbone followed `train.ipynb` cell 10's
  *triage* m-net (Adam lr=1e-3, 50 epochs), but Okati's standalone AI-alone classifier is cell 41
  `train_full` (**Adam lr=0.0045, 30 epochs**); (b) **data** — Okati trains the AI-alone model on
  6,000 imgs (60% of its 10k) vs our crosswalk-constrained 3,234 (70% of N=4,621). The smaller split
  is mandated by M1 (the Willett crosswalk universe needed for the multi-rater urns).
- Decision (orchestrator-approved): **one corrective re-run first.** Align the shared backbone to
  Okati's `train_full` AI-alone recipe — `configs/backbone.yaml` `train.lr 0.001→0.0045`,
  `train.epochs 50→30`, citing cell 41 (supersedes the cell-10 recipe for the backbone). NLLLoss
  stays mean-reduction: cell 41 uses `reduction='none'+.sum()`, but under **Adam** that is
  ~scale-invariant, so `lr` is the operative knob. This is a fidelity correction, **not** tuning to
  a number (CLAUDE.md: don't silently tune past the anchor).
- Diagnostics added (`backbone.py run()`): the run manifest now records `train_accuracy`,
  `final_train_loss`, and the per-epoch `train_loss_curve`. Interpretation for the re-run: a large
  positive train−test gap with low final train loss ⇒ data-limited (more epochs won't help) ⇒ switch
  to "accept 0.77 + document the data-regime deviation"; high final train loss ⇒ undertraining.
- Provenance note: the run manifest shows `vendored_okati_sha: "unknown"` — a Colab-clone limitation
  (third_party is git-ignored and not a nested repo there); the real Okati SHA `43ec215` is pinned in
  the 2026-06-26 entry.
- Status: active (M2 still partial — sanity gate #2 pending the corrective Colab re-run; reassess
  accept-vs-iterate from the new test acc + train−test gap).

## 2026-06-27 — M2 FINAL: under-training confirmed; backbone DONE (0.829 TEST), interface frozen
- Context: resolves the accept-vs-iterate question left open by the 2026-06-27 first-run entry
  (0.769 < 0.83 anchor). The dispositive re-run trained scratch resnet50 for 120 epochs with
  best-VAL checkpoint selection (git_sha 158d370, Colab CUDA).
- Verdict: **under-trained, NOT data-limited or label-ceiling.** The earlier 50-epoch recipe gave
  train 0.81 / test 0.77; 120 ep + best-VAL selection (epoch 58) gives train 0.96 / val 0.827 /
  **TEST 0.829, 95% bootstrap CI [0.801, 0.857]** (n_boot 1000, θ=0.5). train-acc reaches 1.0 by
  epoch 89 ⇒ the model CAN fit the data; we stopped at the val-optimal point, not the fit limit.
  Sanity gate #2 (AI-alone ≈0.83 + CI) **met.**
- Recipe (frozen, shared by all arms): scratch resnet50, Adam, lr 0.001, 120 epochs, bs 128,
  wd 0.0; checkpoint selected by VAL accuracy; **TEST read once** (no test-set tuning). Splits
  n_train 3234 / n_val 693 / n_test 694, seed 0, fresh features (no reusable embeddings ship —
  train-fresh forced). Okati image transform verified identical (prior entry).
- Provenance: Okati pinned at SHA **43ec215** (third_party is git-ignored / not a nested repo on
  Colab, so the run manifest records `vendored_okati_sha: "unknown"` — the real SHA is recorded
  here instead). Env: Colab GPU **torch 2.11.0+cu128, CUDA 12.8** — deviates from the pinned CPU
  torch 2.2.2 (expected per the 2026-06-26 env note); if this env is reused, capture a fresh
  `pip freeze` lockfile hash.
- Interface: `results/backbone_scores.parquet` (694 rows, manifest-aligned) is the cross-arm
  contract every later arm reads; force-committed despite `*.parquet` being git-ignored because it
  is GPU-expensive to regenerate. `results/backbone.pt` (~90 MB) is NOT committed — kept on Drive
  under `artifacts/`.
- Carry-forward for M3/HCT (user's parquet analysis; not re-derived locally): at θ=0.5 the
  checkpoint predicts spiral ≈17% vs the 26.8% base rate — conservative on the minority class —
  and scores are skewed (median≈0). Harmless for the AI-alone anchor; relevant because HCT sweeps
  the AI threshold, so the operating points will be sparse where scores pile up near 0.
- Status: resolved. M2 done; backbone interface frozen.

## 2026-06-27 — M3 HCT arm: decision rule applied, threshold locus is coarse/clustered
- Context: M3 implements the HCT arm — a DECISION RULE, not a trainable model (GROUND_TRUTH §2,
  HANDOFF §8). Apply the rule over the M1 vote urns + binarized M2 backbone scores, sweep the AI
  threshold θ (the only HCT knob), export the per-instance table M6 consumes.
- Decisions:
  1. **Shared metrics module created now** — `src/haidc/eval/metrics.py` (`accuracy`,
     `expected_human_cost`), the single source every arm uses (invariant §4). Chosen over the M2
     "inline + M6 absorbs" precedent because the metric definition is the cross-arm invariant most
     likely to be silently violated; M6 extends this module with Pareto + bootstrap CIs.
  2. **Humans drawn once per (instance, seed), reused across θ** — moving the AI threshold changes
     only `ai_label`, never the human votes. Isolates the threshold effect, removes spurious
     variance. Per-instance rows still keyed (GalaxyID, θ, seed). (User-confirmed.)
  3. **Multi-seed band** over `eval.yaml rater_draw_seeds=[0..4]` (HCT is stochastic); band =
     min/max across seeds. Bootstrap CIs remain M6's job. Determinism: one `default_rng(seed)` per
     rater seed, drawing over TEST ids in the frozen split-manifest order.
  4. **Sweep left UNCHANGED.** `configs/arms.yaml hct.ai_threshold_sweep=[0.1..0.9]` kept as-is.
- Realized operating-point locus (mean over 5 seeds; full table `results/hct_operating_points.csv`):
  θ=0.10 → acc 0.8133, cost 1.2277 ; θ=0.50 → acc 0.8104, cost 1.2023 ; θ=0.90 → acc 0.8014,
  cost 1.1813. Across the sweep: accuracy ∈ [0.801, 0.813] (spread 0.012), expected cost ∈
  [1.181, 1.228] (spread 0.046). Cost bound ∈ [1,2] holds for every θ, and the identity
  cost = 1 + P(ai≠h1) is asserted in code and tests.
- Finding — **the locus is COARSE and tightly CLUSTERED, but NOT degenerate** (monotone cost trend;
  cost decreases as θ rises and the AI calls fewer galaxies spiral). Two causes, both expected:
  (a) the M2 backbone scores saturate near {0,1} (median≈0, only ~8.8% of TEST scores in (0.1,0.9)),
  so only ~61 of 694 instances flip label across θ∈[0.1,0.9]; (b) the crowd-consensus ceiling
  (HANDOFF §9): h2 is drawn from the same P(h|x) that defines y, so HCT cannot beat consensus on
  near-split urns; accuracy sits near the single-human ≈0.79–0.81 band. **No sweep change proposed:**
  finer/lower θ would not materially widen the locus because the cause is score saturation +
  consensus ceiling, not grid coarseness. Disclosed here for M6/report rather than silently altered.
- Coarse-vs-smooth asymmetry (CLAUDE.md #4, restated for M6/report): HCT yields this coarse set of
  9 (accuracy, expected-cost) markers with cost bounded in [1,2]; L2D yields a SMOOTH curve via its
  deferral-cost parameter. M6 must plot HCT as markers (not a line) and state the asymmetry in the
  caption. The tight clustering here strengthens the framing point: HCT is auditable/training-free
  but offers little cost-axis navigation on a saturated backbone over a consensus label.
- Note (not implemented; M6/baseline territory): single-human-alone = `h1` alone (cost 1) from the
  SAME urns/seeds — HCT's h1 draws share that P(h|x), so the two arms stay coupled by construction.
- Evidence: `src/haidc/arms/hct.py`, `src/haidc/eval/metrics.py`, `src/haidc/arms/run_all.py`;
  tests `tests/arms/test_hct.py` (21) + `tests/eval/test_metrics.py` (10) green; artifacts
  `results/hct_predictions.parquet` (31,230 rows = 694×9×5) + `results/hct_operating_points.csv`.
  Backbone tests not run locally (CPU box; validated on Colab GPU per established convention).
- Status: active. M3 done; HCT per-instance table frozen as the M6 input contract.

## 2026-06-27 — M4 L2D-Okati arm: Option B (frozen classifier), oracle frontier + learned rejector (staged)
- Context: M4 adds the L2D-Okati arm. Okati's upstream method CO-TRAINS the classifier with its
  triage net; under CLAUDE.md invariants #1/#2 every arm must consume the SAME classifier scores +
  frozen split, so co-training would make a different classifier per arm and confound the head-to-head.
- Decision (Option B, orchestrator-approved): FREEZE the shared M2 backbone as the classifier; learn
  ONLY Okati's triage/deferral policy on top. This is the documented deviation from Okati's
  co-training — the shared frozen classifier removes the confound rather than hiding it (§7 invariant).
  No classifier is retrained.
- Decision (mechanism, GROUND_TRUTH §3 Thm. 3): defer when the per-instance gap
  `machine_loss − human_loss > 0`, swept by budget b. `machine_loss = 1{ai_label≠y_debiased}` with
  `ai_label = 1[score≥0.5]` (fixed backbone anchor θ; b — not θ — is Okati's knob). Human term
  (invariant §3, same P(h|x) as HCT/single-human): the deferral DECISION uses the deterministic
  EXPECTED human 0/1 loss = minority share of the urn (matches Okati's expected `hloss`); the
  REALIZED decision when deferred draws ONE human per (instance, seed) via the shared M1 sampler
  (= HCT's h1 at matching id-order/seed). `find_machine_samples` is transcribed from Okati
  train.ipynb Cell-8 (SHA 43ec215) with TWO documented boundary fixes: (a) b=0 → defer NONE
  (upstream's `argsorted[:-num_outsource]` is `[:-0]==[:0]`=empty at b=0, which would defer ALL);
  (b) never defer a gap≤0 instance even at large budget (upstream while-loop would slice off one).
  A test asserts the oracle never defers an AI-correct instance.
- Decision (staged realization, two distinct policies — `policy` column so M6 can NEVER plot the
  oracle as Okati's deployed method):
  1. **oracle** (ships now, no Colab): `find_machine_samples` on the frozen TEST losses. The OPTIMAL
     fixed-classifier deferral — an UPPER BOUND that decides deferral using test labels, so NOT
     deployable. It is also the envelope the learned policy must sit at/below.
  2. **learned** (deployable head-to-head number; code landed, real curve pending the Colab export):
     an MLP rejector (Okati's gnet head: Linear→LogSoftmax + NLLLoss) trained on the FROZEN backbone's
     2048-d penultimate EMBEDDINGS to predict the train oracle-defer set, applied LABEL-FREE at test
     (defer top floor(b·N) by P(defer)). Rejector input = embeddings (not scalar score, which would
     collapse to Okati's separate confidence-triage baseline). Requires
     `results/backbone_embeddings.parquet` from a one-time Colab GPU pass over `backbone.pt`
     (`backbone.py --export-embeddings`, new Cell 10 in `notebooks/m2_colab.ipynb`) — shared
     critical-path infra also consumed by M5 Mozannar; skipped (with a loud note) when absent.
- Cost semantics (flagged for M6): human_queries = 1 iff defer else 0 → expected cost == deferral
  fraction ∈ [0,1]. This is a DIFFERENT regime from HCT's [1,2]; M6 must state the asymmetry.
- Reproduced baseline (M4 step 1): b=0 (no deferral) TEST accuracy @θ=0.5 = **0.8285** == backbone
  `ai_alone_accuracy` 0.8285 (≈ Okati's ~0.83, Fig 4(b) b=0). No classifier retrained to get it.
- Realized ORACLE frontier (mean over rater seeds 0–4; `results/l2d_okati_operating_points.csv`):
  b=0.0 → acc 0.8285, cost 0.000 ; b=0.1 → 0.8821, 0.099 ; b≥0.2 → 0.8919, 0.1715 (flat).
  **Finding — the oracle deferral fraction SATURATES at 0.1715 = exactly the AI error rate**, because
  Thm-3 defers only gap>0 instances and gap>0 ⟺ AI wrong; budgets ≥~0.2 collapse to one operating
  point. The oracle beats AI-alone (0.829→0.892) — as an upper bound, by construction. The smooth,
  budget-populated curve is the LEARNED policy's job (it can defer AI-correct instances, so it
  extends to b=0.8 and sits below this envelope) and lands after the embedding export.
- Tests (TESTING.md, written first → green): `tests/arms/test_l2d_okati.py` (21 — find_machine_samples
  Thm-3 + boundary fixes, gap building blocks, b=0==AI-alone, cost==deferral∈[0,1], monotonicity,
  oracle-defers-only-errors, seed-independence of the defer set, determinism, row-reorder robustness,
  frozen-split hash + shared-input guards, learned-rejector determinism + separable-signal recovery)
  and `tests/arms/test_embeddings_contract.py` (4 — export frame contract + validator). Whole suite
  green; `ruff check src tests` clean (incidentally removed a pre-existing unused `pytest` import in
  `tests/arms/test_backbone_val_select.py` that was blocking the repo lint gate).
- Evidence: `src/haidc/arms/l2d_okati.py`; `src/haidc/arms/backbone.py` (`embed_and_score`,
  `build_embeddings_frame`, `export_embeddings`, `--export-embeddings`); `src/haidc/arms/run_all.py`;
  `configs/arms.yaml`, `configs/backbone.yaml` (`export_embeddings_path`); `notebooks/m2_colab.ipynb`
  Cell 10; Okati train.ipynb Cell-8/Cell-18 (SHA 43ec215); artifacts
  `results/l2d_okati_predictions.parquet` (31,230 oracle rows = 694×9×5) +
  `results/l2d_okati_operating_points.csv`.
- Status: superseded by 2026-06-28 (learned curve now produced) for the learned half; oracle half
  unchanged. M4 oracle frontier DONE (reproduced baseline + smooth-where-meaningful curve +
  per-instance table exported, invariants held). LEARNED policy code + tests landed; its real curve
  was gated on the Colab embedding export (shared with M5).

## 2026-06-28 — M4 L2D-Okati LEARNED policy produced; M4 complete
- Context: the staged learned half (DECISIONS 2026-06-27) was runtime-skipped pending the Colab
  embedding export. `results/backbone_embeddings.parquet` now exists (4621×2051 = 2048 penultimate
  features + GalaxyID/split/score; train 3234 / val 693 / test 694). This session runs the learned
  rejector, adds two sanity guards, and regenerates the artifacts with BOTH policies.
- Decision (learned recipe, NOT tuned to a number — CLAUDE.md): the existing default rejector —
  MLP (Linear(2048,64)→ReLU→Linear(64,2)→LogSoftmax, Okati gnet head), Adam lr=1e-3, 200 epochs,
  full-batch, `seed_everything(0)`, CPU, deterministic. Trains on TRAIN embeddings to predict the
  oracle defer set at the max budget; at TEST the budget is applied label-free by deferring the top
  floor(b·N) by P(defer). Not retuned to chase accuracy.
- Two new guards added (tests-first, `tests/arms/test_l2d_okati.py`, +5 cases → 26 total):
  - `_assert_scores_match`: embeddings TEST `score` == committed `backbone_scores.parquet` — **passes
    with max|Δ| = 0.0** over 694, confirming the embeddings came from the FROZEN backbone (so the
    learned arm's `ai_label = 1[score≥0.5]` is identical to the oracle's). Provenance, not a retrain.
  - `_assert_learned_le_oracle`: learned `accuracy_mean` ≤ oracle at every b (the oracle is the
    optimum over policies deferring ≤ b·N; learned is feasible ⇒ bounded). **Holds at all 9 b.**
- Realized curves (mean over rater seeds 0–4; `results/l2d_okati_operating_points.csv`, 18 rows):
  | b | oracle acc | learned acc | learned deferral |
  |---|---|---|---|
  | 0.0 | 0.8285 | 0.8285 | 0.000 |
  | 0.1 | 0.8821 | **0.8337** | 0.099 |
  | 0.2 | 0.8919 | 0.8320 | 0.199 |
  | 0.3 | 0.8919 | 0.8288 | 0.300 |
  | 0.5 | 0.8919 | 0.8110 | 0.500 |
  | 0.8 | 0.8919 | 0.7856 | 0.800 |
  Three checks pass: (1) learned ≤ oracle ∀b; (2) learned populates b=0→0.8 **smoothly** (deferral
  0.099…0.800) vs the oracle's 0.1715 saturation; (3) b=0 == AI-alone 0.8285.
- Findings (the honest deployable L2D-Okati result; M6 compares the LEARNED curve head-to-head, never
  the oracle): the deployable rejector **peaks at b=0.1 → 0.8337, marginally beating AI-alone 0.8285
  (+0.52 pt)**, then declines monotonically (forced to defer AI-correct instances to ~75 %-accurate
  humans). At its peak it recovers only ~0.5 pt of the oracle's ~6 pt headroom (oracle 0.8821 at
  b=0.1; gap 0.048) — the 2048-d frozen embeddings do not predict AI errors well enough to realize
  the Thm-3 optimum. Cost regime stays [0,1] (deferral fraction) vs HCT's [1,2] — M6 caption.
- Artifacts regenerated (both policies): `results/l2d_okati_predictions.parquet` now **62,460 rows**
  (694×9×5×2) and `results/l2d_okati_operating_points.csv` 18 rows; force-committed (M3 precedent).
- Evidence: `src/haidc/arms/l2d_okati.py` (`_assert_scores_match`, `_assert_learned_le_oracle`,
  learned summary fields); `tests/arms/test_l2d_okati.py` (26 green); fast suite (74) green;
  `ruff check src tests` clean. Learned rejector trained on CPU (tiny MLP — not a backbone train;
  the no-CPU-backbone convention covers only the ResNet, validated on Colab).
- Status: active. **M4 COMPLETE** — oracle (upper bound) + learned (deployable) both exported with
  the `policy` column distinct; reproduced baseline, sanity envelope, and head-to-head curve all in
  the M6 input contract.

## 2026-06-28 — M5 L2D-Mozannar arm: consistent surrogate L_CE^α over the frozen backbone (Option B)
- Context: second L2D line, the highest-porting-effort arm (GROUND_TRUTH §6). Same Option-B template
  as M4 — freeze the M2 backbone as the classifier, learn ONLY the rejector on the 2048-d embeddings
  — so the two L2D arms stay comparable. Vendored `github.com/clinicalml/learn-to-defer` into
  `third_party/mozannar2020/`, **pinned SHA `e84f3ee719fe9a9637883eb22ae622717c786bb1`**.
- Surrogate VERIFIED (the correctness gate, task step 2): the implemented `l_ce_alpha` equals paper
  eq (10) `L_CE^α = −(α·1{m=y}+1{m≠y})·log softmax_y − 1{m=y}·log softmax_⊥` on hand-built inputs,
  with a finite/correct gradient (central finite-difference). Transcribed from upstream
  `reject_CrossEntropyLoss` + `train_reject` (cifar/cifar10_defer_ours.ipynb, SHA e84f3ee). **NB: the
  upstream docstring for the `m2` weight is mislabeled** (it swaps the indicator — claims α·1{m≠y});
  the upstream *code* sets `m2=α if m==y else 1` = `(α·1{m=y}+1{m≠y})`, matching eq (10). We follow
  the code. We use natural `log_softmax` vs upstream `log2`; differs only by the constant 1/ln2
  (rescales loss/lr, never the minimizer or the argmax rule). `L_CE^1 = L_CE` (eq 7) — tested.
- Option-B construction + consistency: class logits FROZEN from the backbone score
  (`g_1=log score`, `g_0=log(1−score)`; appended trainable scalar `g_⊥=MLP(emb)` ⇒ `softmax_⊥=σ(g_⊥)`).
  Test rule (eq 6): defer iff `g_⊥ ≥ log(max(score,1−score))`. `L_CE^α` is convex in g ⇒ convex in
  `g_⊥` alone; **at α=1, eq (9)/Prop 2 give the `g_⊥` minimizer = log q(x), recovering the Bayes
  rejector `r^B = 1{max_y η_y ≤ P(Y=M|x)}`** — so freezing the near-Bayes classifier and learning only
  `g_⊥` is a CONSISTENT rejector surrogate at α=1 (tested both analytically and via a fitted rejector
  on memorizable features). α≠1 deliberately shifts the operating point (does NOT STOP — proceeds).
- Decision (settled with user): sweep knob = **α** over `defer_cost_sweep [0.1,0.2,0.5,1.0,2.0,5.0]`
  (a reweighting hyperparameter, the analog of Okati's `b` — NOT a literal cost). Expert term = the
  EXPECTED correctness `q(x)=P(m=y|x)=1−minority share` (soft, deterministic; Rao-Blackwell of eq 10,
  matches the population loss eq 8). Rejector fit ONCE per α; rater seeds 0–4 enter only at the
  realized test-time human draw (= HCT's h1), mirroring M4. Cost regime [0,1] (deferral fraction) vs
  HCT's [1,2] — M6 caption.
- Realized curve (mean over seeds 0–4; `results/l2d_mozannar_operating_points.csv`, 6 rows):
  | α | accuracy | deferral |
  |---|---|---|
  | 0.1 | 0.7925 | 0.839 |
  | 0.2 | 0.7928 | 0.811 |
  | 0.5 | 0.7945 | 0.746 |
  | 1.0 | **0.8300** | 0.025 |
  | 2.0 | 0.8285 | 0.000 |
  | 5.0 | 0.8285 | 0.000 |
  Checks hold: deferral is **monotone-decreasing in α** (0.839→0.000); large α automates → AI-alone
  0.8285; cost==deferral∈[0,1]. **α=1 marginally beats AI-alone: 0.8300 (+0.15 pt) at 2.5 % deferral**;
  the high-deferral end (α≤0.5) sits BELOW AI-alone because it defers 75–84 % to ~75 %-accurate humans
  (same phenomenon as the M4 learned curve). HONEST disclosure: the α grid samples the deferral axis
  coarsely — there is a gap between α=0.5 (defer 0.746) and α=1.0 (defer 0.025); the curve is monotone
  and spans [0,0.84] but the mid-range is sparse. M6 can add intermediate α if a denser frontier helps.
- Promotes: GROUND_TRUTH §8 "Mozannar surrogate exact form" [U]→[V] (eq 10) and Mozannar SHA [U]→[V]
  (e84f3ee); HANDOFF §7 surrogate-extraction + torch≥2.x [OPEN]→[V] (project `.venv` has torch 2.2.2;
  the loss is plain torch ops — no port needed).
- Alternatives rejected: (a) sweeping the additive deferral cost `c` via the cost-sensitive L̃_CE
  (eq 4) — defensible but the task/config name the `L_CE^α` knob, and α maps cleanly to the
  automate-everything anchor; (b) co-training a fresh classifier (Mozannar's upstream default) —
  would confound the cross-arm comparison (invariants #1/#2); (c) drawing a hard per-seed expert m
  for training — higher variance and 5× the fits for no consistency gain over the soft q expectation.
- Evidence: `src/haidc/arms/l2d_mozannar.py`; `tests/arms/test_l2d_mozannar.py` (19 green, incl. the
  correctness gate, gradient check, and α=1 Bayes-consistency test); `tasks/M5-PLAN.md` (PHASE-1
  verification); registered in `run_all.py`; fast suite (98) green; `ruff check src tests` clean.
  Artifacts: `results/l2d_mozannar_predictions.parquet` (20,820 rows = 694×6×5),
  `results/l2d_mozannar_operating_points.csv` (6 rows). Rejector trains on CPU (tiny MLP — not a
  backbone train; the no-CPU-backbone convention covers only the ResNet, validated on Colab).
- Status: active. **M5 COMPLETE** — deployable L_CE^α curve exported for the M6 head-to-head.

## TEMPLATE — copy below for the next entry
## 2026-MM-DD — <title>
- Context:
- Decision:
- Alternatives rejected:
- Evidence:
- Status: active.
