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

## TEMPLATE — copy below for the next entry
## 2026-MM-DD — <title>
- Context:
- Decision:
- Alternatives rejected:
- Evidence:
- Status: active.
