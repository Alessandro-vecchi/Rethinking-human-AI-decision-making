# M2 — Shared AI backbone + score export — PLAN (not yet implemented)

Owner: `arm-implementer` (mode=backbone). Predecessor: M1 (done). Blocks: M3–M5.
This is the plan only. **Do not vendor, train, or touch any model until the backbone decision
below is approved by the orchestrator** (GROUND_TRUTH §8 open item).

## 0. Inputs M2 consumes from M1 — UNCHANGED

- `data/split_manifest.json` (committed): the frozen 70/15/15 GalaxyID lists (train 3234 / val 693
  / test 694, N=4,621). M2 reads these lists directly; it does **not** recompute or reshuffle.
  Run `make verify-split` first as a guard.
- `data/label_table.parquet`: per-image `y_debiased` (the **approved default target**, DECISIONS.md
  2026-06-26), plus the urn counts (not used by the backbone, used later by HCT). Backbone trains
  and is evaluated against `y_debiased` — this is what makes the ~0.83 anchor and the L2D
  reproduction coherent.
- One shared backbone, one frozen test split → every arm (AI-alone, HCT's AI judgment, both L2D
  classifiers) consumes the *same* scores (GROUND_TRUTH §7 invariants; CLAUDE.md #2).

## 1. THE decision to approve first: reuse Okati features vs train fresh

Default recommendation: **train fresh, but gate it behind a ~15-min probe of the vendored Okati
repo.** Rationale and the exact branch:

**Step 1 (cheap, do first): vendor + probe.** Clone `github.com/Networks-Learning/
differentiable-learning-under-triage` into `third_party/okati2021/` (currently only
`third_party/README.md` exists), pin its commit SHA in DECISIONS.md. Probe: does it ship a
**precomputed, GalaxyID-aligned 2048-d embeddings artifact** (e.g. `galaxy_data.pkl` with
`(features, Y, h_pred)`, HANDOFF §3b), or only `prepare_data.py` (code that needs the images)?

- **If embeddings ship AND align to GalaxyID → REUSE them.** Biggest time saver in the project:
  no ~3 GB Kaggle image download, no GPU training. The two usual worries are already retired:
  - *Coverage*: our 4,621 ⊂ Okati's 10k by construction, so every frozen-split galaxy has features.
  - *ID alignment*: Okati's 10k is `sorted(GalaxyID)[:10000]` (HANDOFF §3b), so row→GalaxyID is
    recoverable; join features to the frozen split on GalaxyID and assert 100% coverage.
  Verify the reused features reproduce ≈0.83 AI-alone on our `y_debiased` test split before trusting.
- **If only `prepare_data.py` ships → "reuse" collapses into "train"** (features require running
  images through a ResNet), so **train fresh**.

**Train-fresh recipe (Okati-faithful, HANDOFF §3a):** `torchvision.models.resnet50()` **scratch
init, no ImageNet weights**; head `nn.Sequential(nn.Linear(2048,2), nn.LogSoftmax(dim=-1))`;
`NLLLoss`; input 3×224×224 with ImageNet mean/std normalization. Requires the ~3 GB Kaggle
`galaxy-zoo-the-galaxy-challenge` image zip (Okati fn 9). Train on the frozen **train** split only.

Either path requires Okati vendored (features OR the training recipe + baseline reproduction).
**Flag for orchestrator approval** — this is GROUND_TRUTH §8's open `[U]` and must not be chosen
silently. Log the resolution + pinned SHA in DECISIONS.md.

Env note for whoever implements: torch is pinned **2.2.2** with **numpy 1.26.4** (the macOS x86_64
constraint, DECISIONS.md 2026-06-26). `seed_everything` already seeds torch CPU+CUDA and sets
deterministic algorithms. If training moves to a CUDA/Linux box, re-pin torch and re-record the
lockfile hash.

## 2. Outputs

- Backbone artifact (`*.pt`, git-ignored) + a run manifest in `results/` (config, seed, repo SHA,
  vendored SHA, metrics — REPRODUCIBILITY.md).
- Per-instance **test** score table → `results/backbone_scores.parquet` (config
  `export_scores_path`): one row per frozen **test** GalaxyID, continuous pre-threshold
  `P(spiral|x) ∈ [0,1]`. This is the single score source for AI-alone, HCT's AI judgment, and both
  L2D classifiers.
- AI-alone baseline accuracy at threshold 0.5 with **bootstrap CI** (REPRODUCIBILITY.md).

## 3. Sanity gates (mandatory before any full run; M2 ticket + TESTING.md)

1. **Single-batch overfit → ~0 train loss** (the model can learn at all).
2. **AI-alone test accuracy ≈ 0.83** (Okati Fig. 4(b), Galaxy Zoo, b=0 full automation → P(y≠ŷ)
   ≈ 0.17). Coherent now that the target is `y_debiased`. If reusing features, this doubles as the
   reproduction check; if training fresh, reproduce the upstream number before changing anything.
3. **Determinism**: two same-seed runs produce identical exported scores.

## 4. Tests-first list (write before implementation; TESTING.md)

- Score table has exactly one row per frozen **test** GalaxyID (= 694), no extras/dupes; every
  score ∈ [0,1]; ordering/coverage matches `split_manifest.json["test"]`.
- Determinism: two same-seed score exports are byte-/value-identical (reproduction guard).
- Single-batch overfit reaches ~0 loss (small fixture / few steps).
- Feature-reuse join (if that branch is taken): 100% of frozen-split GalaxyIDs map to a feature
  row; fail loud on any miss (mirrors the M1 join assertion).
- Wrapper-only: test our loader/exporter and the numbers, **not** Okati's internals (TESTING.md).

## 5. Out of scope for M2

No HCT rule, no L2D losses, no Pareto/eval. Those are M3–M6 and consume
`results/backbone_scores.parquet` + the frozen split unchanged.
