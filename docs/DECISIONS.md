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

## TEMPLATE — copy below for the next entry
## 2026-MM-DD — <title>
- Context:
- Decision:
- Alternatives rejected:
- Evidence:
- Status: active.
