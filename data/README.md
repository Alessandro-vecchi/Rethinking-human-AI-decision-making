# Data — Galaxy Zoo binary subset (M1)

This directory is **git-ignored** except `README.md` and `split_manifest.json`. Raw inputs, the
crosswalk, and the built label table are not committed; place the inputs locally (provenance
below) and run `make data` to rebuild. The committed `split_manifest.json` fully pins the frozen
split (CLAUDE.md invariant #2); `make verify-split` recomputes it from the inputs and asserts the
hash.

## Pipeline

```
make data         -> builds data/label_table.parquet (git-ignored) + data/split_manifest.json (committed)
make verify-split -> recomputes the split from seed=0 and asserts the committed content_hash
```
Code: `src/haidc/data/{labels,split,sampler,build_split,verify_split}.py`. Config: `configs/data.yaml`.

## Inputs (place locally; not committed)

| File | Provenance | Role |
|---|---|---|
| `galaxyid_to_dr7objid.csv` | Content-based crosswalk (this project; `match_galaxyid_to_dr7objid.py`) | Kaggle `GalaxyID` → GZ2 `dr7objid` |
| `raw/zoo2MainSpecz.csv.gz` | Willett et al. 2013 (spectroscopic-z main sample) | raw Task-01 vote **counts** (the urn) |
| `raw/zoo2MainPhotoz.csv.gz` | Willett et al. 2013 (photometric-z main sample) | raw Task-01 vote **counts** (the urn) |
| `raw/training_solutions_rev1.csv` | Kaggle `galaxy-zoo-the-galaxy-challenge` (Okati **fn 9**) | aggregate vote **fractions** → `y_debiased` |

The 10k binary subset is Okati 2021 **fns 8–9** (`|D|=10,000`, He et al. 2015 ResNet features).
Per-image rater counts do **not** exist in Okati's repo (Data risk #1, resolved — HANDOFF §3c);
they are recovered from Willett via the crosswalk.

### Crosswalk coverage caveat (HANDOFF §5d–§5f)

The crosswalk matches Kaggle vote signatures to Willett *debiased* fractions and keeps **mutual
nearest neighbours within L2 distance 0.30** (`accepted==True`). Of the 10k subset:
- `mutual==True` = 5,904 (the HANDOFF §5c headline); `accepted==True` = **4,626** (dist-capped,
  median per-dim diff ≈ 0.0001). We use `accepted` — the 1,278 mutual-only pairs (median dist
  0.368, ≤0.703) are poor matches that would corrupt the urns (DECISIONS.md 2026-06-26).
- Disclose: "real per-rater labels are recovered for ~59% of the 10k via a debiased-vote-fraction
  crosswalk to Willett 2013; the unmatched remainder is largely a GZ2 subsample (likely Stripe 82)
  absent from the main catalog tables, not a morphology-selected drop." The matched subset is
  morphologically representative (binary-hardness 0.52 vs overall 0.48).

## `label_table.parquet` schema (one row per surviving image, sorted by GalaxyID)

| column | dtype | meaning |
|---|---|---|
| `GalaxyID` | int64 | Kaggle id |
| `dr7objid` | int64 | GZ2 SDSS DR7 object id (18-digit; int64 enforced — float64 collides) |
| `n_smooth` | int64 | raw Willett `t01_..._a01_smooth_count` |
| `n_features` | int64 | raw Willett `t01_..._a02_features_or_disk_count` |
| `n_artifact` | int64 | raw Willett `t01_..._a03_star_or_artifact_count` |
| `y_count` | int64 | argmax of raw counts (tie → 0=smooth). GROUND_TRUTH §4's literal label |
| `y_debiased` | int64 | argmax of Kaggle (== Willett debiased) fractions. **Approved default for all arms** |
| `source_table` | str | `zoo2MainSpecz` or `zoo2MainPhotoz` |

Labels: **0 = smooth / early-type, 1 = features-disk / spiral.**

Filters (build order): drop Task-01 star/artifact-mode (`n_artifact` ≥ both); require
`n_smooth + n_features ≥ 2` (`min_votes_per_image`). Final **N = 4,621** (5 artifact-mode dropped,
0 below min-votes). Votes/image: min 21, median 42, max 68.

## The two labels, and why both exist

`y_count` (6.1% spiral) and `y_debiased` (26.8% spiral) **disagree on 20.86% (964/4,621)** of
images. This is a known GZ2 artifact: redshift debiasing corrects citizen scientists under-calling
features in faint/distant galaxies. `y_debiased` == Okati's published label (Kaggle Class1.1/1.2
argmax) == the target the ~0.83 AI-alone anchor was measured against == the L2D reproduction
target, so **all arms consume `y_debiased`** (DECISIONS.md 2026-06-26). `y_count` is retained for
auditability and reversibility.

## Invariants this table must uphold (GROUND_TRUTH §7)

- **One P(h|x).** The single human-label model is the **raw Willett counts** `{n_smooth, n_features}`.
  It feeds *all three* human consumers identically: the single-human baseline, HCT's `h1`/`h2`
  draw (`sampler.draw_h1_h2`, two distinct votes without replacement), and the "human" term in both
  L2D losses. No arm builds its own human model.
- **y is crowd consensus, not truth** (HANDOFF §9). Report accuracy as "agreement with crowd
  consensus," never "agreement with truth." With `y = y_debiased` drawn against *raw-count* human
  votes, single-human accuracy ≈ **79%** and no longer equals the majority vote-share by
  construction — this **dissolves the §9 degeneracy**. The real story to disclose: HCT inherits
  uncorrected human bias on resolution-hard galaxies (h1/h2 vote raw, cannot apply the debiasing
  the AI learned), while L2D can learn to override there. State this so a reviewer does not read
  L2D's edge as pure artifact.
- **h1 ⊥ h2 | x** is structurally enforced; h2 is a second draw from the same urn, not a distinct
  second human. The opinion-leader α / ρ analysis (Berger) is **not** supported (needs
  identity-linked rows, not counts). Within-urn draws are slightly *negatively* correlated; the
  positive human–human correlation lives in *between-image* urn variance (HANDOFF §6).

## `split_manifest.json` (committed)

70/15/15 split at `seed=0` over the sorted surviving GalaxyIDs; sizes 3234 / 693 / 694.
Keys: `final_N`, `seed`, `split` ratios, ordered `train`/`val`/`test` GalaxyID lists,
per-split `split_hashes`, top-level `content_hash` (over seed+ratios+N+splits only — volatile
fields excluded), `accepted_crosswalk_count`, `y_count_vs_y_debiased_disagreement`,
`default_label_for_arms`, `generated_at_utc`, `git_sha`.
