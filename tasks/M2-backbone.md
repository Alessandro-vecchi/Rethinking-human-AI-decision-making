# M2 — Shared AI backbone + score export

Owner: `arm-implementer` (mode=backbone). Predecessor: M1. Blocks: M3–M5.

## Objective
One AI classifier, reused everywhere, with per-instance scores exported on the frozen test split.
Establishes the AI-alone baseline.

## Inputs
`GROUND_TRUTH.md §3,§4,§7`. Frozen split + label table from M1. Okati's ResNet feature choice (§4).

## Steps
1. Decide: reuse Okati's ResNet (He et al. 2015) features, or train a fresh backbone. Whatever is
   chosen is used by ALL arms (invariant §7). Log the decision + the `[U]` resolution in `DECISIONS.md`.
2. Train/load on the frozen train split. **Overfit one batch first** (sanity).
3. Export per-instance test scores (continuous, pre-threshold) to `results/` — these feed HCT's AI
   judgment and the L2D classifiers.
4. Record the AI-alone accuracy (threshold 0.5) with bootstrap CI.

## Outputs
Backbone artifact, per-instance score table, AI-alone baseline number + manifest.

## Tests first
- Single-batch overfit reaches ~0 train loss.
- Deterministic: two same-seed runs produce identical scores.
- Score table has one row per test instance, scores in [0,1].

## Acceptance (gate)
Sanity passes; scores exported for the full frozen test split; AI-alone baseline recorded with CI;
backbone decision logged; tests pass.
