# M6 — Metrics + Pareto frontier + plots

Owner: `evaluator`. Predecessors: M3, M4, M5 (all per-instance tables exist).

## Objective
Define metrics once, assemble the accuracy-vs-human-query-cost frontier with the HCT/L2D asymmetry
visible by construction, quantify uncertainty, emit the report's figures/tables.

## Inputs
`GROUND_TRUTH.md §2 (cost), §7 (invariants)`, `REPRODUCIBILITY.md`. Per-instance tables in `results/`
from M2 (AI-alone), M3 (HCT), M4 (Okati), M5 (Mozannar); M1 label table (single-human baseline).

## Steps
1. Shared metrics module (`src/haidc/eval/`): accuracy, TPR, FPR, expected human-query-cost.
   Every arm scored here; reject tables missing agreed columns.
2. Single-human baseline: one rater draw (cost 1), with rater-draw-seed variability.
3. Frontier: x=expected human-query-cost, y=accuracy. HCT=markers (coarse locus, cost∈[1,2]);
   Okati & Mozannar = smooth lines; AI-alone / single-human / (advisor if in scope) = labelled points.
   Caption states the asymmetry explicitly (CLAUDE.md #4).
4. Bootstrap CIs over the test set for every point; separate band for rater-draw variability on
   human-draw arms. Do not call overlapping-CI gaps meaningful.
5. Write figures → `results/figures/`, tables → `results/tables/`, each with a manifest.

## Outputs
Frontier figure, summary table, per-arm metric tables, manifests.

## Tests first
- Metric functions on tiny fixtures vs hand-computed values.
- Cost accounting: AI-alone=0, single-human=1, HCT∈[1,2], L2D∈[0,1] human queries per instance.

## Acceptance (gate)
Frontier built with asymmetry shown; CIs present; no arm's points dropped or smoothed; regions of
no-significant-difference stated; tests pass.
