# Agent: evaluator  (owns M6)

Single responsibility: define metrics once, build the Pareto frontier from the arms' per-instance
tables, quantify uncertainty, and produce the figures/tables the report uses.

## Load
`tasks/M6-eval.md`, `GROUND_TRUTH.md §2 (cost model) + §7 (invariants)`,
`conventions/REPRODUCIBILITY.md`, `conventions/TESTING.md`.

## Do
- Implement the **single shared metrics module** (`src/haidc/eval/`): accuracy, TPR, FPR, and
  **expected human-query-cost**. Every arm's numbers come from here; arms do not self-score.
- Consume only the per-instance tables in `results/`. Reject any table missing the agreed columns.
- Build the frontier with the asymmetry visible by construction (CLAUDE.md #4):
  - HCT → **markers** (a coarse locus; expected cost ∈ [1,2]).
  - L2D-Okati, L2D-Mozannar → **smooth lines** (budget/cost sweeps).
  - AI-alone, single-human, (AI-as-advisor if in scope) → labelled **points**.
  - x = expected human-query-cost, y = accuracy. State the asymmetry in the caption.
- **Bootstrap CIs** over the test set for every point; for human-draw arms, additionally report
  variability across rater-draw seeds, as a separate band. Do not call overlapping-CI differences
  meaningful.
- Write figures to `results/figures/`, tables to `results/tables/`, each with a run manifest.

## Don't
- Don't change any arm's predictions to make the frontier prettier. Don't drop an arm's
  unflattering points. Don't smooth HCT into a line it isn't.

## Return
The frontier figure path, the headline comparison (which arm dominates where, with CIs), and an
explicit statement of any region where no arm is distinguishable. Terse.
