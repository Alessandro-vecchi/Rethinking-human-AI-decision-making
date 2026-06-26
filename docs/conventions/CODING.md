# conventions/CODING.md

Python 3.11+, PyTorch. Research code that someone must re-run in a year — optimize for being
**obvious and reproducible**, not clever or fast.

## Layout
```
src/haidc/
  data/      # split, multi-rater label table, h1/h2 sampling
  arms/      # backbone, hct, l2d_okati (wraps vendored), l2d_mozannar (ports surrogate), run_all
  eval/      # shared metrics, pareto, plotting  ← the ONLY place metrics are defined
tests/       # mirrors src/haidc/
configs/     # one yaml per stage (data, backbone, arms, eval); all knobs live here, not in code
third_party/ # vendored Okati + Mozannar repos, pinned by SHA (DECISIONS.md), read-mostly
```

## Rules
- **No bloat.** Add a module/file only when a named caller needs it. No "utils dumping ground",
  no speculative abstraction, no framework for a one-run study. Delete dead scaffolding you find.
- **Config over constants.** Hyperparameters, paths, seeds, thresholds → `configs/*.yaml`. No
  magic numbers in code. The threshold/budget/cost sweeps are config-driven so the Pareto curves
  are reproducible.
- **Don't re-implement upstream.** L2D-Okati and L2D-Mozannar wrap/port the vendored code
  (`third_party/`). Write thin adapters in `src/haidc/arms/` that feed them the shared backbone
  features + frozen split and read back per-instance predictions. Re-implement only if upstream is
  broken — and then record the breakage in `DECISIONS.md` first.
- **Per-instance prediction tables are the interface between arms and eval.** Each arm writes a
  table (instance_id, true_label, decision, human_queries_used, plus arm params) to `results/`.
  `eval/` consumes only these. Arms never compute their own headline metrics.
- **Determinism by construction** (`REPRODUCIBILITY.md`): `seed_everything` at entry; no hidden
  global state; no reliance on dict ordering for anything numeric.
- **Style:** `ruff` clean. Type hints on public functions. Docstrings state *what + why + source*
  (cite the paper section the function implements). Comments explain non-obvious modeling choices,
  not syntax.
- **Failure is loud.** Assert preconditions (e.g. rater counts ≥ 2 before an h2 draw; split hash
  matches before a run). A confound should crash, not silently produce a wrong number.

## Commits
- Small, single-purpose, with a message that says what changed and why. Reference the milestone
  (`M3:`) and any `DECISIONS.md` entry. Tests in the same commit as the code they cover.
