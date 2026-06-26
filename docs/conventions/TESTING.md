# conventions/TESTING.md

Tests are written **before** the implementation they cover. An arm is not "done" until its tests
existed first and pass. This is enforced at milestone gates (`ROADMAP.md`).

## Order of work for any unit
1. Write the test from the acceptance criteria in the `tasks/M*.md` ticket. It must fail (red).
2. Write the minimum code to pass it (green). 3. Refactor without breaking it.

## What actually needs a test here (research code — test behavior, not lines)
- **Decision rules and cost accounting.** HCT: on a hand-built toy of agree/disagree cases, assert
  the chosen label and the per-instance cost (1 vs 2) are correct, and that expected cost ∈ [1,2].
  This is the cheapest place to catch the most damaging bugs.
- **Metric definitions.** Accuracy, TPR, FPR, expected-human-cost on tiny fixtures with
  hand-computed answers. The shared `eval` module is the single source; every arm uses it.
- **The frozen split is frozen.** A test recomputes the split from `seed=0` config and asserts it
  matches the committed manifest hash. Guards CLAUDE.md invariant #2.
- **Label sampling.** `h1`/`h2` draw without replacement from per-image counts: assert two distinct
  draws, correct marginal frequencies over many samples (statistical test with a fixed seed),
  and a clear error when counts < 2.
- **Surrogate-loss sanity (Mozannar).** Gradient-check / known-input check against the form in
  arXiv:2006.01862 §3 before trusting any training run.
- **Reproduction guard.** A smoke test that a 1-epoch / 1-batch run is deterministic across two
  invocations with the same seed (`REPRODUCIBILITY.md`).

## What not to test
- Vendored upstream internals (test our *wrappers* and the numbers they produce, not Okati's code).
- Plot pixel output (assert the underlying table instead).
- Things that only restate the framework.

## Sanity checks that are not unit tests but are mandatory before a full run
- **Overfit one batch** to ~0 loss (model can learn at all).
- **Reproduce the upstream baseline number** before changing vendored behavior; record it.

Run: `make test` (with coverage). A failing test blocks the milestone. Do not `xfail` to pass a gate.
