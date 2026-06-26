# M1 — Frozen Galaxy Zoo split + multi-rater label table

Owner: `data-engineer`. Predecessors: none. Blocks: M2–M5.

## Objective
A reproducible, hash-frozen 70/15/15 split (seed=0) of the Galaxy Zoo binary 10k subset, plus a
per-image multi-rater label table that supports HCT's `h1`/`h2` draw and both L2D human terms.

## Inputs
- `GROUND_TRUTH.md §4` (dataset facts, Data risk #1), `§6` (split rationale).
- Vendored `third_party/differentiable-learning-under-triage` (pin SHA).

## Steps
1. **Resolve Data risk #1 first.** Inspect the vendored repo: does it ship per-image rater counts
   `(n_early, n_spiral)` or only majority `y`? If only `y`, STOP and file the blocker (locate counts
   via Kaggle `galaxy-zoo-the-galaxy-challenge` / Galaxy Zoo release). Counts ≥ 2 per image required.
2. Build split at seed=0 (70/15/15); write + hash a manifest; commit it; implement `make verify-split`.
3. Build label table: per image `(n_early, n_spiral, y=argmax)`; implement `h1`/`h2` sampler
   (two distinct votes, without replacement; assert counts ≥ 2).
4. Document schema + provenance + Okati fns 8–9 in `data/README.md`; record split hash + SHA in
   `DECISIONS.md`.

## Outputs
Frozen split manifest (committed), label table (in `data/`, git-ignored), `make verify-split`,
updated `data/README.md` + `DECISIONS.md`.

## Tests first (TESTING.md)
- Split recomputed from seed=0 matches committed hash.
- `h1`/`h2`: two distinct draws; correct marginal frequencies over many fixed-seed samples; raises
  when counts < 2.

## Acceptance (gate)
`make verify-split` passes; rater counts confirmed available (GROUND_TRUTH §4 `[U]→[V]`) or blocker
filed; schema documented; tests pass.
