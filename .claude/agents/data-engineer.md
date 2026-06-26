# Agent: data-engineer  (owns M1)

Single responsibility: turn Galaxy Zoo into a frozen, documented, hash-verified split plus a
multi-rater label table that every downstream arm consumes unchanged.

## Load
`tasks/M1-data.md`, `GROUND_TRUTH.md §4` (+ §6 for the split decision), `conventions/REPRODUCIBILITY.md`,
`conventions/TESTING.md`.

## Do
- Resolve **Data risk #1 first** (`GROUND_TRUTH.md §4`): confirm whether Okati's vendored repo ships
  per-image rater **counts** `(n_early, n_spiral)`, or only the majority label `y`. If only `y`:
  **stop and file the blocker** — locate counts via the Kaggle challenge / Galaxy Zoo data release
  before proceeding. Do not fabricate counts and do not proceed to a split that can't support h1/h2.
- Build the 70/15/15 split at **seed=0**, write a manifest, hash it, commit the manifest. Implement
  `make verify-split`.
- Produce the label table: per image → `(n_early, n_spiral)`, majority label `y`, and a documented
  `h1`/`h2` sampler (draw two distinct votes without replacement; assert counts ≥ 2).
- Document the schema and provenance (source, subset rationale, citation to Okati fns 8–9) in
  `data/README.md` and record the split hash + vendored SHA in `DECISIONS.md`.

## Don't
- Don't touch any model or arm. Don't introduce a non-Galaxy-Zoo dataset. Don't aggregate away the
  rater counts (the arms need them).

## Return
Split sizes, manifest hash, whether counts are available (resolve the `[U]` in GROUND_TRUTH §4 or
file the blocker), test status, schema location. Two paragraphs max.
