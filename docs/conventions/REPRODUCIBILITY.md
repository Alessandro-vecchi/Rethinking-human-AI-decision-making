# conventions/REPRODUCIBILITY.md

The whole report must regenerate from a clean checkout via `make all`. If a number cannot be
reproduced, it is not a result.

## Seeds & determinism
- Global seed = **0** unless a config overrides it. Seed Python `random`, `numpy`, and `torch`
  (CPU + CUDA) in one `seed_everything(seed)` helper in `src/haidc/`; call it at every entry point.
- Set `torch.use_deterministic_algorithms(True)` and `cudnn.deterministic = True`. Document any op
  that cannot be made deterministic in `DECISIONS.md` rather than silently leaving it nondeterministic.
- The Galaxy Zoo split (70/15/15, seed=0) is generated once, hashed, and committed as a manifest
  (`make verify-split` re-checks the hash). It is the same split for every arm.

## Environment capture
- Pin exact versions in `requirements.txt` after the first clean build; record the resolved
  lockfile hash and the Python/CUDA versions in `DECISIONS.md`.
- Record the pinned commit SHA of each vendored repo in `third_party/` (`DECISIONS.md`).

## Run manifests
- Every run writes a small JSON manifest to `results/` capturing: config used, seed, git SHA of
  this repo, vendored SHAs, and the produced metric values. Figures/tables reference their manifest.

## Uncertainty (a result without it is incomplete)
- Report **bootstrap confidence intervals** over the test set for every accuracy/cost point.
- For arms involving a random human draw (single-human baseline, HCT `h1`/`h2`), also resample the
  rater draw across seeds and report variability from that source separately — it is a real and
  distinct source of noise from test-set sampling. State both.
- Do not report a difference between arms as meaningful if the CIs overlap; say so plainly.

## Preregistration discipline (the comparison is the novelty — protect it)
- Fix metrics, split, and the set of arms **before** looking at comparative results. They are
  fixed by `GROUND_TRUTH.md §7` + this scaffold. Changing them after seeing results is p-hacking;
  if a change is genuinely necessary, log it in `DECISIONS.md` with the before/after rationale.
