# M7 — Technical report (2–3 pages) + repo polish

Owner: orchestrator + `verifier`. Predecessor: M6.

## Objective
A 2–3 page technical report whose every claim traces to results actually produced, plus a repo that
reproduces those results from a clean checkout.

## Inputs
All of `results/`, `GROUND_TRUTH.md`, `DECISIONS.md`, `CITATIONS.md`.

## Required content (and the caveats that must appear — not optional)
1. **Framing.** The novelty is the HCT-vs-L2D head-to-head (unpublished), not the Pareto plot.
   Cite HCT-1/HCT-2, Okati, Mozannar precisely; cite De Toni 2024 for framing only.
2. **Dataset.** Galaxy Zoo binary 10k subset (Okati fns 8–9). **State plainly that the 30+ raters
   are citizen-science volunteers, not domain experts**, and what that implies for the human ceiling
   and the `h2` tiebreak quality.
3. **Methods.** HCT rule + cost model (expected cost ∈ [1,2]); Okati Thm. 3 / budget `b`; Mozannar
   `L_CE^α`. State the shared-backbone + frozen-split invariants that make the comparison valid.
4. **Results.** The frontier with CIs. **State the HCT (coarse markers) vs L2D (smooth curve)
   asymmetry explicitly**; do not present HCT as a continuous curve.
5. **Honesty.** Report where arms are statistically indistinguishable; report null/negative results;
   distinguish upstream self-claims (e.g. HCT's 28–44% saving on its own datasets) from this study's
   measurements on Galaxy Zoo. Do not import an upstream claim as a finding here.
6. **Limitations.** Single dataset, single run, non-expert raters, any logged invariant deviations.

## Acceptance (gate)
`make all` reproduces every figure/table from a clean checkout; every report claim traces to a
produced result or a checked citation; all required caveats present; `verifier` PASS.
