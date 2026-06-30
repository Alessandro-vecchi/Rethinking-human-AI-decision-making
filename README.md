# HCT vs Learning-to-Defer on Galaxy Zoo

A reproducible head-to-head comparison of human–AI decision-making paradigms on a single binary
benchmark with real multi-rater human labels. The contribution is the **comparison itself**
(unpublished); the Pareto frontier is the visualization.

**Research question.** On the Galaxy Zoo binary subset (spiral vs. early-type), how do these arms
compare on the accuracy-vs-human-query-cost Pareto frontier?

| Arm | Source | Mechanism | Curve shape |
|---|---|---|---|
| Hybrid Confirmation Tree (HCT) | Berger et al. 2026 (arXiv:2602.02375, 2603.29866) | independent human+AI; 2nd human breaks ties | coarse markers, cost ∈ [1,2] |
| L2D — Differentiable Triage | Okati, De & Gómez-Rodríguez, NeurIPS 2021 (2103.08902) | learned threshold deferral (budget `b`) | smooth |
| L2D — Consistent Surrogate | Mozannar & Sontag, ICML 2020 (2006.01862) | consistent surrogate `L_CE^α` + rejector | smooth |
| Baselines | — | AI alone, single human, AI-as-advisor (if in scope) | points |

**Dataset.** Galaxy Zoo binary 10k subset from Okati 2021 (fns 8–9). 30+ labels/image from
**citizen-science volunteers, not domain experts** — stated explicitly in the report.

**Non-negotiables.** Binary only; one shared AI backbone + one frozen split across all
arms; use upstream code (vendored in `third_party/`) as the baseline; state the HCT/L2D curve-shape
asymmetry everywhere; tests before implementation.

## Reproduce
```bash
make env           # Python 3.11+, pinned deps
make test          # unit tests (must pass before any arm is "done")
make all           # data → verify-split → backbone → arms → eval → report
```
`make all` regenerates every figure/table from a clean checkout, with bootstrap CIs.

## Layout
```
src/haidc/               data / arms / eval
third_party/             vendored Okati + Mozannar repos (pinned by SHA)
```
