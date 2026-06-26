# DECISIONS.md

Append-only. Newest at the bottom. One entry per decision/deviation. An agent that deviates from
the plan, pins a dependency, fixes a hyperparameter, or promotes a `[U]` fact to `[V]` records it
here in the same change. Format:

```
## YYYY-MM-DD — <short title>
- Context: why a decision was needed.
- Decision: what was chosen.
- Alternatives rejected: and why.
- Evidence: commit SHA / file / paper section / test that backs it.
- Status: active | superseded by <date>.
```

---

## 2026-06-13 — Repository scaffold and invariants
- Context: project kickoff; agents need a grounded, low-context starting point.
- Decision: adopt the orchestrator/subagent layout, the five cross-arm invariants
  (`GROUND_TRUTH.md §7`), the M1–M7 DAG, and the four CLAUDE.md non-negotiables.
- Alternatives rejected: per-arm bespoke agents (replaced by one reusable `arm-implementer` role to
  cut duplication and context); separate baseline milestones (folded into arms).
- Evidence: this scaffold; brief; papers verified 2026-06-13 (see GROUND_TRUTH verification basis).
- Status: active.

## 2026-06-13 — Brief corrections recorded
- Context: three claims in the original brief did not match the sources.
- Decision: (a) Galaxy Zoo 10k subset + Kaggle source are Okati **fns 8–9**, not fn 7;
  (b) raters are citizen-science **volunteers, not domain experts** — state explicitly in report;
  (c) HCT operating points are a **coarse locus over the AI threshold with expected cost ∈ [1,2]**,
  not literally two points.
- Alternatives rejected: propagating the brief verbatim (would put errors in the report).
- Evidence: Okati arXiv:2103.08902v4 p.9 fns 8–9; HCT-1 arXiv:2602.02375v1 abstract + p.2.
- Status: active.

## TEMPLATE — copy below for the next entry
## 2026-MM-DD — <title>
- Context:
- Decision:
- Alternatives rejected:
- Evidence:
- Status: active.
