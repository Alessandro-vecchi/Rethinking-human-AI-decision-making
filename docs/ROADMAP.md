# ROADMAP.md

The map. Tickets live in `tasks/M*.md`; facts live in `docs/GROUND_TRUTH.md`. This file is the
dependency graph and the gate criteria. The orchestrator works this top-to-bottom.

## Dependency DAG (what may run in parallel, and what may not)

```
M1 data ──► M2 backbone ──┬─► M3 HCT          ┐
                          ├─► M4 L2D-Okati     ├─► M6 eval+Pareto ──► M7 report
                          └─► M5 L2D-Mozannar  ┘
```

- **M1 → M2 → {M3,M4,M5} are the only safe parallel fan-out.** Do **not** start any arm (M3–M5)
  until M2 has produced the shared backbone **and** M1's split is frozen and hash-verified
  (`make verify-split`). Parallel arms before the split is frozen create confounds (CLAUDE.md #2).
- M3, M4, M5 are independent of each other → may be dispatched to three parallel `arm-implementer`
  subagents *once the gate above is green*.
- M6 consumes all three arms' per-instance prediction tables; it does not start until all are done.

## Milestones, owners, and gates

| M | Title | Owner agent | Done when (gate) |
|---|---|---|---|
| **M1** | Frozen Galaxy Zoo split + multi-rater label table | `data-engineer` | Split manifest hashed & committed; per-image rater counts present (resolves GROUND_TRUTH §4 risk #1) **or** blocker filed; schema documented; `make verify-split` passes; tests pass. |
| **M2** | Shared AI backbone + score export | `arm-implementer` (backbone mode) | One classifier; per-instance scores exported for the frozen test split; single-batch overfit sanity check passes; AI-alone baseline number recorded; tests pass. |
| **M3** | HCT arm | `arm-implementer` | HCT rule over `h1`/`h2` + binarized backbone; AI-threshold sweep → coarse operating points with expected cost ∈ [1,2]; per-instance table exported; tests pass. |
| **M4** | L2D-Okati arm | `arm-implementer` | Vendored Okati code wrapped; deferral-budget `b` sweep → smooth curve; baseline number reproduced before any change; per-instance table exported; tests pass. |
| **M5** | L2D-Mozannar arm | `arm-implementer` | `L_CE^α` + rejector extracted from vendored notebooks, verified vs arXiv:2006.01862; cost sweep → smooth curve; per-instance table exported; tests pass. |
| **M6** | Metrics + Pareto frontier + plots | `evaluator` | One shared metrics module; Pareto frontier with HCT as markers + L2D as lines + baselines as points; bootstrap CIs; asymmetry stated in caption; figures/tables in `results/`. |
| **M7** | 2–3 page technical report + repo polish | orchestrator + `verifier` | Report claims trace to produced results; uncertainty quantified; non-expert raters + HCT/L2D asymmetry + null results stated honestly; `make all` reproduces everything from clean checkout. |

Every arm milestone (M2–M5) must also satisfy the cross-arm invariants in `GROUND_TRUTH.md §7`.

## Baselines (folded into the arms, not separate milestones)
- **AI alone** — M2 byproduct (cost 0 humans).
- **Single human alone** — one draw from the rater pool (cost 1 human); computed in M6 from the
  label table.
- **AI-as-advisor** — only if the vendored L2D code supports an advisor mode cleanly (HCT-2's
  comparator). If it requires non-trivial new modeling, it is **out of scope** for the one-run
  budget; record the decision in `DECISIONS.md`. Do not build a human-subjects advisor study.

## Standing risks (carried until closed in DECISIONS.md)
1. Okati repo may not ship per-image rater counts → blocks HCT h1/h2 (GROUND_TRUTH §4). **Highest.**
2. Mozannar surrogate lives in notebooks → porting effort + correctness risk (GROUND_TRUTH §6).
3. Arm-specific co-training may diverge from the shared backbone → confound to disclose (§7).
4. HCT produces few operating points → frontier looks sparse; this is real, not a bug. State it.

## Context-management protocol (how the orchestrator stays lean)
- Dispatch one milestone per subagent invocation with a single `tasks/M*.md` ticket + pointers to
  the *minimum* docs that ticket names. Do not paste GROUND_TRUTH wholesale into a subagent.
- Subagents return: files changed, test status, the one or two numbers that matter, and any new
  `[U]` → `[V]` promotions or new blockers. Nothing else.
- After each milestone, the orchestrator updates `DECISIONS.md` and re-reads only the gate criteria
  for the next milestone — not the whole repo.
