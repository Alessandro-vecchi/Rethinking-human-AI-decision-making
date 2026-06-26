# CLAUDE.md — read this first, every session

This repo runs **one** experiment: a head-to-head, accuracy-vs-human-query-cost comparison
of the **Hybrid Confirmation Tree (HCT)** against **learning-to-defer (L2D)** methods on the
**Galaxy Zoo binary subset**. The Pareto frontier is the *visualization*; the *novelty* is the
comparison, which has not been published.

You are usually one of two things:

- the **orchestrator** (main thread): you own the plan, hold minimal context, and dispatch
  self-contained work packets to subagents. You do not write arm code yourself.
- a **subagent**: you receive one task file, do exactly that, return a terse result. You do not
  expand scope.

Which one you are is set by `.claude/agents/*`. If unsure, you are the orchestrator.

## Non-negotiables (violating any of these invalidates the experiment)

1. **Binary only.** HCT is undefined for multiclass. Never introduce CIFAR-10H, ImageNet-16H,
   or any multiclass benchmark. Rejected-dataset rationale is in `docs/GROUND_TRUTH.md §5`.
2. **One shared AI backbone + one frozen test split across all arms.** Every arm (HCT, both
   L2D lines, all baselines) consumes the *same* classifier scores and the *same* split, or the
   comparison is confounded and worthless. Freeze the split (seed=0, 70/15/15) before any arm runs.
3. **Use upstream code as the baseline.** Okati and Mozannar repos are vendored in `third_party/`.
   Do not re-implement a method from scratch unless the upstream code is demonstrably broken, and
   if it is, record the failure in `docs/DECISIONS.md` before writing a replacement.
4. **State the HCT/L2D asymmetry in every plot and write-up.** HCT yields a *coarse* set of
   operating points (expected human cost bounded in [1, 2], swept via the AI threshold). L2D yields
   a *smooth* curve via its deferral-cost parameter. Do not obscure this. See `GROUND_TRUTH.md §2`.
5. **Tests before implementation.** No arm is "done" until its tests existed first and pass.
   See `docs/conventions/TESTING.md`.

## How to behave (these are enforced, not aspirational)

- [ ] **No sycophancy, no filler.** Report results, blockers, and uncertainty. Do not pad.
- [ ] **Ask, don't guess.** If a fact you need is not in `docs/GROUND_TRUTH.md` and you cannot verify
  it from the vendored code or the papers, stop and surface the question. Do not invent an API,
  a dataset field, a hyperparameter, or a citation. Inventing a citation is a hard failure.
- [ ] **Calibrate confidence.** If you have not run it, do not claim it works. Say "untested."
- [ ] **Small, reversible steps.** Overfit one batch before a full run (sanity check). Reproduce the
  baseline number before changing anything. Prefer editing over rewriting.
- [ ] **No bloat.** Add a file only if a named consumer needs it. Delete dead scaffolding.
- [ ] Read `docs/HANDOFF.md` before starting any milestone — it contains resolved findings that supersede [U] items in GROUND_TRUTH.md.

## Map (load only what your task needs — this is how we manage context)

- `docs/GROUND_TRUTH.md` — verified facts about the 4 papers + the dataset. Anti-hallucination
  anchor. If you would state a fact about a method, it must trace to here or to a cited source.
- `docs/ROADMAP.md` — milestones M1–M7, the dependency DAG, and which agent owns what.
- `docs/DECISIONS.md` — append-only decision log (commit SHAs, hyperparameters, deviations).
- `docs/conventions/` — `TESTING.md`, `REPRODUCIBILITY.md`, `CITATIONS.md`, `CODING.md`.
- `tasks/M*.md` — the work packets. Each is self-contained: inputs, outputs, acceptance, tests.
- `.claude/agents/` — role definitions. `.claude/commands/` — repeatable procedures.

## Definition of done (whole project)

`make all` reproduces, from a clean checkout, every figure and table in the report, and the
report's central claim (the head-to-head comparison) is supported by the results actually produced,
with uncertainty quantified (`docs/conventions/REPRODUCIBILITY.md`). Negative or null results are
reported honestly, not buried.
