# M4 — L2D-Okati arm

Owner: `arm-implementer` (mode=okati). Predecessors: M1, M2.

## Objective
Wrap the vendored Okati implementation, reproduce its baseline, sweep the deferral budget `b` to get
a smooth Pareto curve on the frozen split; export the per-instance table.

## Inputs
`GROUND_TRUTH.md §3` (Thm. 3, Alg. 1, budget `b`), `§7`. Vendored
`third_party/differentiable-learning-under-triage` (pinned SHA). M1 split + label table. M2 features.

## Steps
1. **Reproduce Okati's reported Galaxy Zoo behavior with the vendored code before changing anything.**
   Record the number; if it cannot be reproduced, STOP and file the blocker.
2. Write a thin adapter in `src/haidc/arms/l2d_okati.py` feeding the shared features + frozen split;
   read back per-instance model/defer decisions and the human term.
3. Sweep `b` (config-driven) → smooth curve.
4. Export the per-instance table (instance_id, true_label, decision, human_queries_used, b).
   In L2D, human_queries_used = 1 iff the policy defers that instance, else 0.

## Outputs
Adapter, per-instance table across the `b` sweep, reproduced-baseline note in `DECISIONS.md`.

## Tests first
- Adapter feeds the frozen split (hash check) and shared features; rejects mismatched inputs.
- Deferred-fraction increases monotonically (within noise) with `b`; cost accounting correct.

## Acceptance (gate)
Upstream baseline reproduced; smooth curve produced; invariants held (or divergence logged); table
exported; tests pass. Re-implementation only if upstream broken, logged first.
