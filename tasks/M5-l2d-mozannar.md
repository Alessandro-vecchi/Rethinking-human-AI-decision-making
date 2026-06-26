# M5 — L2D-Mozannar arm

Owner: `arm-implementer` (mode=mozannar). Predecessors: M1, M2. Highest porting effort.

## Objective
Port the consistent surrogate `L_CE^α` + rejector from the vendored Mozannar notebooks into the
package, verify it against the paper, train on the frozen split, sweep cost → smooth curve, export
the per-instance table.

## Inputs
`GROUND_TRUTH.md §6` (surrogate, repo, fresh split rationale), `§7`. Vendored
`third_party/learn-to-defer` (pinned SHA). M1 split + label table. M2 features. arXiv:2006.01862 §3.

## Steps
1. Extract `L_CE^α` + rejector training from the notebooks into `src/haidc/arms/l2d_mozannar.py`.
2. **Verify the loss against arXiv:2006.01862 §3** (gradient-check / known-input) BEFORE trusting any
   run. This is the correctness gate for this arm.
3. Train on the frozen 70/15/15 split (the shared split overrides any split inside the notebooks).
4. Sweep the deferral cost parameter (config-driven) → smooth curve.
5. Export the per-instance table (instance_id, true_label, decision, human_queries_used, cost_param).

## Outputs
Ported module, loss-verification test, per-instance table across the cost sweep.

## Tests first
- Surrogate loss matches the paper's form on known inputs; gradient is finite/correct.
- Uses the frozen split (hash check); single-batch overfit sanity.

## Acceptance (gate)
Loss verified vs paper; trains deterministically; smooth curve produced; invariants held (or
divergence logged); table exported; tests pass.
