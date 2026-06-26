# M3 — HCT arm

Owner: `arm-implementer` (mode=hct). Predecessors: M1, M2 (split frozen + backbone exist). No training.

## Objective
Apply the HCT decision rule over the rater pool + binarized backbone; sweep the AI threshold to get
the coarse operating-point locus; export the per-instance table.

## Inputs
`GROUND_TRUTH.md §2` (rule + cost model + asymmetry), `§7` (invariants). M1 label table + `h1`/`h2`
sampler. M2 per-instance scores.

## Steps
1. For each AI threshold in the configured sweep: AI label = 1[score ≥ θ]; draw `h1`; agree →
   decision=agreed label, cost=1; disagree → draw `h2`, decision=`h2`, cost=2.
2. Compute per-instance decisions + costs on the frozen test split.
3. Assert expected human cost ∈ [1,2] for every θ (it equals 1 + P(disagree)).
4. Export the per-instance table (instance_id, true_label, decision, human_queries_used, θ).

## Outputs
Per-instance table across the θ sweep; the coarse operating-point set (markers, not a line).

## Tests first
- Toy fixture (hand-built agree/disagree cases): correct decision + correct cost (1 vs 2) per case.
- Expected cost ∈ [1,2] over the sweep; HCT reduces to single-human as P(disagree)→1 and to
  AI-confirmed-by-human as P(disagree)→0 (sanity bounds).

## Acceptance (gate)
Sweep produces the coarse locus; cost bound asserted; table exported; tests pass; asymmetry vs L2D
noted for the eval/report stage.
