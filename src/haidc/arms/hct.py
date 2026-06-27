"""`make arms` (HCT portion) — the Hybrid Confirmation Tree arm (M3).

HCT is a DECISION RULE, not a trainable model (GROUND_TRUTH §2, HANDOFF §8): there is no model to
train and no loss. We apply the rule per TEST instance over the M1 vote urns and the binarized M2
backbone score, sweep the AI threshold theta (the only HCT knob), and export the durable
per-instance table M6 consumes.

Rule, per instance, per theta (GROUND_TRUTH §2):
  ai_label = 1[score >= theta]; draw (h1, h2) from the urn (M1 sampler, without replacement).
  agree (ai_label == h1) -> decision = that label, human_queries = 1.
  disagree                -> decision = h2,         human_queries = 2.
Expected human cost = 1 + P(ai != h1), bounded in [1, 2] for every theta. A human always approves
the final decision (true by construction).

Design (tasks/M3-PLAN.md):
  - Humans are drawn ONCE per (instance, seed) and reused across all theta — moving the AI
    threshold changes only `ai_label`, never the humans. This isolates the threshold effect.
  - Determinism: one np.random.default_rng(seed) per rater seed, drawing over the TEST ids in the
    frozen split-manifest order (so the draw sequence is reproducible).
  - Multi-seed band over eval.yaml rater_draw_seeds (HCT is a stochastic arm); the band is the
    min/max across seeds. Bootstrap CIs are M6's job, not this arm's.

Coarse-vs-smooth asymmetry (CLAUDE.md #4): the theta sweep yields a COARSE set of (accuracy,
expected-cost) markers with cost in [1, 2] — qualitatively unlike L2D's smooth deferral-cost curve.
M6 plots these as markers, not a line, and states the asymmetry in the caption.

Label convention: 0 = smooth/early-type, 1 = features/disk = spiral (matches y_debiased and the
backbone's P(spiral|x)).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from haidc.data.sampler import draw_h1_h2
from haidc.eval.metrics import accuracy, expected_human_cost
from haidc.seed import seed_everything

PRED_COLUMNS = [
    "GalaxyID", "y_debiased", "ai_label", "h1", "h2",
    "decision", "human_queries", "theta", "seed",
]


def apply_hct(ai_label, h1, h2):
    """The HCT rule. Scalars or numpy arrays; returns (decision, human_queries).

    agree (ai_label == h1) -> (ai_label, 1); disagree -> (h2, 2). Pure and vectorized.
    """
    ai = np.asarray(ai_label)
    a = np.asarray(h1)
    b = np.asarray(h2)
    agree = ai == a
    decision = np.where(agree, ai, b)
    queries = np.where(agree, 1, 2)
    if decision.ndim == 0:  # preserve scalar-in/scalar-out for the toy tests
        return int(decision), int(queries)
    return decision.astype(np.int64), queries.astype(np.int64)


def _draw_per_seed(label_df: pd.DataFrame, test_ids, seed: int):
    """Draw (h1, h2) once per id for one rater seed, iterating ids in the given (frozen) order.

    Returns {GalaxyID: (h1, h2)}. One rng per seed over a fixed id order => reproducible.
    """
    urn = label_df.set_index("GalaxyID")
    rng = np.random.default_rng(seed)
    draws = {}
    for gid in test_ids:
        row = urn.loc[gid]
        draws[gid] = draw_h1_h2(int(row["n_smooth"]), int(row["n_features"]), rng)
    return draws


def run_sweep(scores_df, label_df, test_ids, thresholds, seeds) -> pd.DataFrame:
    """Apply the HCT rule over the theta sweep x rater seeds; return the per-instance table.

    Columns = PRED_COLUMNS. Humans drawn once per (id, seed); only ai_label varies with theta.
    """
    test_ids = [int(g) for g in test_ids]
    score = scores_df.set_index("GalaxyID")["score"]
    y = label_df.set_index("GalaxyID")["y_debiased"]

    rows = []
    for seed in seeds:
        draws = _draw_per_seed(label_df, test_ids, int(seed))
        for theta in thresholds:
            for gid in test_ids:
                h1, h2 = draws[gid]
                ai_label = int(score.loc[gid] >= theta)
                decision, queries = apply_hct(ai_label, h1, h2)
                rows.append((
                    gid, int(y.loc[gid]), ai_label, h1, h2,
                    decision, queries, float(theta), int(seed),
                ))
    return pd.DataFrame(rows, columns=PRED_COLUMNS)


def operating_points(pred: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-instance table to the coarse (accuracy, cost) marker locus.

    Per theta: mean over seeds + min/max band. accuracy/cost are computed per (theta, seed) via the
    SHARED eval metrics, then aggregated across seeds.
    """
    recs = []
    for theta, g in pred.groupby("theta"):
        accs, costs = [], []
        for _seed, gs in g.groupby("seed"):
            accs.append(accuracy(gs["decision"].to_numpy(), gs["y_debiased"].to_numpy()))
            costs.append(expected_human_cost(gs["human_queries"].to_numpy()))
        accs, costs = np.array(accs), np.array(costs)
        recs.append({
            "theta": float(theta),
            "accuracy_mean": accs.mean(), "accuracy_lo": accs.min(), "accuracy_hi": accs.max(),
            "cost_mean": costs.mean(), "cost_lo": costs.min(), "cost_hi": costs.max(),
            "n_seeds": len(accs),
        })
    return pd.DataFrame(recs).sort_values("theta").reset_index(drop=True)


def _assert_binary(pred: pd.DataFrame) -> None:
    """Label-encoding guard (highest-yield bug to prevent): everything is in {0,1}."""
    for col in ["ai_label", "h1", "h2", "decision", "y_debiased"]:
        bad = set(pd.unique(pred[col])) - {0, 1}
        if bad:
            raise AssertionError(f"{col} has non-binary values {bad}; expected 0/1 (1=spiral)")


def _assert_cost_bound(ops: pd.DataFrame) -> None:
    """Expected human cost must lie in [1, 2] for every theta (GROUND_TRUTH §2)."""
    for _i, r in ops.iterrows():
        if not (1.0 <= r["cost_lo"] and r["cost_hi"] <= 2.0):
            raise AssertionError(
                f"theta={r['theta']}: expected cost outside [1,2] "
                f"(lo={r['cost_lo']:.4f}, hi={r['cost_hi']:.4f})"
            )


def run(cfg: dict, eval_cfg: dict) -> dict:
    """Run the HCT arm end-to-end; write artifacts; return a terse summary (no printing)."""
    seed_everything(int(cfg.get("seed", 0)))

    manifest = json.loads(Path(cfg["split_manifest_path"]).read_text())
    test_ids = [int(g) for g in manifest["test"]]  # frozen order

    scores_df = pd.read_parquet(cfg["scores_path"])
    label_df = pd.read_parquet(cfg["label_table_path"])

    missing = set(test_ids) - set(label_df["GalaxyID"])
    missing_scores = set(test_ids) - set(scores_df["GalaxyID"])
    if missing:
        raise AssertionError(f"{len(missing)} TEST ids missing urns in label_table (e.g. {list(missing)[:3]})")
    if missing_scores:
        raise AssertionError(f"{len(missing_scores)} TEST ids missing backbone scores (e.g. {list(missing_scores)[:3]})")

    thresholds = list(cfg["ai_threshold_sweep"])
    seeds = list(eval_cfg["rater_draw_seeds"])

    pred = run_sweep(scores_df, label_df, test_ids, thresholds, seeds)
    _assert_binary(pred)
    ops = operating_points(pred)
    _assert_cost_bound(ops)

    Path(cfg["export_predictions_path"]).parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(cfg["export_predictions_path"], index=False)
    Path(cfg["export_operating_points_path"]).parent.mkdir(parents=True, exist_ok=True)
    ops.to_csv(cfg["export_operating_points_path"], index=False)

    # Degeneracy check (PLAN decision #5): is the marker locus well-spread or clustered?
    acc_spread = float(ops["accuracy_mean"].max() - ops["accuracy_mean"].min())
    cost_spread = float(ops["cost_mean"].max() - ops["cost_mean"].min())
    return {
        "n_test": len(test_ids),
        "n_thresholds": len(thresholds),
        "n_seeds": len(seeds),
        "n_rows": len(pred),
        "operating_points": ops.to_dict(orient="records"),
        "accuracy_spread": acc_spread,
        "cost_spread": cost_spread,
        "predictions_path": cfg["export_predictions_path"],
        "operating_points_path": cfg["export_operating_points_path"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="HCT arm — sweep the AI threshold over urns + backbone.")
    ap.add_argument("--config", default="configs/arms.yaml")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    args = ap.parse_args()

    arms = yaml.safe_load(Path(args.config).read_text())
    eval_cfg = yaml.safe_load(Path(args.eval_config).read_text())

    hct_cfg = arms["hct"]
    # Defaults wire the arm to the M1/M2 artifacts without bloating arms.yaml; override there if needed.
    hct_cfg.setdefault("split_manifest_path", "data/split_manifest.json")
    hct_cfg.setdefault("label_table_path", "data/label_table.parquet")
    hct_cfg.setdefault("scores_path", "results/backbone_scores.parquet")
    hct_cfg.setdefault("export_predictions_path", "results/hct_predictions.parquet")
    hct_cfg.setdefault("export_operating_points_path", "results/hct_operating_points.csv")
    hct_cfg.setdefault("seed", 0)

    summary = run(hct_cfg, eval_cfg)

    print(f"HCT arm: {summary['n_rows']} rows "
          f"({summary['n_test']} test x {summary['n_thresholds']} theta x {summary['n_seeds']} seeds)")
    print(f"  predictions -> {summary['predictions_path']}")
    print(f"  operating points -> {summary['operating_points_path']}")
    print(f"  accuracy spread = {summary['accuracy_spread']:.4f} | "
          f"cost spread = {summary['cost_spread']:.4f}")
    print("  theta -> (accuracy_mean [lo,hi], cost_mean [lo,hi])")
    for r in summary["operating_points"]:
        print(f"    {r['theta']:.2f} -> acc {r['accuracy_mean']:.4f} "
              f"[{r['accuracy_lo']:.4f},{r['accuracy_hi']:.4f}]  "
              f"cost {r['cost_mean']:.4f} [{r['cost_lo']:.4f},{r['cost_hi']:.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
