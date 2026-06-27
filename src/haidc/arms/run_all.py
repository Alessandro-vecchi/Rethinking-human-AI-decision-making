"""`make arms` — run every arm over the SHARED backbone scores + frozen split (M3-M5).

Thin dispatcher. M3 wires the HCT arm; L2D-Okati (M4), L2D-Mozannar (M5), and the baselines
land here as those milestones complete. Each arm consumes the same `results/backbone_scores.parquet`
and the same `data/split_manifest.json` (CLAUDE.md invariants #2/#7) — this module just sequences
them; it owns no arm logic of its own.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from haidc.arms import hct


def main() -> int:
    ap = argparse.ArgumentParser(description="Run all implemented arms.")
    ap.add_argument("--config", default="configs/arms.yaml")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    args = ap.parse_args()

    arms = yaml.safe_load(Path(args.config).read_text())
    eval_cfg = yaml.safe_load(Path(args.eval_config).read_text())

    # --- HCT (M3) -----------------------------------------------------------
    hct_cfg = arms["hct"]
    hct_cfg.setdefault("split_manifest_path", "data/split_manifest.json")
    hct_cfg.setdefault("label_table_path", "data/label_table.parquet")
    hct_cfg.setdefault("scores_path", "results/backbone_scores.parquet")
    hct_cfg.setdefault("export_predictions_path", "results/hct_predictions.parquet")
    hct_cfg.setdefault("export_operating_points_path", "results/hct_operating_points.csv")
    hct_cfg.setdefault("seed", 0)
    summary = hct.run(hct_cfg, eval_cfg)
    print(f"[HCT] {summary['n_rows']} rows -> {summary['predictions_path']} "
          f"(acc spread {summary['accuracy_spread']:.4f}, cost spread {summary['cost_spread']:.4f})")

    # --- L2D-Okati (M4), L2D-Mozannar (M5), baselines: not yet implemented --
    print("[L2D-Okati]    pending (M4)")
    print("[L2D-Mozannar] pending (M5)")
    print("[baselines]    pending (M6)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
