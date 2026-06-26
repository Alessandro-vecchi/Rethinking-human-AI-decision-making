"""`make data` — build the label/urn table + the committed frozen-split manifest (M1).

Deterministic given configs/data.yaml + seed. Writes:
  * <label_table_path> (parquet, git-ignored) — the per-image urn/label table.
  * <manifest_path> (json, committed)         — frozen split + hashes + provenance numbers.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import yaml

from haidc.data.labels import build_label_table, load_crosswalk
from haidc.data.split import build_manifest, compute_split
from haidc.seed import seed_everything


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(cfg["seed"])
    seed_everything(seed)

    accepted_count = len(load_crosswalk(cfg["inputs"]["crosswalk"]))
    table = build_label_table(cfg)
    n = len(table)
    disagree = int((table["y_count"] != table["y_debiased"]).sum())

    Path(cfg["label_table_path"]).parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cfg["label_table_path"], index=False)

    split = compute_split(table["GalaxyID"].tolist(), seed, cfg["split"])
    extra = {
        "accepted_crosswalk_count": accepted_count,
        "y_count_vs_y_debiased_disagreement": {
            "count": disagree,
            "rate": round(disagree / n, 4),
        },
        "default_label_for_arms": "y_debiased",
        "urn_source": "raw Willett vote counts (HANDOFF §6)",
        "label_table_path": cfg["label_table_path"],
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
    }
    manifest = build_manifest(split, seed, cfg["split"], extra)
    Path(cfg["manifest_path"]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(
        f"label table: N={n} (accepted={accepted_count}) -> {cfg['label_table_path']}\n"
        f"split: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}\n"
        f"y_count vs y_debiased disagreement: {disagree}/{n} ({disagree / n:.4f})\n"
        f"manifest: {cfg['manifest_path']} content_hash={manifest['content_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
