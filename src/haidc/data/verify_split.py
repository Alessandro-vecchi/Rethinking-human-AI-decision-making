"""`make verify-split` — recompute the split from seed=0 and assert the committed hash.

Guards CLAUDE.md invariant #2: every arm must consume the same frozen split. The volatile
manifest fields (timestamp, git sha) are excluded from ``content_hash`` by construction, so a
clean recomputation reproduces the hash exactly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from haidc.data.labels import build_label_table
from haidc.data.split import build_manifest, compute_split
from haidc.seed import seed_everything


def recompute_manifest(config_path: str | Path) -> dict:
    """Rebuild the manifest deterministically from config (no IO side effects)."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    seed = int(cfg["seed"])
    seed_everything(seed)
    table = build_label_table(cfg)
    split = compute_split(table["GalaxyID"].tolist(), seed, cfg["split"])
    return build_manifest(split, seed, cfg["split"], extra={})


def recompute_and_compare(config_path: str | Path) -> dict:
    """Recompute and assert the committed manifest's content_hash matches. Returns the recomputed manifest."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    committed_path = Path(cfg["manifest_path"])
    if not committed_path.exists():
        raise FileNotFoundError(f"no committed manifest at {committed_path} — run `make data` first")
    committed = json.loads(committed_path.read_text())
    recomputed = recompute_manifest(config_path)
    if recomputed["content_hash"] != committed["content_hash"]:
        raise RuntimeError(
            "frozen-split mismatch: committed content_hash="
            f"{committed['content_hash']} but recomputed={recomputed['content_hash']} "
            f"(committed N={committed.get('final_N')}, recomputed N={recomputed['final_N']})"
        )
    return recomputed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    try:
        m = recompute_and_compare(args.config)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"FAIL: {e}")
        return 1
    print(f"OK: split reproduced (N={m['final_N']}, content_hash={m['content_hash']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
