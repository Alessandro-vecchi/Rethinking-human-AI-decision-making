"""
verify_galaxyid_assetid.py

Go/no-go gate for Option B (porting the raw GZ2 per-rater protocol).

Question: is the Kaggle Galaxy Challenge `GalaxyID` the same integer as the
GZ2 `asset_id`?  If yes, the join GalaxyID -> asset_id -> dr7objid -> GZ2 row
is a clean CSV merge and Option B is cheap.  If no, this file is useless and
you fall back to image/coordinate matching (a separate sub-project).

Method (pure CSV; no images, no SDSS):
  H: GalaxyID == asset_id.
  Under H, GalaxyID -> asset_id -> objid (gz2_filename_mapping) -> GZ2 row.
  Then Kaggle Class1.1/1.2/1.3 (Task-01 weighted vote fractions, path mult = 1
  at the root) must equal the GZ2 catalogue's Task-01 weighted vote fractions
  for that objid.
    correct identity  -> Pearson r ~ 1.0, tiny median|diff|
    wrong/coincidental-> r ~ 0
  Integer membership alone is NOT sufficient: GalaxyID and asset_id occupy the
  same numeric range, so spurious overlap is high.  The content check decides.

Inputs (all CSV):
  --kaggle   training_solutions_rev1.csv  (cols: GalaxyID, Class1.1 .. Class11.6)
  --mapping  gz2_filename_mapping.csv     (cols: objid, sample, asset_id)
  --gz2      GZ2 Table 1, Willett et al. 2013  (key dr7objid + t01_* weighted fractions)

Use the WEIGHTED (not debiased) GZ2 columns; Kaggle is weighted-but-not-debiased.
"""
from __future__ import annotations
import argparse
import sys
import numpy as np
import pandas as pd

# --- GZ2 Table-1 column names (override via CLI if your release differs) -------
GZ2_OBJID = "dr7objid"
GZ2_T01 = {  # Task-01 response -> GZ2 weighted-fraction column
    "smooth":   "t01_smooth_or_features_a01_smooth_weighted_fraction",
    "features": "t01_smooth_or_features_a02_features_or_disk_weighted_fraction",
    "artifact": "t01_smooth_or_features_a03_star_or_artifact_weighted_fraction",
}
KAGGLE_T01 = {"smooth": "Class1.1", "features": "Class1.2", "artifact": "Class1.3"}


def okati_10k_ids(kaggle: pd.DataFrame) -> pd.Index:
    """Okati's deterministic subset = 10k lowest-sorted GalaxyIDs.
    (sorted(filenames)[:10000]; six-digit IDs => lexicographic == numeric.)"""
    return kaggle["GalaxyID"].astype(int).sort_values().head(10_000).values


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kaggle", required=True)
    p.add_argument("--mapping", required=True)
    p.add_argument("--gz2", required=True)
    p.add_argument("--gz2-objid", default=GZ2_OBJID)
    p.add_argument("--full", action="store_true",
                   help="check all 61,578 Kaggle IDs, not just Okati's 10k")
    p.add_argument("--r-pass", type=float, default=0.99)
    p.add_argument("--diff-pass", type=float, default=0.02)
    a = p.parse_args()

    # Enforce int64 on every join key at PARSE time, not after. The keys (esp.
    # 18-digit SDSS dr7objid, ~5.9e17) exceed float64's exact-integer range (2**53),
    # so if pandas ever infers float64 for a key (it does the moment a null appears),
    # distinct objIDs in the same field round to the same double -> spurious
    # many-to-one merges. A post-hoc .astype(int64) cannot recover that; the precision
    # is already gone. dtype= here also fails LOUD ("Integer column has NA values") on
    # a null instead of silently downcasting -- the desired behavior.
    kaggle = pd.read_csv(a.kaggle, dtype={"GalaxyID": "int64"})
    mapping = pd.read_csv(a.mapping, dtype={"asset_id": "int64", "objid": "int64"})
    gz2 = pd.read_csv(a.gz2, dtype={a.gz2_objid: "int64"}, low_memory=False)

    for df, name, cols in [
        (kaggle, "kaggle", ["GalaxyID", *KAGGLE_T01.values()]),
        (mapping, "mapping", ["asset_id", "objid"]),
        (gz2, "gz2", [a.gz2_objid, *GZ2_T01.values()]),
    ]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            sys.exit(f"[{name}] missing columns: {missing}\n"
                     f"  available (first 20): {list(df.columns)[:20]}")

    ids = kaggle["GalaxyID"].values if a.full else okati_10k_ids(kaggle)
    sub = kaggle[kaggle["GalaxyID"].isin(ids)].copy()
    n = len(sub)

    # ---- Step A: membership (necessary, not sufficient) ----
    # NB: asset_id is a contiguous 1-based row index (range ~[1, 355_990]); GalaxyID
    # is sparse 6-digit and reaches 999_967. So under --full, ~72% of GalaxyIDs simply
    # exceed the asset_id range and this gate REJECTs for a *range* reason, not an
    # identity one. Okati's 10k (all <= ~250k) sit inside the range, giving the clean
    # ~100% membership -- which is why the content check (Step C), not this gate, is
    # what actually decides identity.
    asset_set = set(mapping["asset_id"].values)
    in_map = sub["GalaxyID"].isin(asset_set)
    print(f"checked IDs                : {n}")
    print(f"present as asset_id        : {in_map.sum()} ({in_map.mean():.1%})")
    if in_map.mean() < 0.5:
        print("\nVERDICT: REJECT (identity fails at membership). "
              "GalaxyID is not asset_id; use image/coordinate matching or fall back to Option A.")
        return 1

    # ---- Step B: join GalaxyID==asset_id -> objid -> GZ2 row ----
    m = sub.merge(mapping[["asset_id", "objid"]],
                  left_on="GalaxyID", right_on="asset_id", how="inner")
    m = m.merge(gz2[[a.gz2_objid, *GZ2_T01.values()]],
                left_on="objid", right_on=a.gz2_objid, how="inner")
    print(f"joined to GZ2 rows         : {len(m)} ({len(m)/n:.1%})")
    if len(m) < 0.5 * n:
        print("\nVERDICT: REJECT (join cardinality too low).")
        return 1

    # ---- Step C: content check (decisive) ----
    print("\nTask-01 fraction agreement (Kaggle vs GZ2 weighted):")
    rs, diffs = [], []
    for resp, kcol in KAGGLE_T01.items():
        gcol = GZ2_T01[resp]
        x = m[kcol].astype(float).values
        y = m[gcol].astype(float).values
        ok = np.isfinite(x) & np.isfinite(y)
        r = np.corrcoef(x[ok], y[ok])[0, 1]
        md = float(np.median(np.abs(x[ok] - y[ok])))
        rs.append(r)
        diffs.append(md)
        print(f"  {resp:9s}  Pearson r = {r:6.3f}   median|diff| = {md:.4f}")

    r_min, diff_max = min(rs), max(diffs)
    print(f"\nworst r = {r_min:.3f} (pass >= {a.r_pass}) | "
          f"worst median|diff| = {diff_max:.4f} (pass <= {a.diff_pass})")

    if r_min >= a.r_pass and diff_max <= a.diff_pass:
        print("\nVERDICT: CONFIRMED. GalaxyID == asset_id. "
              "Option B join is valid; coverage ~= join rate above. "
              "Proceed: pull *_count columns from this same GZ2 table to build per-image urns.")
        return 0
    if r_min >= 0.8:
        print("\nVERDICT: LIKELY but below threshold. Probable column mismatch "
              "(debiased vs weighted) or a partial release. Inspect a few rows by hand.")
        return 2
    print("\nVERDICT: REJECT. IDs join but content disagrees -> not the same galaxies. "
          "Use image/coordinate matching or fall back to Option A.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())