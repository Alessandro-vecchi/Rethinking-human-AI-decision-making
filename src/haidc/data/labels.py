"""Per-image multi-rater label/urn table (M1).

One row per surviving Galaxy Zoo image on the `accepted` crosswalk subset. The urn for HCT's
h1/h2 draw is the RAW Willett vote counts {n_smooth, n_features} (HANDOFF §6). Two label columns
are emitted (DECISIONS.md, 2026-06-26):

  * ``y_count``    = argmax of the raw counts (ties -> smooth). GROUND_TRUTH §4's literal label.
  * ``y_debiased`` = argmax of the Kaggle (== Willett debiased) vote fractions. Okati's published
                     label and the APPROVED DEFAULT every arm consumes.

These disagree on ~21% of images (GZ2 redshift debiasing reclassifies faint smooth->features);
that gap is the disclosed single-human-vs-consensus error rate, not a bug. int64 is enforced on
GalaxyID / dr7objid everywhere (float64 silently collides 18-digit objids).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

# Willett 2013 Task-01 raw vote-count columns.
T01_SMOOTH = "t01_smooth_or_features_a01_smooth_count"
T01_FEATURES = "t01_smooth_or_features_a02_features_or_disk_count"
T01_ARTIFACT = "t01_smooth_or_features_a03_star_or_artifact_count"

# Kaggle training_solutions_rev1 Task-01 aggregate fractions (== Willett debiased).
KAGGLE_SMOOTH = "Class1.1"
KAGGLE_FEATURES = "Class1.2"

OUT_COLS = [
    "GalaxyID", "dr7objid", "n_smooth", "n_features", "n_artifact",
    "y_count", "y_debiased", "source_table",
]


def load_crosswalk(path: str | Path) -> pd.DataFrame:
    """Crosswalk rows with ``accepted == True`` -> [GalaxyID, dr7objid] (int64)."""
    df = pd.read_csv(path, dtype={"GalaxyID": "int64", "dr7objid": "int64"})
    df = df[df["accepted"].astype(bool)]
    return df[["GalaxyID", "dr7objid"]].reset_index(drop=True)


def load_willett(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Concatenate Willett tables -> [dr7objid, n_smooth, n_features, n_artifact, source_table].

    Counts are raw integers. dr7objid is read as int64. Cross-table duplicate objids are
    dropped (keep first); the count is surfaced by build_label_table for the manifest.
    """
    cols = [T01_SMOOTH, T01_FEATURES, T01_ARTIFACT]
    parts = []
    for p in paths:
        src = Path(p).name.split(".")[0]
        df = pd.read_csv(p, dtype={"dr7objid": "int64"}, usecols=["dr7objid", *cols], low_memory=False)
        df = df.rename(columns={T01_SMOOTH: "n_smooth", T01_FEATURES: "n_features", T01_ARTIFACT: "n_artifact"})
        for c in ("n_smooth", "n_features", "n_artifact"):
            df[c] = df[c].round().astype("int64")
        df["source_table"] = src
        parts.append(df[["dr7objid", "n_smooth", "n_features", "n_artifact", "source_table"]])
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates("dr7objid", keep="first").reset_index(drop=True)


def load_kaggle(path: str | Path) -> pd.DataFrame:
    """Kaggle aggregate fractions -> [GalaxyID, frac_smooth, frac_features]."""
    df = pd.read_csv(path, dtype={"GalaxyID": "int64"}, usecols=["GalaxyID", KAGGLE_SMOOTH, KAGGLE_FEATURES])
    return df.rename(columns={KAGGLE_SMOOTH: "frac_smooth", KAGGLE_FEATURES: "frac_features"})


def assemble_label_table(
    crosswalk: pd.DataFrame, willett: pd.DataFrame, kaggle: pd.DataFrame, min_votes: int
) -> pd.DataFrame:
    """Join, filter (star/artifact mode + min votes), and label. Fails loud on any unmatched row."""
    m = crosswalk.merge(willett, on="dr7objid", how="left")
    missing = m["n_smooth"].isna()
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} accepted crosswalk rows have no Willett vote counts "
            "(crosswalk/table mismatch) — refusing to silently drop them."
        )

    # Drop images whose Task-01 mode is star_or_artifact (mirrors Okati's Class1.3 filter).
    artifact_mode = (m["n_artifact"] >= m["n_smooth"]) & (m["n_artifact"] >= m["n_features"])
    m = m[~artifact_mode]

    # Require enough votes for a without-replacement h1/h2 draw.
    m = m[(m["n_smooth"] + m["n_features"]) >= min_votes]

    # y_count: argmax of raw counts; exact ties -> smooth (0), documented.
    m["y_count"] = (m["n_features"] > m["n_smooth"]).astype("int64")

    # y_debiased: argmax of the debiased Kaggle fractions (Okati's label).
    m = m.merge(kaggle, on="GalaxyID", how="left")
    if m["frac_smooth"].isna().any():
        raise ValueError("some surviving galaxies have no Kaggle fractions — cannot set y_debiased.")
    m["y_debiased"] = (m["frac_features"] > m["frac_smooth"]).astype("int64")

    for c in ("GalaxyID", "dr7objid", "n_smooth", "n_features", "n_artifact", "y_count", "y_debiased"):
        m[c] = m[c].astype("int64")
    return m[OUT_COLS].sort_values("GalaxyID").reset_index(drop=True)


def build_label_table(cfg: Mapping, *, frames: Mapping[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Build the label table from config paths, or from injected normalized ``frames`` (tests)."""
    min_votes = int(cfg["dataset"]["min_votes_per_image"])
    if frames is not None:
        cw, wil, kag = frames["crosswalk"], frames["willett"], frames["kaggle"]
    else:
        inp = cfg["inputs"]
        cw = load_crosswalk(inp["crosswalk"])
        wil = load_willett(inp["willett"])
        kag = load_kaggle(inp["kaggle"])
    return assemble_label_table(cw, wil, kag, min_votes)
