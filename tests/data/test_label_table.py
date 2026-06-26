"""M1 label/urn table — written before haidc.data.labels (TESTING.md).

Exercises the pure assembler on tiny normalized fixtures (no big-file dependency):
the join, the star/artifact-mode drop, the min-votes drop, and both label columns.
"""
import numpy as np
import pandas as pd
import pytest

from haidc.data.labels import assemble_label_table

MIN_VOTES = 2


def _crosswalk(pairs):
    return pd.DataFrame(
        {"GalaxyID": [g for g, _ in pairs], "dr7objid": [o for _, o in pairs]}
    ).astype({"GalaxyID": "int64", "dr7objid": "int64"})


def _willett(rows):
    # rows: (dr7objid, n_smooth, n_features, n_artifact, source_table)
    return pd.DataFrame(
        rows, columns=["dr7objid", "n_smooth", "n_features", "n_artifact", "source_table"]
    ).astype(
        {"dr7objid": "int64", "n_smooth": "int64", "n_features": "int64", "n_artifact": "int64"}
    )


def _kaggle(rows):
    # rows: (GalaxyID, frac_smooth, frac_features)
    return pd.DataFrame(rows, columns=["GalaxyID", "frac_smooth", "frac_features"]).astype(
        {"GalaxyID": "int64"}
    )


@pytest.fixture
def fixture_tables():
    # five galaxies covering every code path; dr7objid are 18-digit to stress int64.
    cw = _crosswalk(
        [
            (100, 587733398102409306),  # survives, smooth-dominant, labels agree
            (200, 587733398102409307),  # survives, feature-dominant, labels agree
            (300, 587733398102409308),  # survives, exact count tie -> y_count=0
            (400, 587733398102409309),  # star/artifact mode -> dropped
            (500, 587733398102409310),  # below min votes -> dropped
            (600, 587733398102409311),  # survives, y_count != y_debiased (debias flip)
        ]
    )
    wil = _willett(
        [
            (587733398102409306, 30, 5, 1, "specz"),
            (587733398102409307, 4, 26, 0, "photoz"),
            (587733398102409308, 10, 10, 0, "specz"),
            (587733398102409309, 2, 1, 40, "specz"),   # artifact is the mode
            (587733398102409310, 1, 0, 0, "photoz"),    # n_smooth+n_features = 1 < 2
            (587733398102409311, 18, 12, 0, "specz"),    # raw -> smooth (0)
        ]
    )
    kag = _kaggle(
        [
            (100, 0.85, 0.15),  # debiased -> smooth (0)  == y_count
            (200, 0.10, 0.90),  # debiased -> features (1) == y_count
            (300, 0.55, 0.45),  # debiased -> smooth (0)  == y_count (tie rule)
            (400, 0.50, 0.50),
            (500, 0.99, 0.01),
            (600, 0.40, 0.60),  # debiased -> features (1) != y_count (0)
        ]
    )
    return cw, wil, kag


def test_one_row_per_surviving_galaxy(fixture_tables):
    cw, wil, kag = fixture_tables
    out = assemble_label_table(cw, wil, kag, min_votes=MIN_VOTES)
    # 400 (artifact) and 500 (min votes) dropped -> 4 survivors, no duplicate GalaxyIDs.
    assert sorted(out["GalaxyID"].tolist()) == [100, 200, 300, 600]
    assert out["GalaxyID"].is_unique


def test_min_votes_and_artifact_filters(fixture_tables):
    cw, wil, kag = fixture_tables
    out = assemble_label_table(cw, wil, kag, min_votes=MIN_VOTES)
    assert (out["n_smooth"] + out["n_features"] >= MIN_VOTES).all()
    # no row whose artifact count is the Task-01 mode
    assert not ((out["n_artifact"] >= out["n_smooth"]) & (out["n_artifact"] >= out["n_features"])).any()


def test_y_count_is_argmax_with_tie_to_smooth(fixture_tables):
    cw, wil, kag = fixture_tables
    out = assemble_label_table(cw, wil, kag, min_votes=MIN_VOTES).set_index("GalaxyID")
    assert out.loc[100, "y_count"] == 0      # 30 vs 5
    assert out.loc[200, "y_count"] == 1      # 4 vs 26
    assert out.loc[300, "y_count"] == 0      # 10 vs 10 -> tie -> smooth
    assert out.loc[600, "y_count"] == 0      # 18 vs 12


def test_y_debiased_present_and_can_differ(fixture_tables):
    cw, wil, kag = fixture_tables
    out = assemble_label_table(cw, wil, kag, min_votes=MIN_VOTES).set_index("GalaxyID")
    assert out.loc[600, "y_debiased"] == 1            # debias flips this galaxy
    assert out.loc[600, "y_count"] != out.loc[600, "y_debiased"]
    assert out.loc[100, "y_debiased"] == 0


def test_dtypes_int64(fixture_tables):
    cw, wil, kag = fixture_tables
    out = assemble_label_table(cw, wil, kag, min_votes=MIN_VOTES)
    for col in ("GalaxyID", "dr7objid", "n_smooth", "n_features", "n_artifact", "y_count", "y_debiased"):
        assert out[col].dtype == np.int64, col


def test_sorted_by_galaxyid(fixture_tables):
    cw, wil, kag = fixture_tables
    out = assemble_label_table(cw, wil, kag, min_votes=MIN_VOTES)
    assert out["GalaxyID"].tolist() == sorted(out["GalaxyID"].tolist())


def test_missing_willett_match_fails_loud(fixture_tables):
    cw, wil, kag = fixture_tables
    wil_missing = wil[wil["dr7objid"] != 587733398102409306]  # drop a needed match
    with pytest.raises(Exception):
        assemble_label_table(cw, wil_missing, kag, min_votes=MIN_VOTES)
