"""
match_galaxyid_to_dr7objid.py  (v3 — debiased flavor, mutual-NN acceptance)

Recover Kaggle GalaxyID -> GZ2 dr7objid by matching vote signatures.

Findings that drove v3 (verified on the real files):
  * Kaggle targets are the Willett 2013 DEBIASED vote fractions (path-scaled),
    NOT weighted_fraction. Among true (mutual-NN) pairs, Kaggle == GZ2 debiased
    to a median per-dimension diff of 0.0001.
  * True matches can still show moderate L2 because a few debiasing-edge
    dimensions differ; a strict eps rejected them. MUTUAL nearest-neighbor is
    the robust acceptance: accept (K,H) iff H is K's NN and K is H's NN.
  * zoo2MainSpecz is spectroscopic-only; ~28% of Kaggle is absent (photo-z).
    Pass BOTH zoo2MainSpecz and zoo2MainPhotoz to --gz2 to recover them.

Signature: 37-dim path-scaled debiased vote vector (Willett Table 2 tree order).
Requires: numpy, pandas, scipy.
"""
from __future__ import annotations
import argparse, re
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

TASK_SIZES = [3, 2, 2, 2, 4, 2, 3, 7, 3, 3, 6]
OFF = np.cumsum([0] + TASK_SIZES)


def blocks(): return [slice(OFF[i], OFF[i + 1]) for i in range(len(TASK_SIZES))]
def i(t, a): return OFF[t - 1] + a


def path_scale(within: np.ndarray) -> np.ndarray:
    N = within.shape[0]; w = np.ones((N, 12))
    w[:, 2] = within[:, i(1, 1)]
    w[:, 7] = within[:, i(1, 0)]
    w[:, 3] = within[:, i(2, 1)] * w[:, 2]
    w[:, 4] = within[:, i(2, 1)] * w[:, 2]
    w[:, 5] = within[:, i(2, 1)] * w[:, 2]
    w[:, 9] = within[:, i(2, 0)] * w[:, 2]
    w[:, 8] = within[:, i(6, 0)]
    w[:, 10] = within[:, i(4, 0)] * w[:, 4]
    w[:, 11] = within[:, i(4, 0)] * w[:, 4]
    out = within.copy()
    for t, sl in enumerate(blocks(), 1):
        out[:, sl] = within[:, sl] * w[:, t][:, None]
    return out


def kaggle_sig(kag):
    cols = [f"Class{t}.{a}" for t, sz in enumerate(TASK_SIZES, 1) for a in range(1, sz + 1)]
    ids = kag["GalaxyID"].astype(int).values
    return ids, np.nan_to_num(kag[cols].values.astype(float), nan=0.0)


def gz2_sig(paths, flavor):
    objids, sigs = [], []
    for p in paths:
        df = pd.read_csv(p, dtype={"dr7objid": "int64"}, low_memory=False)
        suf = f"_{flavor}"
        cols = [c for c in df.columns
                if c.endswith(suf) and not (flavor == "fraction" and c.endswith("_weighted_fraction"))]
        cols = sorted(cols, key=lambda c: int(re.search(r"_a(\d+)_", c).group(1)))
        if len(cols) != 37:
            raise SystemExit(f"{p}: found {len(cols)} '{flavor}' cols, expected 37")
        objids.append(df["dr7objid"].astype(np.int64).values)
        sigs.append(path_scale(np.nan_to_num(df[cols].values.astype(float), nan=0.0)))
    return np.concatenate(objids), np.concatenate(sigs, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kaggle", required=True)
    p.add_argument("--gz2", required=True, nargs="+",
                   help="one or more GZ2 tables (pass zoo2MainSpecz AND zoo2MainPhotoz)")
    p.add_argument("--flavor", default="debiased",
                   choices=["fraction", "weighted_fraction", "debiased"])
    p.add_argument("--cap", type=float, default=0.30, help="max L2 dist for an accepted mutual pair")
    p.add_argument("--out", default="galaxyid_to_dr7objid.csv")
    p.add_argument("--full", action="store_true")
    a = p.parse_args()

    kag = pd.read_csv(a.kaggle, dtype={"GalaxyID": "int64"})
    k_ids, k_sig = kaggle_sig(kag)
    if not a.full:
        keep = np.isin(k_ids, np.sort(k_ids)[:10_000]); k_ids, k_sig = k_ids[keep], k_sig[keep]
    h_obj, h_sig = gz2_sig(a.gz2, a.flavor)
    print(f"matching {len(k_ids)} Kaggle vs {len(h_obj)} GZ2 rows "
          f"(flavor={a.flavor}, {len(a.gz2)} table(s))")

    d1, i1 = cKDTree(h_sig).query(k_sig, k=1)
    _, back = cKDTree(k_sig).query(h_sig[i1], k=1)
    mutual = (k_ids[back] == k_ids)
    accept = mutual & (d1 < a.cap)

    print(f"mutual-NN          : {mutual.sum():5d} ({mutual.mean():.1%})")
    print(f"accepted (mutual & dist<{a.cap}): {accept.sum():5d} ({accept.mean():.1%})")
    if accept.any():
        md = np.median(np.abs(k_sig[accept] - h_sig[i1[accept]]))
        print(f"precision check: median per-dim|diff| on accepted = {md:.4f} (expect ~0)")
    print(f"likely absent (non-mutual, dist>0.30): {((~mutual)&(d1>0.30)).mean():.1%} "
          f"(add the photo-z table if high)")

    pd.DataFrame({"GalaxyID": k_ids, "dr7objid": h_obj[i1], "dist": d1,
                  "mutual": mutual, "accepted": accept}).to_csv(a.out, index=False)
    print(f"wrote {a.out} ({accept.sum()} accepted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())