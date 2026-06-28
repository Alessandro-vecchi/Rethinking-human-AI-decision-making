"""M6 — accuracy vs expected human-query-cost Pareto frontier (the project's central deliverable).

Every arm's per-instance table (M2–M5) is read through ONE adapter into a common frame, scored by
the ONE shared metrics module (`haidc.eval.metrics`, invariant §4), and assembled into the
frontier. The HCT/L2D asymmetry is preserved by construction: HCT is a coarse marker locus
(cost∈[1,2]); the L2D arms are smooth lines (cost∈[0,1]); AI-alone and single-human are labelled
points. The Okati ORACLE policy uses test labels and is NOT deployable — it is excluded from the
frontier and drawn only as a dashed upper-bound reference.

Two uncertainty sources, reported SEPARATELY (never conflated — REPRODUCIBILITY.md):
  (a) bootstrap CI over the 694 TEST instances  -> finite-test-set noise, drawn as error bars.
      Built on the per-instance correctness AVERAGED over rater seeds, so the CI centres on the
      plotted point and is uniform with AI-alone's deterministic bootstrap.
  (b) seed band = min/max accuracy across rater_draw_seeds [0..4] -> rater-draw noise, drawn as
      whiskers, for the stochastic arms only (HCT, Mozannar, Okati-learned, single-human).

Accuracy is agreement with the crowd-consensus label `y_debiased`, NOT objective truth (HANDOFF §9).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from haidc.eval.metrics import (
    accuracy,
    bootstrap_accuracy_ci,
    expected_human_cost,
    fpr,
    tpr,
)

# Columns every arm's per-instance table must carry before the per-arm sweep column (invariant §4:
# reject a table that cannot be scored on the agreed footing).
_REQUIRED = ["GalaxyID", "y_debiased", "decision", "human_queries", "seed"]
COMMON_COLUMNS = [
    "arm", "policy", "operating_param", "seed",
    "GalaxyID", "decision", "y_debiased", "human_queries",
]

# Per-arm adapter spec: (parquet key in cfg["inputs"], display arm name, sweep column, policy column).
_ARMS = [
    ("hct", "hct", "theta", None),
    ("okati", "l2d_okati", "b", "policy"),
    ("mozannar", "l2d_mozannar", "alpha", "policy"),
]

# Default input locations (overridable via cfg["inputs"]).
_DEFAULT_INPUTS = {
    "scores": "results/backbone_scores.parquet",
    "label_table": "data/label_table.parquet",
    "hct": "results/hct_predictions.parquet",
    "okati": "results/l2d_okati_predictions.parquet",
    "mozannar": "results/l2d_mozannar_predictions.parquet",
}

AI_ALONE = "ai_alone"
SINGLE_HUMAN = "single_human"
OKATI_ORACLE = ("l2d_okati", "oracle")


# --------------------------------------------------------------------------- adapter
def to_common_frame(df, *, arm, sweep_col, policy_col=None, policy=None):
    """Normalise one arm's per-instance table to the common frame; reject if a column is missing.

    `sweep_col` (theta/b/alpha) becomes `operating_param`. Policy comes from `policy_col` if the
    table carries one (Okati/Mozannar), else the constant `policy` (default = the arm name).
    """
    needed = _REQUIRED + [sweep_col] + ([policy_col] if policy_col else [])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{arm} table missing agreed column(s): {missing}")
    pol = df[policy_col] if policy_col else (policy if policy is not None else arm)
    out = pd.DataFrame({
        "arm": arm,
        "policy": pol,
        "operating_param": df[sweep_col].to_numpy(),
        "seed": df["seed"].to_numpy(),
        "GalaxyID": df["GalaxyID"].to_numpy(),
        "decision": df["decision"].to_numpy(),
        "y_debiased": df["y_debiased"].to_numpy(),
        "human_queries": df["human_queries"].to_numpy(),
    })
    return out


# --------------------------------------------------------------------------- scoring
def _seed_mean_correctness(group):
    """Per-instance correctness averaged over rater seeds (the bootstrap target — source (a)).

    Returns a vector with one entry per GalaxyID; for a single-seed (deterministic) arm this is just
    the 0/1 correctness vector, keeping AI-alone uniform with the stochastic arms.
    """
    g = group.copy()
    g["correct"] = (g["decision"].to_numpy() == g["y_debiased"].to_numpy()).astype(float)
    return g.groupby("GalaxyID")["correct"].mean().to_numpy()


def score_points(frame, *, n_resamples, ci, seed):
    """Score every (arm, policy, operating_param) point with the shared metrics + both uncertainties.

    Per point: accuracy/TPR/FPR/cost averaged over seeds (the plotted estimates); bootstrap CI (a)
    on the seed-mean correctness vector; seed band (b) = min/max accuracy across seeds.
    """
    rows = []
    keys = ["arm", "policy", "operating_param"]
    for (arm, policy, op), grp in frame.groupby(keys, sort=False):
        per_seed_acc, per_seed_cost, per_seed_tpr, per_seed_fpr = [], [], [], []
        for _s, gs in grp.groupby("seed"):
            d, y, q = gs["decision"].to_numpy(), gs["y_debiased"].to_numpy(), gs["human_queries"].to_numpy()
            per_seed_acc.append(accuracy(d, y))
            per_seed_cost.append(expected_human_cost(q))
            # TPR/FPR are undefined if a class is absent; the 694-instance test split always has both,
            # but guard for toy inputs so a degenerate slice doesn't crash the whole frontier.
            try:
                per_seed_tpr.append(tpr(d, y))
            except ValueError:
                per_seed_tpr.append(np.nan)
            try:
                per_seed_fpr.append(fpr(d, y))
            except ValueError:
                per_seed_fpr.append(np.nan)
        lo, hi = bootstrap_accuracy_ci(
            _seed_mean_correctness(grp), n_resamples=n_resamples, ci=ci, seed=seed
        )
        n_seeds = grp["seed"].nunique()
        rows.append({
            "arm": arm, "policy": policy, "operating_param": op,
            "accuracy_mean": float(np.mean(per_seed_acc)),
            "ci_lo": lo, "ci_hi": hi,
            "seed_lo": float(np.min(per_seed_acc)), "seed_hi": float(np.max(per_seed_acc)),
            "cost_mean": float(np.mean(per_seed_cost)),
            "tpr": float(np.nanmean(per_seed_tpr)), "fpr": float(np.nanmean(per_seed_fpr)),
            "n_seeds": int(n_seeds), "stochastic": bool(n_seeds > 1),
            "deployable": (arm, policy) != OKATI_ORACLE,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- baselines
def ai_alone_frame(scores_df, y_by_id, *, threshold=0.5):
    """AI-alone as a common frame: one deterministic 'seed', cost 0, decision = score>=threshold."""
    gid = scores_df["GalaxyID"].to_numpy()
    decision = (scores_df["score"].to_numpy() >= threshold).astype(int)
    y = np.array([y_by_id[g] for g in gid])
    return pd.DataFrame({
        "arm": AI_ALONE, "policy": AI_ALONE, "operating_param": float(threshold),
        "seed": 0, "GalaxyID": gid, "decision": decision, "y_debiased": y,
        "human_queries": 0,
    })


def single_human_frame(hct_df):
    """Single-human baseline = HCT's h1 draw (one human, cost 1) — the SAME P(h|x)/sampler as HCT's
    h1 (invariant §3): we reuse the realised h1 column rather than re-sampling, so the baseline and
    HCT's first query are the identical draw. h1 is fixed per (id, seed), so any theta slice serves.
    """
    theta0 = sorted(hct_df["operating_param"].unique())[0]
    s = hct_df[hct_df["operating_param"] == theta0]
    return pd.DataFrame({
        "arm": SINGLE_HUMAN, "policy": SINGLE_HUMAN, "operating_param": 1.0,
        "seed": s["seed"].to_numpy(), "GalaxyID": s["GalaxyID"].to_numpy(),
        "decision": s["h1"].to_numpy(), "y_debiased": s["y_debiased"].to_numpy(),
        "human_queries": 1,
    })


# --------------------------------------------------------------------------- frontier / claim
def _is_deployable(points):
    return ~((points["arm"] == OKATI_ORACLE[0]) & (points["policy"] == OKATI_ORACLE[1]))


def deployable_frontier(points):
    """Non-dominated deployable points. x = cost_mean (lower better), y = accuracy_mean (higher
    better). Point i is dominated iff some deployable j has cost_j<=cost_i AND acc_j>=acc_i with at
    least one strict. The Okati oracle (uses test labels) is excluded before the test."""
    dep = points[_is_deployable(points)].reset_index(drop=True)
    cost = dep["cost_mean"].to_numpy()
    acc = dep["accuracy_mean"].to_numpy()
    keep = np.ones(len(dep), dtype=bool)
    for i in range(len(dep)):
        for j in range(len(dep)):
            if i == j:
                continue
            if cost[j] <= cost[i] and acc[j] >= acc[i] and (cost[j] < cost[i] or acc[j] > acc[i]):
                keep[i] = False
                break
    return dep[keep].sort_values("cost_mean").reset_index(drop=True)


def beats_ai_alone(points, baseline=0.829):
    """Central-claim verdict: per deployable arm, take its best-accuracy point and ask whether its
    bootstrap CI LOWER bound exceeds AI-alone. CI overlapping the baseline = no significant
    difference (do not read a higher point estimate as a win). Oracle is excluded (not deployable).
    """
    dep = points[_is_deployable(points)]
    dep = dep[dep["arm"] != AI_ALONE]
    rows = []
    for arm, grp in dep.groupby("arm", sort=False):
        best = grp.loc[grp["accuracy_mean"].idxmax()]
        rows.append({
            "arm": arm, "policy": best["policy"], "operating_param": best["operating_param"],
            "accuracy_mean": best["accuracy_mean"], "ci_lo": best["ci_lo"], "ci_hi": best["ci_hi"],
            "cost_mean": best["cost_mean"],
            "beats": bool(best["ci_lo"] > baseline),
            "overlaps_baseline": bool(best["ci_lo"] <= baseline <= best["ci_hi"]),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- plotting
def plot_frontier(points, *, baseline, figure_path, caption=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    def _np(rows, col):  # numpy view -> avoids matplotlib's single-element-Series FutureWarning
        return rows[col].to_numpy()

    def _band(rows):  # seed-band whiskers (source b) for stochastic arms
        st = rows[rows["stochastic"]]
        if len(st):
            x, y = _np(st, "cost_mean"), _np(st, "accuracy_mean")
            ax.errorbar(
                x, y, yerr=[y - _np(st, "seed_lo"), _np(st, "seed_hi") - y],
                fmt="none", ecolor="0.6", elinewidth=1.0, capsize=2, zorder=1,
            )

    def _ci(rows, color):  # bootstrap CI error bars (source a)
        x, y = _np(rows, "cost_mean"), _np(rows, "accuracy_mean")
        ax.errorbar(
            x, y, yerr=[y - _np(rows, "ci_lo"), _np(rows, "ci_hi") - y],
            fmt="none", ecolor=color, elinewidth=1.2, capsize=3, alpha=0.8, zorder=2,
        )

    # L2D arms: smooth lines (cost∈[0,1]).
    okati_learned = points[(points["arm"] == "l2d_okati") & (points["policy"] == "learned")].sort_values("cost_mean")
    mozannar = points[points["arm"] == "l2d_mozannar"].sort_values("cost_mean")
    for rows, color, label in [
        (okati_learned, "tab:blue", "L2D-Okati (learned)"),
        (mozannar, "tab:green", "L2D-Mozannar"),
    ]:
        _band(rows)
        _ci(rows, color)
        ax.plot(_np(rows, "cost_mean"), _np(rows, "accuracy_mean"), "-o", color=color, label=label, zorder=3)

    # Okati ORACLE: dashed, non-deployable upper bound.
    oracle = points[(points["arm"] == "l2d_okati") & (points["policy"] == "oracle")].sort_values("cost_mean")
    if len(oracle):
        ax.plot(_np(oracle, "cost_mean"), _np(oracle, "accuracy_mean"), "--", color="tab:blue", alpha=0.5,
                label="L2D-Okati (oracle, non-deployable upper bound)", zorder=2)

    # HCT: coarse marker locus (cost∈[1,2]).
    hct = points[points["arm"] == "hct"]
    _band(hct)
    _ci(hct, "tab:red")
    ax.scatter(_np(hct, "cost_mean"), _np(hct, "accuracy_mean"), marker="s", color="tab:red", s=45,
               label="HCT (coarse marker locus)", zorder=4)

    # Labelled points.
    for arm, color, marker, label in [
        (AI_ALONE, "black", "*", "AI-alone"),
        (SINGLE_HUMAN, "tab:purple", "D", "Single-human"),
    ]:
        r = points[points["arm"] == arm]
        if len(r):
            _band(r)
            _ci(r, color)
            ax.scatter(_np(r, "cost_mean"), _np(r, "accuracy_mean"), marker=marker, color=color, s=90,
                       label=label, zorder=5)

    ax.axhline(baseline, color="black", ls=":", lw=1.0, alpha=0.6)
    ax.set_xlabel("Expected human-query cost (queries / instance)")
    ax.set_ylabel("Accuracy (agreement with crowd consensus, not truth)")
    ax.set_title("HCT vs L2D — accuracy vs human-query cost (Galaxy Zoo binary)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    if caption:
        fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=7, wrap=True)
    Path(figure_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- manifests / io
def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _write_manifest(out_path, *, cfg, inputs, extra):
    manifest = {
        "stage": "eval",
        "config": cfg,
        "seed": int(cfg["bootstrap"]["seed"]),
        "git_sha": _git_sha(),
        "input_table_sha256": {k: _sha256(v) for k, v in inputs.items() if Path(v).exists()},
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }
    Path(out_path).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


CAPTION = (
    "HCT is a coarse marker locus (cost in [1,2], swept via the AI threshold); the L2D arms are "
    "smooth lines (cost in [0,1], swept via their deferral parameter) — a genuine shape asymmetry, "
    "not an artefact. The Okati oracle (dashed) uses test labels and is a non-deployable upper bound. "
    "Accuracy is agreement with the crowd-consensus label, not objective truth. Error bars = "
    "bootstrap CI over the 694 test instances; whiskers = min/max across rater-draw seeds. Regions "
    "where an arm's CI overlaps AI-alone are not significant differences."
)


def build(cfg):
    """Assemble all points, the frontier, the verdict; return the artifacts (no file IO)."""
    inputs = {**_DEFAULT_INPUTS, **cfg.get("inputs", {})}
    bs = cfg["bootstrap"]

    label_df = pd.read_parquet(inputs["label_table"])
    scores_df = pd.read_parquet(inputs["scores"])
    y_by_id = dict(zip(label_df["GalaxyID"], label_df["y_debiased"]))

    frames = [ai_alone_frame(scores_df, y_by_id, threshold=float(cfg.get("ai_threshold", 0.5)))]
    hct_common = None
    for key, arm, sweep, pol_col in _ARMS:
        raw = pd.read_parquet(inputs[key])
        common = to_common_frame(raw, arm=arm, sweep_col=sweep, policy_col=pol_col)
        frames.append(common)
        if arm == "hct":
            hct_common = pd.read_parquet(inputs["hct"]).rename(columns={"theta": "operating_param"})
    frames.append(single_human_frame(hct_common))

    frame = pd.concat(frames, ignore_index=True)
    points = score_points(frame, n_resamples=int(bs["n_resamples"]), ci=float(bs["ci"]), seed=int(bs["seed"]))

    baseline = float(points.loc[points["arm"] == AI_ALONE, "accuracy_mean"].iloc[0])
    frontier = deployable_frontier(points)
    verdict = beats_ai_alone(points, baseline=baseline)
    return {"points": points, "frontier": frontier, "verdict": verdict, "baseline": baseline, "inputs": inputs}


def run(cfg):
    art = build(cfg)
    points, baseline = art["points"], art["baseline"]

    table_path = cfg["table_path"]
    Path(table_path).parent.mkdir(parents=True, exist_ok=True)
    cols = ["arm", "policy", "operating_param", "cost_mean", "accuracy_mean", "ci_lo", "ci_hi",
            "seed_lo", "seed_hi", "tpr", "fpr", "n_seeds", "stochastic", "deployable"]
    points.sort_values(["arm", "policy", "cost_mean"])[cols].to_csv(table_path, index=False)

    plot_frontier(points, baseline=baseline, figure_path=cfg["figure_path"], caption=CAPTION)

    extra = {
        "baseline_ai_alone_accuracy": baseline,
        "frontier_points": art["frontier"][["arm", "policy", "cost_mean", "accuracy_mean"]].to_dict("records"),
        "central_claim_verdict": art["verdict"].to_dict("records"),
        "any_deployable_arm_beats_ai_alone": bool(art["verdict"]["beats"].any()) if len(art["verdict"]) else False,
        "figure_path": cfg["figure_path"], "table_path": table_path,
    }
    _write_manifest(table_path + ".manifest.json", cfg=cfg, inputs=art["inputs"], extra=extra)
    _write_manifest(cfg["figure_path"] + ".manifest.json", cfg=cfg, inputs=art["inputs"], extra=extra)
    return art


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    art = run(cfg)
    v = art["verdict"]
    print(f"AI-alone baseline accuracy = {art['baseline']:.4f}")
    print("Per-deployable-arm best accuracy + bootstrap CI vs AI-alone:")
    for _, r in v.iterrows():
        flag = "BEATS" if r["beats"] else ("ns (overlaps)" if r["overlaps_baseline"] else "below")
        print(f"  {r['arm']:<14} cost={r['cost_mean']:.3f}  acc={r['accuracy_mean']:.4f} "
              f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]  -> {flag}")
    any_beat = bool(v["beats"].any()) if len(v) else False
    print(f"\nVerdict: {'a deployable arm beats AI-alone' if any_beat else 'NO deployable arm significantly beats AI-alone'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
