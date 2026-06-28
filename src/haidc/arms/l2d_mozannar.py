"""`make arms` (L2D-Mozannar portion) — Mozannar & Sontag 2020 "Consistent Estimators for Learning
to Defer to an Expert" (ICML 2020, arXiv:2006.01862) (M5).

Option B (CLAUDE.md invariants #1/#2): the classifier is the FROZEN shared M2 backbone; we learn
ONLY the rejector logit `g_⊥` on the backbone's 2048-d embeddings. Co-training a fresh classifier
(Mozannar's upstream default) would make a different classifier per arm and confound the head-to-head
comparison — the shared frozen backbone removes that confound. This is the documented deviation
(DECISIONS 2026-06-28), identical in spirit to the L2D-Okati Option-B treatment, which keeps the two
L2D arms comparable.

Surrogate (paper §4): a single softmax over `K+1 = Y ∪ {⊥}` logits `g_i`, with the consistent loss

    L_CE^α(h,r,x,y,m) = −(α·1{m=y} + 1{m≠y})·log softmax_y(g) − 1{m=y}·log softmax_⊥(g)      (eq 10)

and `L_CE^1 = L_CE` (eq 7). Test-time rule (eq 6): predict `argmax_{y∈Y} g_y`, else DEFER iff
`g_⊥ ≥ max_{y∈Y} g_y`. `l_ce_alpha` below is transcribed from the upstream `reject_CrossEntropyLoss`
+ `train_reject` cost assignment in `third_party/mozannar2020` (SHA e84f3ee), cifar/cifar10_defer_ours
.ipynb — VERIFIED against eq (10) (tasks/M5-PLAN.md §2). NB: the upstream *docstring* for the `m2`
weight is mislabeled (it swaps the indicator); the upstream *code* and our port follow eq (10), where
α multiplies `1{m=y}` on the classifier term. We use natural `log_softmax` (numerically stable); the
upstream `log2` differs only by the constant 1/ln2, which rescales the loss/lr and never changes the
minimizer or the argmax deferral rule.

Option-B construction. The two frozen class logits are reconstructed from the backbone `score`:
`g_1 = log(score)`, `g_0 = log(1−score)` (so `softmax(g_0,g_1) = (1−score, score)` and
`exp(g_0)+exp(g_1)=1`). The trainable scalar head emits `g_⊥(x) = MLP(embeddings)`; then `Z = 1+e^{g_⊥}`,
`softmax_⊥ = σ(g_⊥)`, `softmax_y = p_y/(1+e^{g_⊥})`, and defer iff `g_⊥ ≥ log(max(score,1−score))`.
`L_CE^α` is convex in g, hence convex in `g_⊥` alone; at α=1, eq (9)/Prop 2 give the `g_⊥` minimizer
(holding the near-Bayes class logits fixed) = `log q(x)`, recovering the Bayes rejector
`r^B = 1{max_y η_y ≤ P(Y=M|x)}`. So freezing the (near-Bayes) M2 classifier and learning only `g_⊥`
is a CONSISTENT rejector surrogate at α=1; α≠1 deliberately shifts the operating point.

Sweep knob = α (`configs/arms.yaml l2d_mozannar.defer_cost_sweep`): small α → more deferral, large
α → automate-everything → AI-alone. α is a reweighting hyperparameter (paper p.6: "encourage or
hinder the action of deferral"), the analog of Okati's budget `b` — NOT a literal monetary cost.

Expert term (invariant §3 — the SAME P(h|x) as HCT/Okati/single-human): the loss uses the EXPECTED
expert correctness `q(x)=P(m=y|x)` = the urn vote-share matching `y_debiased` = `1 − minority share`
(soft, deterministic). This is the Rao-Blackwellized per-sample loss and matches the population loss
(eq 8 uses `P(Y=M|X)`). The rejector is fit ONCE per α; the rater seeds enter only at the REALIZED
test-time human draw (one human per (instance, seed) via the shared M1 sampler = HCT's h1), exactly
mirroring the L2D-Okati Option-B treatment.

Cost semantics (a DIFFERENT regime from HCT's [1,2] — flagged for M6): human_queries = 1 iff the
policy DEFERS, else 0, so the expected human cost == the deferral fraction in [0,1].

Label convention (project-wide): 0 = smooth/early-type, 1 = features/disk = spiral.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Reuse the stable, shared Option-B helpers established by the L2D-Okati arm (keeps the two L2D arms
# wired to identical inputs / guards; do not re-implement). l2d_okati is import-safe (no torch at import).
from haidc.arms.l2d_okati import (
    ai_label_from_score,
    embedding_feature_cols,
    expected_human_loss,
    validate_embeddings_frame,
    _assert_binary,
    _assert_cost_unit,
    _assert_frozen_split,
    _assert_scores_match,
    _draw_humans_per_seed,
)
from haidc.eval.metrics import accuracy, expected_human_cost
from haidc.seed import seed_everything

BOT_INDEX = 2  # logits are [g0, g1, g_bot]; the reject category ⊥ is the last

PRED_COLUMNS = [
    "GalaxyID", "y_debiased", "ai_label", "defer", "human",
    "decision", "human_queries", "alpha", "seed", "policy",
]

OPS_COLUMNS = [
    "alpha", "policy", "accuracy_mean", "accuracy_lo", "accuracy_hi",
    "cost_mean", "deferral_fraction", "n_seeds",
]


# --------------------------------------------------------------------------- surrogate (THE gate)
def l_ce_alpha(logits, y, q, alpha: float):
    """Mozannar eq (10) consistent surrogate, soft-`q` form. Per-sample loss (no reduction).

    Args:
      logits: (N, 3) tensor of [g0, g1, g_bot] (bot = reject category at index 2).
      y:      (N,) long tensor of class labels in {0,1} (indexes the classifier term).
      q:      (N,) float tensor = 1{m=y} (hard) or P(m=y|x) (soft expected correctness).
      alpha:  scalar reweighting parameter (α=1 ⇒ L_CE, eq 7).

    L_i = −(α·q_i + (1−q_i))·log_softmax(logits)_{y_i} − q_i·log_softmax(logits)_⊥.
    Transcribed from upstream `reject_CrossEntropyLoss` (SHA e84f3ee), verified vs eq (10).
    """
    import torch

    logp = torch.log_softmax(logits, dim=-1)
    logp_y = logp[torch.arange(logits.shape[0]), y]
    logp_bot = logp[:, BOT_INDEX]
    return -(alpha * q + (1.0 - q)) * logp_y - q * logp_bot


# --------------------------------------------------------------------------- frozen classifier (Option B)
def class_logits_from_score(score):
    """Reconstruct the two FROZEN class logits from the backbone score: (g0, g1) = (log(1−s), log(s)).

    softmax(g0, g1) == (1−s, s) and exp(g0)+exp(g1)==1, so appending a trainable g_bot makes
    softmax_⊥ == σ(g_bot) over the 3-way head. Clipped off {0,1} to keep the logs finite.
    """
    s = np.clip(np.asarray(score, dtype=float), 1e-12, 1.0 - 1e-12)
    return np.log(1.0 - s), np.log(s)


def expected_correctness(n_smooth, n_features, y):
    """q(x) = P(single urn draw == y) = vote-share matching y = 1 − expected_human_loss(...)."""
    return 1.0 - expected_human_loss(n_smooth, n_features, y)


def bayes_defer_mask(score, q):
    """Bayes rejector r^B = 1{max_y η_y ≤ P(Y=M|x)} with η = (1−score, score), P(Y=M|x)=q."""
    score = np.asarray(score, dtype=float)
    top = np.maximum(score, 1.0 - score)
    return np.asarray(q, dtype=float) >= top


def mozannar_defer_mask(g_bot, score):
    """Test-time rule (eq 6): defer iff g_bot ≥ max(g0,g1) = log(max(score,1−score))."""
    g_bot = np.asarray(g_bot, dtype=float)
    g0, g1 = class_logits_from_score(score)
    return g_bot >= np.maximum(g0, g1)


# --------------------------------------------------------------------------- rejector training
def fit_rejector_mozannar(emb, score, y, q, alpha: float, *, seed: int = 0, hidden: int = 64,
                          epochs: int = 200, lr: float = 1e-3, weight_decay: float = 0.0):
    """Train the scalar rejector head g_⊥ = MLP(emb) by minimizing mean `L_CE^α` over the batch.

    The class logits g0,g1 are FROZEN (from `score`); only the MLP (Linear(d,hidden)→ReLU→Linear(hidden,1))
    is trained. Full-batch Adam, CPU, `seed_everything` ⇒ deterministic. Mirrors the L2D-Okati
    rejector recipe (hidden=64, 200 epochs, lr=1e-3); NOT tuned to a target number (CLAUDE.md).
    """
    import torch
    from torch import nn

    seed_everything(int(seed))
    X = torch.tensor(np.asarray(emb, dtype=np.float32))
    g0, g1 = class_logits_from_score(score)
    g0 = torch.tensor(np.asarray(g0, dtype=np.float32))
    g1 = torch.tensor(np.asarray(g1, dtype=np.float32))
    yv = torch.tensor(np.asarray(y, dtype=np.int64))
    qv = torch.tensor(np.asarray(q, dtype=np.float32))

    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    net.train()
    for _ in range(int(epochs)):
        opt.zero_grad()
        g_bot = net(X).squeeze(-1)
        logits = torch.stack([g0, g1, g_bot], dim=1)
        l_ce_alpha(logits, yv, qv, alpha).mean().backward()
        opt.step()
    net.eval()
    return net


def mozannar_g_bot(model, emb) -> np.ndarray:
    """Apply the trained rejector head: g_⊥(x) for each row of `emb`."""
    import torch

    with torch.no_grad():
        return model(torch.tensor(np.asarray(emb, dtype=np.float32))).squeeze(-1).cpu().numpy()


# --------------------------------------------------------------------------- alpha sweep
def _frozen_test_arrays(emb_df, label_df, ids):
    """Return (emb, score, ai_label, y) aligned to the frozen id order for `ids`."""
    feats = embedding_feature_cols(emb_df)
    e = emb_df.set_index("GalaxyID")
    lab = label_df.set_index("GalaxyID")
    emb = e.loc[ids, feats].to_numpy(dtype=float)
    score = e.loc[ids, "score"].to_numpy(dtype=float)
    y = lab.loc[ids, "y_debiased"].to_numpy().astype(np.int64)
    return emb, score, ai_label_from_score(score), y


def run_alpha_sweep(emb_df, label_df, train_ids, test_ids, alphas, seeds,
                    *, seed: int = 0, **fit_kw) -> pd.DataFrame:
    """policy="mozannar": for each α, fit g_⊥ on TRAIN (frozen classifier), apply the deferral rule
    label-free at TEST, and realize the human draw per seed. One smooth, deployable curve over α.
    """
    train_ids = [int(g) for g in train_ids]
    test_ids = [int(g) for g in test_ids]
    feats = embedding_feature_cols(emb_df)
    e = emb_df.set_index("GalaxyID")
    lab = label_df.set_index("GalaxyID")

    tr_emb = e.loc[train_ids, feats].to_numpy(dtype=float)
    tr_score = e.loc[train_ids, "score"].to_numpy(dtype=float)
    tr_y = lab.loc[train_ids, "y_debiased"].to_numpy().astype(np.int64)
    tr_q = expected_correctness(lab.loc[train_ids, "n_smooth"].to_numpy(),
                                lab.loc[train_ids, "n_features"].to_numpy(), tr_y)

    te_emb, te_score, ai, te_y = _frozen_test_arrays(emb_df, label_df, test_ids)
    humans_by_seed = {int(s): _draw_humans_per_seed(label_df, test_ids, int(s)) for s in seeds}

    rows = []
    for alpha in alphas:
        model = fit_rejector_mozannar(tr_emb, tr_score, tr_y, tr_q, float(alpha), seed=seed, **fit_kw)
        defer = mozannar_defer_mask(mozannar_g_bot(model, te_emb), te_score).astype(np.int64)
        for s in seeds:
            h_arr = np.array([humans_by_seed[int(s)][g] for g in test_ids], dtype=np.int64)
            decision = np.where(defer == 1, h_arr, ai)
            for k, gid in enumerate(test_ids):
                rows.append((
                    gid, int(te_y[k]), int(ai[k]), int(defer[k]), int(h_arr[k]),
                    int(decision[k]), int(defer[k]), float(alpha), int(s), "mozannar",
                ))
    return pd.DataFrame(rows, columns=PRED_COLUMNS)


# --------------------------------------------------------------------------- operating points
def operating_points(pred: pd.DataFrame) -> pd.DataFrame:
    """Collapse to (alpha, policy) operating points via the SHARED eval metrics (mean + band over seeds)."""
    recs = []
    for (alpha, policy), g in pred.groupby(["alpha", "policy"]):
        accs, costs, defs = [], [], []
        for _seed, gs in g.groupby("seed"):
            accs.append(accuracy(gs["decision"].to_numpy(), gs["y_debiased"].to_numpy()))
            costs.append(expected_human_cost(gs["human_queries"].to_numpy()))
            defs.append(float(gs["defer"].mean()))
        accs, costs = np.array(accs), np.array(costs)
        recs.append({
            "alpha": float(alpha), "policy": policy,
            "accuracy_mean": accs.mean(), "accuracy_lo": accs.min(), "accuracy_hi": accs.max(),
            "cost_mean": costs.mean(), "deferral_fraction": float(np.mean(defs)),
            "n_seeds": len(accs),
        })
    return pd.DataFrame(recs, columns=OPS_COLUMNS).sort_values(["policy", "alpha"]).reset_index(drop=True)


# --------------------------------------------------------------------------- driver
def run(cfg: dict, eval_cfg: dict) -> dict:
    """Run the L2D-Mozannar arm end-to-end; write artifacts; return a terse summary (no printing)."""
    seed_everything(int(cfg.get("seed", 0)))

    manifest = json.loads(Path(cfg["split_manifest_path"]).read_text())
    test_ids = [int(g) for g in manifest["test"]]
    _assert_frozen_split(manifest, test_ids)  # guard CLAUDE.md #2 before touching anything else
    train_ids = [int(g) for g in manifest["train"]]

    scores_df = pd.read_parquet(cfg["scores_path"])
    label_df = pd.read_parquet(cfg["label_table_path"])
    missing = set(test_ids) - set(label_df["GalaxyID"])
    if missing:
        raise AssertionError(f"{len(missing)} TEST ids missing urns in label_table (e.g. {list(missing)[:3]})")

    emb_path = cfg.get("embeddings_path", "results/backbone_embeddings.parquet")
    if not Path(emb_path).exists():
        raise FileNotFoundError(
            f"L2D-Mozannar needs the backbone embeddings ({emb_path}); run the Colab export first."
        )
    emb_df = pd.read_parquet(emb_path)
    validate_embeddings_frame(emb_df, manifest)
    _assert_scores_match(emb_df, scores_df, test_ids)  # frozen-model provenance

    alphas = list(cfg["defer_cost_sweep"])
    seeds = list(eval_cfg["rater_draw_seeds"])

    pred = run_alpha_sweep(emb_df, label_df, train_ids, test_ids, alphas, seeds,
                           seed=int(cfg.get("seed", 0)))

    _assert_binary(pred)
    ops = operating_points(pred)
    _assert_cost_unit(ops)

    Path(cfg["export_predictions_path"]).parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(cfg["export_predictions_path"], index=False)
    Path(cfg["export_operating_points_path"]).parent.mkdir(parents=True, exist_ok=True)
    ops.to_csv(cfg["export_operating_points_path"], index=False)

    # AI-alone reference (no deferral, identical across α) — the automate-everything anchor.
    ai_alone = accuracy(
        pred[pred["seed"] == seeds[0]].drop_duplicates("GalaxyID")["ai_label"].to_numpy(),
        pred[pred["seed"] == seeds[0]].drop_duplicates("GalaxyID")["y_debiased"].to_numpy(),
    )
    best = ops.loc[ops["accuracy_mean"].idxmax()]
    summary = {
        "n_test": len(test_ids),
        "n_alphas": len(alphas),
        "n_seeds": len(seeds),
        "n_rows": len(pred),
        "ai_alone_accuracy": float(ai_alone),
        "best_accuracy": float(best["accuracy_mean"]),
        "best_alpha": float(best["alpha"]),
        "beats_ai_alone": bool(best["accuracy_mean"] > ai_alone + 1e-9),
        "deferral_min": float(ops["deferral_fraction"].min()),
        "deferral_max": float(ops["deferral_fraction"].max()),
        "operating_points": ops.to_dict(orient="records"),
        "predictions_path": cfg["export_predictions_path"],
        "operating_points_path": cfg["export_operating_points_path"],
    }
    return summary


def _wire_defaults(cfg: dict) -> dict:
    cfg.setdefault("split_manifest_path", "data/split_manifest.json")
    cfg.setdefault("label_table_path", "data/label_table.parquet")
    cfg.setdefault("scores_path", "results/backbone_scores.parquet")
    cfg.setdefault("embeddings_path", "results/backbone_embeddings.parquet")
    cfg.setdefault("export_predictions_path", "results/l2d_mozannar_predictions.parquet")
    cfg.setdefault("export_operating_points_path", "results/l2d_mozannar_operating_points.csv")
    cfg.setdefault("seed", 0)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(
        description="L2D-Mozannar arm — consistent surrogate L_CE^α over a frozen backbone."
    )
    ap.add_argument("--config", default="configs/arms.yaml")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    args = ap.parse_args()

    arms = yaml.safe_load(Path(args.config).read_text())
    eval_cfg = yaml.safe_load(Path(args.eval_config).read_text())
    cfg = _wire_defaults(arms["l2d_mozannar"])

    summary = run(cfg, eval_cfg)

    print(f"L2D-Mozannar arm: {summary['n_rows']} rows "
          f"({summary['n_test']} test x {summary['n_alphas']} α x {summary['n_seeds']} seeds)")
    print(f"  predictions -> {summary['predictions_path']}")
    print(f"  operating points -> {summary['operating_points_path']}")
    print(f"  AI-alone = {summary['ai_alone_accuracy']:.4f} | best acc = {summary['best_accuracy']:.4f} "
          f"@ α={summary['best_alpha']:g} | beats AI-alone: {summary['beats_ai_alone']} "
          f"| deferral spans {summary['deferral_min']:.3f}->{summary['deferral_max']:.3f}")
    print("  (α, policy) -> (accuracy_mean [lo,hi], cost_mean=deferral)")
    for r in summary["operating_points"]:
        print(f"    α={r['alpha']:<6g} {r['policy']} -> acc {r['accuracy_mean']:.4f} "
              f"[{r['accuracy_lo']:.4f},{r['accuracy_hi']:.4f}]  cost {r['cost_mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
