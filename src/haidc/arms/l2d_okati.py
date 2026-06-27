"""`make arms` (L2D-Okati portion) — Okati 2021 "Differentiable Learning Under Triage" (M4).

Option B (CLAUDE.md invariants #1/#2): the classifier is the FROZEN shared M2 backbone; we learn
ONLY Okati's triage/deferral policy on top. Co-training a fresh classifier (Okati's upstream
default) would make a different classifier per arm and confound the head-to-head comparison — the
shared frozen backbone removes that confound. This is the documented deviation from Okati's
co-training (DECISIONS 2026-06-27).

Mechanism (GROUND_TRUTH §3, Okati Theorem 3): for a fixed model m, the optimal triage is a
deterministic THRESHOLD on the per-instance gap  E_y[l(m(x),y)] - E_h[l(h,y)], with budget b
capping the deferral fraction E[pi] <= b. `find_machine_samples` below is transcribed from Okati
train.ipynb Cell-8 (SHA 43ec215), with two documented boundary fixes (see that function).

We realize TWO clearly-distinguished policies (the orchestrator's hard constraint — M6 must NEVER
plot the oracle as Okati's deployed method; they answer different questions):
  - policy="oracle": `find_machine_samples` run directly on the frozen TEST losses. This is the
    OPTIMAL fixed-classifier deferral — an UPPER BOUND that decides deferral using test labels, so
    it is NOT deployable. It is also the envelope the learned policy must sit at or below.
  - policy="learned": a small rejector (Okati's gnet, here an MLP on the backbone's 2048-d
    penultimate embeddings) trained to approximate the gap ranking, applied LABEL-FREE at test.
    This is the deployable head-to-head curve vs HCT / AI-alone. It requires the embedding export
    (results/backbone_embeddings.parquet, a Colab GPU pass); when absent, the learned policy is
    skipped and only the oracle frontier is produced.

Cost semantics (a DIFFERENT regime from HCT's [1,2] — flagged for M6): human_queries = 1 iff the
policy DEFERS that instance, else 0, so the expected human cost == the deferral fraction in [0,1].

Human term (invariant §3 — the SAME P(h|x) as HCT and the single-human baseline): the deferral
DECISION uses the deterministic EXPECTED human 0/1 loss = minority share of the urn (matching
Okati's use of the expected `hloss` in find_machine_samples; seed-independent). The REALIZED
decision when deferred draws ONE human per (instance, seed) from the same urn via the shared M1
sampler — this is exactly HCT's `h1` at matching (id-order, seed), so the arms stay coupled by the
shared human model.

Label convention (project-wide): 0 = smooth/early-type, 1 = features/disk = spiral.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from haidc.data.sampler import draw_h1_h2
from haidc.data.split import _sha256
from haidc.eval.metrics import accuracy, expected_human_cost
from haidc.seed import seed_everything

AI_THRESHOLD = 0.5  # the backbone's anchor threshold; b (not theta) is Okati's sweep knob
EMB_PREFIX = "e"    # embedding feature columns are e0..e2047 (penultimate ResNet-50 features)

PRED_COLUMNS = [
    "GalaxyID", "y_debiased", "ai_label", "defer", "human",
    "decision", "human_queries", "b", "seed", "policy",
]

OPS_COLUMNS = [
    "b", "policy", "accuracy_mean", "accuracy_lo", "accuracy_hi",
    "cost_mean", "deferral_fraction", "n_seeds",
]


# --------------------------------------------------------------------------- gap building blocks
def ai_label_from_score(score, threshold: float = AI_THRESHOLD):
    """Fixed AI prediction: 1[score >= threshold]. Scalars or arrays -> int array."""
    return (np.asarray(score, dtype=float) >= threshold).astype(np.int64)


def machine_loss_01(score, y, threshold: float = AI_THRESHOLD):
    """Per-instance machine 0/1 loss: 1{ai_label != y} (matches the project accuracy metric)."""
    return (ai_label_from_score(score, threshold) != np.asarray(y)).astype(np.int64)


def expected_human_loss(n_smooth, n_features, y):
    """Expected human 0/1 loss = P(single urn draw != y) = minority share of the urn wrt y.

    y=1 (spiral) -> n_smooth / N ; y=0 (smooth) -> n_features / N. Deterministic (no seed).
    """
    ns = np.asarray(n_smooth, dtype=float)
    nf = np.asarray(n_features, dtype=float)
    y = np.asarray(y)
    total = ns + nf
    return np.where(y == 1, ns / total, nf / total)


def find_machine_samples(machine_loss, human_loss, budget_b: float) -> np.ndarray:
    """Theorem-3 threshold triage — transcribed from Okati train.ipynb Cell-8 (SHA 43ec215).

    Sort by the gap `diff = machine_loss - human_loss` and route the largest-gap instances (where
    the machine is worse than the human) to the human, up to a budget of `int(b*N)` deferrals, but
    NEVER an instance whose gap <= 0 (the machine is at least as good). Returns the indices routed
    to the MACHINE; deferred = the complement.

    Two boundary fixes vs the verbatim upstream cell (documented in DECISIONS):
      - b=0 -> defer NONE. Upstream computes `argsorted[:-num_outsource]`; at num_outsource=0 the
        Python slice `[:-0] == [:0]` is EMPTY, which would defer everything. We special-case it.
      - if even the largest gap is <= 0 (the machine is never worse), defer NONE rather than force
        one deferral (the upstream while-loop stops at index=-1 and would slice off one instance).
    """
    machine_loss = np.asarray(machine_loss, dtype=float)
    human_loss = np.asarray(human_loss, dtype=float)
    n = machine_loss.shape[0]
    diff = machine_loss - human_loss
    argsorted = np.argsort(diff, kind="stable")  # ascending; largest gap last
    num_outsource = int(budget_b * n)            # budget cap on deferrals
    if num_outsource <= 0:
        return argsorted                          # b=0: all machine, zero deferrals
    index = -num_outsource
    while index < -1 and diff[argsorted[index]] <= 0:
        index += 1
    if diff[argsorted[index]] <= 0:
        return argsorted                          # no positive-gap instance: defer none
    return argsorted[:index]


def oracle_defer_mask(machine_loss, human_loss, budget_b: float) -> np.ndarray:
    """Boolean defer mask (True = defer) = complement of find_machine_samples."""
    n = np.asarray(machine_loss).shape[0]
    machine = find_machine_samples(machine_loss, human_loss, budget_b)
    mask = np.ones(n, dtype=bool)
    mask[machine] = False
    return mask


def topk_defer_mask(scores, budget_b: float) -> np.ndarray:
    """Test-time budget rule for the LEARNED policy: defer the floor(b*N) highest-scoring instances.

    Applies the budget label-free at test (the rejector's P(defer) is the score). Ties broken by a
    stable sort (lowest original index deferred last), so the mask is deterministic.
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.shape[0]
    k = int(budget_b * n)
    mask = np.zeros(n, dtype=bool)
    if k <= 0:
        return mask
    order = np.argsort(scores, kind="stable")  # ascending
    mask[order[-k:]] = True
    return mask


# --------------------------------------------------------------------------- human draws (shared P(h|x))
def _draw_humans_per_seed(label_df: pd.DataFrame, test_ids, seed: int) -> dict:
    """Draw ONE human per id for one rater seed, iterating ids in the frozen order.

    Reuses the M1 sampler's `draw_h1_h2` and takes h1 — so the realized human equals HCT's h1 at the
    same (id-order, seed), keeping the arms coupled by the shared human model (invariant §3).
    """
    urn = label_df.set_index("GalaxyID")
    rng = np.random.default_rng(seed)
    humans = {}
    for gid in test_ids:
        row = urn.loc[gid]
        h1, _h2 = draw_h1_h2(int(row["n_smooth"]), int(row["n_features"]), rng)
        humans[gid] = h1
    return humans


def _frozen_arrays(scores_df, label_df, test_ids):
    """Return (ai_label, y, n_smooth, n_features, score) as arrays in the frozen test-id order."""
    score = scores_df.set_index("GalaxyID")["score"]
    lab = label_df.set_index("GalaxyID")
    sc = np.array([float(score.loc[g]) for g in test_ids], dtype=float)
    y = np.array([int(lab.loc[g, "y_debiased"]) for g in test_ids], dtype=np.int64)
    ns = np.array([int(lab.loc[g, "n_smooth"]) for g in test_ids], dtype=np.int64)
    nf = np.array([int(lab.loc[g, "n_features"]) for g in test_ids], dtype=np.int64)
    return ai_label_from_score(sc), y, ns, nf, sc


def _build_rows(test_ids, y, ai, defer_by_b, humans_by_seed, budgets, seeds, policy):
    """Assemble PRED_COLUMNS rows for one policy across (budget x seed)."""
    rows = []
    for seed in seeds:
        h_arr = np.array([humans_by_seed[seed][g] for g in test_ids], dtype=np.int64)
        for b in budgets:
            defer = defer_by_b[b].astype(np.int64)
            decision = np.where(defer == 1, h_arr, ai)
            for k, gid in enumerate(test_ids):
                rows.append((
                    gid, int(y[k]), int(ai[k]), int(defer[k]), int(h_arr[k]),
                    int(decision[k]), int(defer[k]), float(b), int(seed), policy,
                ))
    return rows


# --------------------------------------------------------------------------- oracle sweep (Stage A)
def run_oracle_sweep(scores_df, label_df, test_ids, budgets, seeds) -> pd.DataFrame:
    """policy="oracle": find_machine_samples on the frozen TEST losses. Optimal upper bound.

    The deferral SET is deterministic (the gap uses the expected human loss), so it is identical
    across seeds; only the realized human draw on the deferred set varies with the seed.
    """
    test_ids = [int(g) for g in test_ids]
    ai, y, ns, nf, _sc = _frozen_arrays(scores_df, label_df, test_ids)
    mloss = (ai != y).astype(float)
    hloss = expected_human_loss(ns, nf, y)
    defer_by_b = {b: oracle_defer_mask(mloss, hloss, b) for b in budgets}
    humans_by_seed = {int(s): _draw_humans_per_seed(label_df, test_ids, int(s)) for s in seeds}
    rows = _build_rows(test_ids, y, ai, defer_by_b, humans_by_seed, budgets, seeds, "oracle")
    return pd.DataFrame(rows, columns=PRED_COLUMNS)


# --------------------------------------------------------------------------- learned rejector (Stage B)
def fit_rejector(emb, target, *, seed: int = 0, hidden: int = 64, epochs: int = 200,
                 lr: float = 1e-3, weight_decay: float = 0.0):
    """Train Okati's triage net (gnet) on backbone embeddings — an MLP with a LogSoftmax head + NLL.

    Faithful to train.ipynb Cell-18 (Linear->LogSoftmax, NLLLoss); the only change is the input
    (precomputed 2048-d embeddings of the FROZEN classifier, not raw images) — the Option-B
    adaptation. Class 1 = defer. CPU + seed_everything => deterministic.
    """
    import torch
    from torch import nn

    seed_everything(int(seed))
    X = torch.tensor(np.asarray(emb, dtype=np.float32))
    yv = torch.tensor(np.asarray(target, dtype=np.int64))
    d = X.shape[1]
    net = nn.Sequential(
        nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 2), nn.LogSoftmax(dim=-1)
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.NLLLoss()
    net.train()
    for _ in range(int(epochs)):
        opt.zero_grad()
        loss_fn(net(X), yv).backward()
        opt.step()
    net.eval()
    return net


def rejector_defer_proba(model, emb) -> np.ndarray:
    """P(defer) = exp(logsoftmax)[:, 1] for each row of `emb`."""
    import torch

    with torch.no_grad():
        logp = model(torch.tensor(np.asarray(emb, dtype=np.float32)))
        return torch.exp(logp[:, 1]).cpu().numpy()


def embedding_feature_cols(df: pd.DataFrame) -> list[str]:
    """Columns e0..e(N-1), numerically sorted."""
    cols = [c for c in df.columns if c.startswith(EMB_PREFIX) and c[len(EMB_PREFIX):].isdigit()]
    return sorted(cols, key=lambda c: int(c[len(EMB_PREFIX):]))


def validate_embeddings_frame(emb_df: pd.DataFrame, manifest: dict) -> None:
    """Guard the Colab embedding export contract before the learned arm trusts it.

    Requires: a `score` column, >=1 feature column, and exactly one row per GalaxyID for each of
    train/val/test in the committed split — loud (AssertionError) on any missing id or duplicate.
    """
    if "score" not in emb_df.columns:
        raise AssertionError("embeddings frame missing the `score` column")
    feats = embedding_feature_cols(emb_df)
    if not feats:
        raise AssertionError(f"embeddings frame has no `{EMB_PREFIX}<n>` feature columns")
    if emb_df["GalaxyID"].duplicated().any():
        raise AssertionError("embeddings frame has duplicate GalaxyIDs")
    have = set(emb_df["GalaxyID"])
    for split in ("train", "val", "test"):
        want = set(int(g) for g in manifest.get(split, []))
        missing = want - have
        if missing:
            raise AssertionError(
                f"embeddings frame missing {len(missing)} {split} ids (e.g. {list(missing)[:3]})"
            )


def run_learned_sweep(emb_df, label_df, train_ids, test_ids, budgets, seeds,
                      *, seed: int = 0, **fit_kw) -> pd.DataFrame:
    """policy="learned": train the rejector on TRAIN embeddings, apply label-free at TEST.

    Train target = the oracle defer set on TRAIN at the LARGEST budget (= the AI-error set the gap
    rewards deferring); the rejector learns to predict it from embeddings. At test, the budget is
    applied label-free by deferring the top floor(b*N) by P(defer) (`topk_defer_mask`), giving a
    smooth, deployable curve that must sit at/below the oracle envelope.
    """
    train_ids = [int(g) for g in train_ids]
    test_ids = [int(g) for g in test_ids]
    feats = embedding_feature_cols(emb_df)
    e = emb_df.set_index("GalaxyID")
    lab = label_df.set_index("GalaxyID")

    def arrays(ids):
        emb = e.loc[ids, feats].to_numpy(dtype=float)
        sc = e.loc[ids, "score"].to_numpy(dtype=float)
        y = lab.loc[ids, "y_debiased"].to_numpy()
        ns = lab.loc[ids, "n_smooth"].to_numpy()
        nf = lab.loc[ids, "n_features"].to_numpy()
        return emb, sc, y, ns, nf

    tr_emb, tr_sc, tr_y, tr_ns, tr_nf = arrays(train_ids)
    tr_mloss = (ai_label_from_score(tr_sc) != tr_y).astype(float)
    tr_hloss = expected_human_loss(tr_ns, tr_nf, tr_y)
    tr_target = oracle_defer_mask(tr_mloss, tr_hloss, max(budgets)).astype(int)
    model = fit_rejector(tr_emb, tr_target, seed=seed, **fit_kw)

    te_emb, te_sc, te_y, _te_ns, _te_nf = arrays(test_ids)
    ai = ai_label_from_score(te_sc)
    proba = rejector_defer_proba(model, te_emb)
    defer_by_b = {b: topk_defer_mask(proba, b) for b in budgets}
    humans_by_seed = {int(s): _draw_humans_per_seed(label_df, test_ids, int(s)) for s in seeds}
    rows = _build_rows(test_ids, te_y, ai, defer_by_b, humans_by_seed, budgets, seeds, "learned")
    return pd.DataFrame(rows, columns=PRED_COLUMNS)


# --------------------------------------------------------------------------- operating points
def operating_points(pred: pd.DataFrame) -> pd.DataFrame:
    """Collapse to (b, policy) operating points: accuracy mean+band over seeds, deferral cost.

    accuracy/cost are computed per (b, policy, seed) via the SHARED eval metrics, then aggregated.
    cost_mean == deferral_fraction by construction (human_queries == defer).
    """
    recs = []
    for (b, policy), g in pred.groupby(["b", "policy"]):
        accs, costs, defs = [], [], []
        for _seed, gs in g.groupby("seed"):
            accs.append(accuracy(gs["decision"].to_numpy(), gs["y_debiased"].to_numpy()))
            costs.append(expected_human_cost(gs["human_queries"].to_numpy()))
            defs.append(float(gs["defer"].mean()))
        accs, costs = np.array(accs), np.array(costs)
        recs.append({
            "b": float(b), "policy": policy,
            "accuracy_mean": accs.mean(), "accuracy_lo": accs.min(), "accuracy_hi": accs.max(),
            "cost_mean": costs.mean(), "deferral_fraction": float(np.mean(defs)),
            "n_seeds": len(accs),
        })
    return pd.DataFrame(recs, columns=OPS_COLUMNS).sort_values(["policy", "b"]).reset_index(drop=True)


# --------------------------------------------------------------------------- guards
def _assert_binary(pred: pd.DataFrame) -> None:
    for col in ["ai_label", "defer", "human", "decision", "y_debiased", "human_queries"]:
        bad = set(pd.unique(pred[col])) - {0, 1}
        if bad:
            raise AssertionError(f"{col} has non-binary values {bad}; expected 0/1 (1=spiral)")


def _assert_cost_unit(ops: pd.DataFrame) -> None:
    """Expected human cost (= deferral fraction) must lie in [0,1] — the L2D regime (≠ HCT's [1,2])."""
    for _i, r in ops.iterrows():
        if not (0.0 <= r["cost_mean"] <= 1.0):
            raise AssertionError(
                f"b={r['b']} policy={r['policy']}: expected cost {r['cost_mean']:.4f} outside [0,1]"
            )


def _assert_oracle_defers_only_errors(pred: pd.DataFrame) -> None:
    """Theorem-3 correctness: the ORACLE must never defer an instance the AI got right (gap<=0)."""
    o = pred[(pred["policy"] == "oracle") & (pred["defer"] == 1)]
    if len(o) and not (o["ai_label"] != o["y_debiased"]).all():
        raise AssertionError("oracle deferred an instance where the AI was correct (gap <= 0)")


def _assert_frozen_split(manifest: dict, test_ids) -> None:
    """Guard CLAUDE.md #2: the test ids we evaluate must match the committed split hash."""
    expect = manifest.get("split_hashes", {}).get("test")
    got = _sha256(list(map(int, test_ids)))
    if expect is not None and got != expect:
        raise AssertionError(
            f"frozen-split mismatch: recomputed test split_hash {got} != committed {expect}"
        )


# --------------------------------------------------------------------------- driver
def run(cfg: dict, eval_cfg: dict) -> dict:
    """Run the L2D-Okati arm end-to-end; write artifacts; return a terse summary (no printing)."""
    seed_everything(int(cfg.get("seed", 0)))

    manifest = json.loads(Path(cfg["split_manifest_path"]).read_text())
    test_ids = [int(g) for g in manifest["test"]]  # frozen order
    _assert_frozen_split(manifest, test_ids)

    scores_df = pd.read_parquet(cfg["scores_path"])
    label_df = pd.read_parquet(cfg["label_table_path"])
    missing = set(test_ids) - set(label_df["GalaxyID"])
    missing_scores = set(test_ids) - set(scores_df["GalaxyID"])
    if missing:
        raise AssertionError(f"{len(missing)} TEST ids missing urns in label_table (e.g. {list(missing)[:3]})")
    if missing_scores:
        raise AssertionError(f"{len(missing_scores)} TEST ids missing backbone scores (e.g. {list(missing_scores)[:3]})")

    budgets = list(cfg["budget_b_sweep"])
    seeds = list(eval_cfg["rater_draw_seeds"])

    pred = run_oracle_sweep(scores_df, label_df, test_ids, budgets, seeds)

    # Stage B (deployable): only if the Colab embedding export is present.
    emb_path = cfg.get("embeddings_path", "results/backbone_embeddings.parquet")
    learned_available = Path(emb_path).exists()
    if learned_available:
        emb_df = pd.read_parquet(emb_path)
        validate_embeddings_frame(emb_df, manifest)
        train_ids = [int(g) for g in manifest["train"]]
        learned = run_learned_sweep(emb_df, label_df, train_ids, test_ids, budgets, seeds,
                                    seed=int(cfg.get("seed", 0)))
        pred = pd.concat([pred, learned], ignore_index=True)

    _assert_binary(pred)
    _assert_oracle_defers_only_errors(pred)
    ops = operating_points(pred)
    _assert_cost_unit(ops)

    Path(cfg["export_predictions_path"]).parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(cfg["export_predictions_path"], index=False)
    Path(cfg["export_operating_points_path"]).parent.mkdir(parents=True, exist_ok=True)
    ops.to_csv(cfg["export_operating_points_path"], index=False)

    oracle = ops[ops["policy"] == "oracle"]
    ai_alone = float(oracle.loc[oracle["b"] == 0.0, "accuracy_mean"].iloc[0]) if (oracle["b"] == 0.0).any() else float("nan")
    best_oracle = float(oracle["accuracy_mean"].max())
    return {
        "n_test": len(test_ids),
        "n_budgets": len(budgets),
        "n_seeds": len(seeds),
        "n_rows": len(pred),
        "policies": sorted(pred["policy"].unique().tolist()),
        "learned_available": learned_available,
        "ai_alone_b0": ai_alone,
        "best_oracle_accuracy": best_oracle,
        "beats_ai_alone": bool(best_oracle > ai_alone + 1e-9),
        "max_deferral_fraction": float(oracle["deferral_fraction"].max()),
        "operating_points": ops.to_dict(orient="records"),
        "predictions_path": cfg["export_predictions_path"],
        "operating_points_path": cfg["export_operating_points_path"],
    }


def _wire_defaults(cfg: dict) -> dict:
    cfg.setdefault("split_manifest_path", "data/split_manifest.json")
    cfg.setdefault("label_table_path", "data/label_table.parquet")
    cfg.setdefault("scores_path", "results/backbone_scores.parquet")
    cfg.setdefault("embeddings_path", "results/backbone_embeddings.parquet")
    cfg.setdefault("export_predictions_path", "results/l2d_okati_predictions.parquet")
    cfg.setdefault("export_operating_points_path", "results/l2d_okati_operating_points.csv")
    cfg.setdefault("seed", 0)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="L2D-Okati arm — Theorem-3 triage over a frozen backbone.")
    ap.add_argument("--config", default="configs/arms.yaml")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    args = ap.parse_args()

    arms = yaml.safe_load(Path(args.config).read_text())
    eval_cfg = yaml.safe_load(Path(args.eval_config).read_text())
    cfg = _wire_defaults(arms["l2d_okati"])

    summary = run(cfg, eval_cfg)

    print(f"L2D-Okati arm: {summary['n_rows']} rows | policies={summary['policies']} "
          f"({summary['n_test']} test x {summary['n_budgets']} b x {summary['n_seeds']} seeds)")
    if not summary["learned_available"]:
        print("  [learned] SKIPPED — results/backbone_embeddings.parquet absent (Colab export pending)")
    print(f"  predictions -> {summary['predictions_path']}")
    print(f"  operating points -> {summary['operating_points_path']}")
    print(f"  AI-alone (b=0) = {summary['ai_alone_b0']:.4f} | best oracle acc = {summary['best_oracle_accuracy']:.4f} "
          f"| beats AI-alone: {summary['beats_ai_alone']} | max deferral = {summary['max_deferral_fraction']:.4f}")
    print("  (b, policy) -> (accuracy_mean [lo,hi], cost_mean=deferral)")
    for r in summary["operating_points"]:
        print(f"    b={r['b']:.2f} {r['policy']:<7} -> acc {r['accuracy_mean']:.4f} "
              f"[{r['accuracy_lo']:.4f},{r['accuracy_hi']:.4f}]  cost {r['cost_mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
