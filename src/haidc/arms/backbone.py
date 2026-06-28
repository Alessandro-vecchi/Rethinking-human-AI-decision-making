"""`make backbone` — the ONE shared AI classifier, reused by every arm (M2).

CLAUDE.md invariant #2: a single backbone + the frozen split feed AI-alone, HCT's AI judgment,
and both L2D classifiers. This module trains the Okati-faithful classifier and exports continuous
per-instance TEST scores P(spiral|x) to `results/backbone_scores.parquet` — the single score
source for all later arms.

Backbone recipe (Okati 2021, arXiv:2103.08902; vendored `train.ipynb` cell 10, GROUND_TRUTH §3/§4,
HANDOFF §3a): `torchvision.models.resnet50()` **scratch init, no ImageNet weights**; head
`nn.Sequential(nn.Linear(2048, 2), nn.LogSoftmax(dim=-1))`; `NLLLoss`; Adam (default lr 1e-3);
batch_size 128; 50 epochs; 3x224x224 input, ImageNet mean/std normalization. Target vs `y_debiased`
(DECISIONS 2026-06-26). Sanity anchor: AI-alone TEST accuracy ~0.83 (Okati Fig. 4(b), b=0).

The probe (DECISIONS 2026-06-26) found Okati ships **no** GalaxyID-aligned embeddings artifact —
`prepare_data.py` stores raw images and the ResNet is trained end-to-end in the notebook — so
`features: resnet_okati` (reuse) is unavailable and `features: fresh` (train) is the only path.
Training needs the Kaggle `images_training_rev1/` folder, which is git-ignored and must be present.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from haidc.eval.metrics import bootstrap_accuracy_ci
from haidc.seed import seed_everything

# P(spiral|x) = P(class 1). Class 0 = smooth/early-type, class 1 = features/disk = spiral, matching
# y_debiased (1 = spiral) and Okati's argmax(Class1.1, Class1.2) ordering (prepare_data.py:52,60).
SPIRAL_CLASS = 1
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- device
def resolve_device(spec="auto"):
    """Resolve a device spec to a `torch.device`. One place owns CPU-vs-CUDA selection.

    `"auto"` -> cuda when available else cpu; `"cuda"`/`"cpu"` honored explicitly. The same code
    runs CPU on this machine (CUDA=False) and CUDA on Colab — no source edits between the two.
    """
    import torch

    if spec in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


# --------------------------------------------------------------------------- model
def build_model():
    """Okati-faithful classifier: scratch resnet50 with a 2-way LogSoftmax head (train.ipynb cell 10)."""
    import torch.nn as nn
    from torchvision import models

    model = models.resnet50(weights=None)  # scratch init, NO ImageNet weights (HANDOFF §3a)
    model.fc = nn.Sequential(nn.Linear(2048, 2), nn.LogSoftmax(dim=-1))
    return model


def select_best_epoch(val_acc_curve) -> int:
    """Index of the best validation accuracy (earliest on ties) — the checkpoint to export.

    Earliest-on-ties (np.argmax semantics) prefers the cheaper, earlier-converged checkpoint and
    keeps selection deterministic across same-seed runs.
    """
    return int(np.argmax(np.asarray(val_acc_curve, dtype=float)))


def _eval_loss_acc(model, X, y, *, batch_size, channels_last, threshold=0.5):
    """Mean NLLLoss and AI-alone accuracy of `model` on (X, y) — eval mode, no grad, one pass.

    Device-agnostic (follows the model's own device, like `predict_spiral_proba`). Used to record
    the per-epoch train/val curves without a second forward pass per metric.
    """
    import torch

    model = model.eval()
    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(y), dtype=torch.long)
    if use_cuda:
        X = X.pin_memory()
    loss_func = torch.nn.NLLLoss(reduction="sum")
    total_loss, correct, n = 0.0, 0, int(X.shape[0])
    with torch.no_grad():
        for b in range(0, n, batch_size):
            xb = X[b : b + batch_size].to(device, non_blocking=use_cuda)
            yb = y[b : b + batch_size].to(device, non_blocking=use_cuda)
            if use_cuda and channels_last:
                xb = xb.to(memory_format=torch.channels_last)
            logp = model(xb)
            total_loss += float(loss_func(logp, yb))
            pred = (torch.exp(logp)[:, SPIRAL_CLASS] >= threshold).long()
            correct += int((pred == yb).sum())
    denom = max(1, n)
    return total_loss / denom, correct / denom


def train_model(
    model, X, y, *, epochs, batch_size, lr, weight_decay=0.0,
    device=None, channels_last=False, return_losses=False,
    X_val=None, y_val=None, select_best_val=False, return_history=False, threshold=0.5,
):
    """Train with NLLLoss + Adam on (X, y). Returns the model, per-epoch losses, or a history dict.

    X: float array/tensor (N, 3, 224, 224); y: int array/tensor (N,) in {0, 1}. Okati uses Adam
    with its default lr; `lr`/`epochs`/`batch_size` are config-driven (CODING.md "config over constants").

    Validation tracking (DECISIONS 2026-06-27, final dispositive run): when `X_val`/`y_val` are given
    (or `return_history`/`select_best_val` set), per-epoch train and val accuracy + loss curves are
    recorded, and the best-VAL-accuracy weights are snapshotted. `select_best_val=True` loads those
    weights back before returning, so the exported checkpoint is the best-VAL one, not the last epoch.
    Return precedence: `return_history` -> a history dict {train_loss, train_acc, val_loss, val_acc,
    best_epoch (1-indexed)}; else `return_losses` -> the per-epoch train-loss list; else the model.

    GPU path (FP32, deterministic-faithful — DECISIONS 2026-06-27): on CUDA the host tensor is
    pinned and batches are moved with `non_blocking=True`; `channels_last` lays conv activations out
    for the GPU. ResNet-50's backward includes ops with no deterministic CUDA implementation
    (e.g. `adaptive_avg_pool2d_backward_cuda`), which would raise under the strict
    `use_deterministic_algorithms(True)` set by `seed_everything`; on CUDA we relax to
    `warn_only=True` (documented best-effort GPU determinism). The CPU path is numerically
    unchanged from M2's original (defaults resolve to cpu, no pinning, no warn_only relaxation).
    """
    import copy

    import torch

    device = resolve_device(device)
    use_cuda = device.type == "cuda"
    if use_cuda:
        # ResNet-50 backward hits nondeterministic CUDA ops; warn instead of raising (DECISIONS).
        torch.use_deterministic_algorithms(True, warn_only=True)
    model = model.to(device)
    if use_cuda and channels_last:
        model = model.to(memory_format=torch.channels_last)
    X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    y_t = torch.as_tensor(np.asarray(y), dtype=torch.long)
    if use_cuda:
        X_t = X_t.pin_memory()  # page-locked host memory -> overlap H2D copy with compute
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = torch.nn.NLLLoss()
    n = X_t.shape[0]
    n_batches = max(1, n // batch_size)

    track = return_history or select_best_val or (X_val is not None)
    train_loss_curve, train_acc_curve, val_loss_curve, val_acc_curve = [], [], [], []
    best_state, best_epoch, best_val = None, 0, -1.0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        batch_losses = []
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            xb = X_t[idx].to(device, non_blocking=use_cuda)
            yb = y_t[idx].to(device, non_blocking=use_cuda)
            if use_cuda and channels_last:
                xb = xb.to(memory_format=torch.channels_last)
            opt.zero_grad()
            loss = loss_func(model(xb), yb)
            loss.backward()
            opt.step()
            batch_losses.append(float(loss.detach()))
        train_loss_curve.append(float(np.mean(batch_losses)))

        if track:
            _, tr_acc = _eval_loss_acc(
                model, X, y, batch_size=batch_size, channels_last=channels_last, threshold=threshold
            )
            train_acc_curve.append(tr_acc)
            if X_val is not None:
                v_loss, v_acc = _eval_loss_acc(
                    model, X_val, y_val, batch_size=batch_size,
                    channels_last=channels_last, threshold=threshold,
                )
                val_loss_curve.append(v_loss)
                val_acc_curve.append(v_acc)
                if v_acc > best_val:  # strict '>' -> earliest argmax, matches select_best_epoch
                    best_val, best_epoch = v_acc, epoch
                    best_state = copy.deepcopy(
                        {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    )

    if select_best_val and best_state is not None:
        model.load_state_dict(best_state)  # export the best-VAL checkpoint, not the last epoch

    if return_history:
        return {
            "train_loss": train_loss_curve,
            "train_acc": train_acc_curve,
            "val_loss": val_loss_curve,
            "val_acc": val_acc_curve,
            "best_epoch": (best_epoch + 1) if val_acc_curve else None,
        }
    if return_losses:
        return train_loss_curve
    return model


def predict_spiral_proba(model, X, *, batch_size=128, channels_last=False):
    """Continuous, pre-threshold P(spiral|x) in [0,1] for every row of X (eval mode, no grad).

    Device-agnostic: batches follow the model's own device (so a CUDA-resident model scores on the
    GPU without the caller juggling `.cpu()`), and results are pulled back to a numpy float64 array.
    """
    import torch

    model = model.eval()
    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    if use_cuda:
        X = X.pin_memory()
    out = []
    with torch.no_grad():
        for b in range(0, X.shape[0], batch_size):
            xb = X[b : b + batch_size].to(device, non_blocking=use_cuda)
            if use_cuda and channels_last:
                xb = xb.to(memory_format=torch.channels_last)
            logp = model(xb)                                 # LogSoftmax outputs
            out.append(torch.exp(logp)[:, SPIRAL_CLASS].cpu().numpy())
    return np.concatenate(out).astype(np.float64)


# --------------------------------------------------------------------------- IO
def load_split(manifest_path) -> dict:
    """Frozen 70/15/15 split lists (M1). The order of `test` is the export order for all arms."""
    m = json.loads(Path(manifest_path).read_text())
    return {k: [int(g) for g in m[k]] for k in ("train", "val", "test")}


def load_labels(label_table_path) -> dict:
    """GalaxyID -> y_debiased (DECISIONS 2026-06-26 default target for all arms)."""
    df = pd.read_parquet(label_table_path, columns=["GalaxyID", "y_debiased"])
    return dict(zip(df["GalaxyID"].astype(int), df["y_debiased"].astype(int)))


def load_images_for_ids(ids, images_dir):
    """Load + preprocess Kaggle images for `ids` as a (N,3,224,224) tensor (Okati prepare_data.py).

    Fails loud (CODING.md) if the image folder or any GalaxyID's `<id>.jpg` is absent — a missing
    image is a confound, not something to silently skip.
    """
    import torch
    from PIL import Image
    from torchvision import transforms

    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(
            f"image dir {images_dir} not found — download the Kaggle "
            "'galaxy-zoo-the-galaxy-challenge' images_training_rev1 set (Okati fn 9)"
        )
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    tensors = []
    for g in ids:
        path = images_dir / f"{g}.jpg"
        if not path.exists():
            raise FileNotFoundError(f"missing image for GalaxyID {g}: {path}")
        tensors.append(preprocess(Image.open(path).convert("RGB")))
    return torch.stack(tensors)


def export_scores(galaxy_ids, scores) -> pd.DataFrame:
    """One row per GalaxyID, in input order, with continuous P(spiral|x). The cross-arm interface."""
    galaxy_ids = [int(g) for g in galaxy_ids]
    scores = np.asarray(scores, dtype=np.float64)
    if len(galaxy_ids) != len(scores):
        raise ValueError(f"ids ({len(galaxy_ids)}) and scores ({len(scores)}) length mismatch")
    if scores.size and (scores.min() < 0.0 or scores.max() > 1.0):
        raise ValueError("scores must be probabilities in [0,1]")
    return pd.DataFrame({"GalaxyID": np.asarray(galaxy_ids, dtype=np.int64), "score": scores})


# --------------------------------------------------------------------------- embeddings export (Stage B infra)
def embed_and_score(model, X, *, batch_size=128, channels_last=False):
    """Penultimate 2048-d features (avgpool output) AND P(spiral|x) in one forward pass.

    The 2048-d vector is the input to the `fc` head — captured via a forward hook on `avgpool`,
    flattened. Device-agnostic, same batching as `predict_spiral_proba`. Returns (emb (N,2048),
    score (N,)) as numpy float64. This is the deferral-rejector input for the L2D arms (Option B):
    the frozen classifier's representation, not raw images (Okati train.ipynb gnet adapted).
    """
    import torch

    model = model.eval()
    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"
    captured = {}
    handle = model.avgpool.register_forward_hook(
        lambda _m, _inp, out: captured.__setitem__("z", torch.flatten(out, 1).detach())
    )
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    if use_cuda:
        X = X.pin_memory()
    embs, scores = [], []
    try:
        with torch.no_grad():
            for b in range(0, X.shape[0], batch_size):
                xb = X[b : b + batch_size].to(device, non_blocking=use_cuda)
                if use_cuda and channels_last:
                    xb = xb.to(memory_format=torch.channels_last)
                logp = model(xb)
                embs.append(captured["z"].cpu().numpy())
                scores.append(torch.exp(logp)[:, SPIRAL_CLASS].cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(embs).astype(np.float64), np.concatenate(scores).astype(np.float64)


def build_embeddings_frame(ids_by_split, emb_by_split, score_by_split) -> pd.DataFrame:
    """Assemble the cross-arm embeddings contract consumed by L2D-Okati(learned) and M5 Mozannar.

    One row per GalaxyID; splits stacked train -> val -> test, each in the given (manifest) id order.
    Columns: GalaxyID, split, score, e0..e(D-1). Pure (no torch) so the contract is unit-testable.
    """
    frames = []
    for split in ("train", "val", "test"):
        ids = [int(g) for g in ids_by_split[split]]
        emb = np.asarray(emb_by_split[split], dtype=np.float64)
        sc = np.asarray(score_by_split[split], dtype=np.float64)
        if not (len(ids) == emb.shape[0] == len(sc)):
            raise ValueError(
                f"{split}: ids/emb/score length mismatch {len(ids)}/{emb.shape[0]}/{len(sc)}"
            )
        cols = {"GalaxyID": np.asarray(ids, dtype=np.int64), "split": split, "score": sc}
        for j in range(emb.shape[1]):
            cols[f"e{j}"] = emb[:, j]
        frames.append(pd.DataFrame(cols))
    return pd.concat(frames, ignore_index=True)


def export_embeddings(cfg: dict) -> dict:
    """Load the FROZEN backbone.pt and export 2048-d features + score for all three splits.

    Shared Stage-B infra (a Colab GPU pass; not run on the CPU box). Writes
    `cfg['export_embeddings_path']` (default results/backbone_embeddings.parquet). The classifier is
    NOT retrained — this only reads off the frozen checkpoint, preserving CLAUDE.md invariant #2.
    """
    import torch

    seed_everything(int(cfg["seed"]))
    split = load_split(cfg["manifest_path"])
    tr = cfg["train"]
    device = resolve_device(tr.get("device", "auto"))
    channels_last = bool(tr.get("channels_last", True))

    model = build_model()
    model.load_state_dict(torch.load(cfg["artifact_path"], map_location="cpu"))
    model.to(device)
    if device.type == "cuda" and channels_last:
        model = model.to(memory_format=torch.channels_last)

    emb_by_split, score_by_split = {}, {}
    for s in ("train", "val", "test"):
        X = load_images_for_ids(split[s], cfg["images_dir"])
        e, sc = embed_and_score(model, X, batch_size=int(tr["batch_size"]), channels_last=channels_last)
        emb_by_split[s], score_by_split[s] = e, sc

    frame = build_embeddings_frame(split, emb_by_split, score_by_split)
    out = cfg.get("export_embeddings_path", "results/backbone_embeddings.parquet")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    return {
        "path": out, "n_rows": len(frame), "n_features": frame.shape[1] - 3,
        "n_train": len(split["train"]), "n_val": len(split["val"]), "n_test": len(split["test"]),
        "device": device.type,
    }


# --------------------------------------------------------------------------- metrics
def ai_alone_accuracy(scores, y_true, *, threshold=0.5) -> float:
    """AI-alone accuracy: predict spiral when P(spiral) >= threshold, compare to y_true."""
    pred = (np.asarray(scores) >= threshold).astype(int)
    return float((pred == np.asarray(y_true)).mean())


def bootstrap_ci(scores, y_true, *, threshold, n_boot, ci, seed):
    """Percentile bootstrap CI for AI-alone accuracy over the test set (REPRODUCIBILITY.md).

    Thin wrapper kept for this module's call site + tests; the estimator now lives in the shared
    eval module (M6 absorbed the inline copy, invariant §4). Building the per-instance correctness
    vector first and delegating is byte-for-byte identical to the old inline loop.
    """
    pred = (np.asarray(scores) >= threshold).astype(int)
    correct = (pred == np.asarray(y_true)).astype(int)
    return bootstrap_accuracy_ci(correct, n_resamples=n_boot, ci=ci, seed=seed)


# --------------------------------------------------------------------------- entry point
def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _vendored_sha(path="third_party/okati2021") -> str:
    # third_party/** is git-ignored and absent in the Colab clone, so `git -C` would print a noisy
    # "fatal: cannot change to ..." to stderr before we catch it; silence stderr (DECISIONS 2026-06-27).
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def run(cfg: dict) -> dict:
    """Train the shared backbone, export TEST scores, return the run manifest (no printing)."""
    import torch

    seed = int(cfg["seed"])
    seed_everything(seed)

    if cfg["features"] != "fresh":
        raise NotImplementedError(
            f"features={cfg['features']!r} unavailable: Okati ships no GalaxyID-aligned embeddings "
            "(DECISIONS 2026-06-26 probe); only features='fresh' (train on images) is supported"
        )

    split = load_split(cfg["manifest_path"])
    labels = load_labels(cfg["label_table_path"])
    missing = [g for g in split["train"] + split["val"] + split["test"] if g not in labels]
    if missing:
        raise RuntimeError(f"{len(missing)} split GalaxyIDs lack a y_debiased label (join confound)")

    tr = cfg["train"]
    thr = float(cfg["threshold_default"])
    device = resolve_device(tr.get("device", "auto"))
    channels_last = bool(tr.get("channels_last", True))
    X_train = load_images_for_ids(split["train"], cfg["images_dir"])
    y_train = np.array([labels[g] for g in split["train"]], dtype="int64")
    X_val = load_images_for_ids(split["val"], cfg["images_dir"])
    y_val = np.array([labels[g] for g in split["val"]], dtype="int64")

    model = build_model()
    # Final dispositive run (DECISIONS 2026-06-27): track per-epoch train/val curves and export the
    # best-VAL-accuracy checkpoint (not the last epoch). The model is trained in place and left
    # holding the best-VAL weights, so we reuse it for the train/test scoring below.
    history = train_model(
        model, X_train, y_train,
        epochs=int(tr["epochs"]), batch_size=int(tr["batch_size"]),
        lr=float(tr["lr"]), weight_decay=float(tr.get("weight_decay", 0.0)),
        device=device, channels_last=channels_last,
        X_val=X_val, y_val=y_val, select_best_val=True, return_history=True, threshold=thr,
    )
    train_loss_curve = history["train_loss"]
    best_epoch = history["best_epoch"]
    val_accuracy = history["val_acc"][best_epoch - 1] if best_epoch else None

    # Convergence diagnostic (DECISIONS 2026-06-27): read the train-acc curve — a climb to ~0.93+
    # means it was under-trained; a plateau ~0.80-0.85 means a label/consensus ceiling, not an
    # optimization bug. `train_scores` reflect the exported best-VAL checkpoint.
    train_scores = predict_spiral_proba(
        model, X_train, batch_size=int(tr["batch_size"]), channels_last=channels_last
    )

    X_test = load_images_for_ids(split["test"], cfg["images_dir"])
    y_test = np.array([labels[g] for g in split["test"]], dtype="int64")
    scores = predict_spiral_proba(
        model, X_test, batch_size=int(tr["batch_size"]), channels_last=channels_last
    )

    df = export_scores(split["test"], scores)
    Path(cfg["export_scores_path"]).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg["export_scores_path"], index=False)

    acc = ai_alone_accuracy(scores, y_test, threshold=thr)
    train_acc = ai_alone_accuracy(train_scores, y_train, threshold=thr)
    bs = cfg["bootstrap"]
    lo, hi = bootstrap_ci(
        scores, y_test, threshold=thr,
        n_boot=int(bs["n_boot"]), ci=float(bs["ci"]), seed=seed,
    )

    Path(cfg["artifact_path"]).parent.mkdir(parents=True, exist_ok=True)
    model.to("cpu")  # portable artifact: a CPU-saved state_dict loads on any device
    torch.save(model.state_dict(), cfg["artifact_path"])

    manifest = {
        "stage": "backbone",
        "config": cfg,
        "seed": seed,
        "git_sha": _git_sha(),
        "vendored_okati_sha": _vendored_sha(),
        "device": device.type,
        "channels_last": channels_last,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "n_train": len(split["train"]),
        "n_val": len(split["val"]),
        "n_test": len(split["test"]),
        "threshold": thr,
        "ai_alone_accuracy": acc,
        "ai_alone_accuracy_ci": [lo, hi],
        "ci_level": float(bs["ci"]),
        "n_boot": int(bs["n_boot"]),
        "train_accuracy": train_acc,
        "val_accuracy": val_accuracy,
        "best_epoch": best_epoch,
        "final_train_loss": float(train_loss_curve[-1]) if train_loss_curve else None,
        "train_loss_curve": [float(v) for v in train_loss_curve],
        "train_acc_curve": [float(v) for v in history["train_acc"]],
        "val_acc_curve": [float(v) for v in history["val_acc"]],
        "val_loss_curve": [float(v) for v in history["val_loss"]],
        "artifact_path": cfg["artifact_path"],
        "export_scores_path": cfg["export_scores_path"],
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    Path(cfg["run_manifest_path"]).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/backbone.yaml")
    ap.add_argument("--export-embeddings", action="store_true",
                    help="skip training; export 2048-d features + score from the frozen backbone.pt "
                         "(Stage-B infra for the L2D arms; run on Colab GPU)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.export_embeddings:
        e = export_embeddings(cfg)
        print(f"embeddings: {e['n_rows']} rows x {e['n_features']} feats "
              f"(train {e['n_train']}, val {e['n_val']}, test {e['n_test']}) on {e['device']} -> {e['path']}")
        return 0
    m = run(cfg)
    lo, hi = m["ai_alone_accuracy_ci"]
    gap = m["train_accuracy"] - m["ai_alone_accuracy"]
    tac = m["train_acc_curve"]
    print(
        f"backbone: trained on {m['n_train']} imgs, val {m['n_val']}, scored {m['n_test']} test imgs\n"
        f"AI-alone test accuracy @ thr {m['threshold']} = {m['ai_alone_accuracy']:.4f} "
        f"(95% CI [{lo:.4f}, {hi:.4f}])\n"
        f"best-VAL epoch = {m['best_epoch']}/{len(tac)} | val acc = {m['val_accuracy']:.4f}\n"
        f"train accuracy = {m['train_accuracy']:.4f} | final train loss = {m['final_train_loss']:.4f} "
        f"| train-test gap = {gap:+.4f}\n"
        f"train-acc curve: first={tac[0]:.3f} max={max(tac):.3f} last={tac[-1]:.3f}  "
        f"(verdict: climb to ~0.93+ => under-trained; plateau ~0.80-0.85 => label ceiling)\n"
        f"scores -> {m['export_scores_path']}   artifact -> {m['artifact_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
