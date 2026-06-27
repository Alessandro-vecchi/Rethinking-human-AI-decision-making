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


def train_model(
    model, X, y, *, epochs, batch_size, lr, weight_decay=0.0,
    device=None, channels_last=False, return_losses=False,
):
    """Train with NLLLoss + Adam on (X, y). Returns the model, or per-epoch mean losses if asked.

    X: float array/tensor (N, 3, 224, 224); y: int array/tensor (N,) in {0, 1}. Okati uses Adam
    with its default lr; `lr`/`epochs`/`batch_size` are config-driven (CODING.md "config over constants").

    GPU path (FP32, deterministic-faithful — DECISIONS 2026-06-27): on CUDA the host tensor is
    pinned and batches are moved with `non_blocking=True`; `channels_last` lays conv activations out
    for the GPU. ResNet-50's backward includes ops with no deterministic CUDA implementation
    (e.g. `adaptive_avg_pool2d_backward_cuda`), which would raise under the strict
    `use_deterministic_algorithms(True)` set by `seed_everything`; on CUDA we relax to
    `warn_only=True` (documented best-effort GPU determinism). The CPU path is numerically
    unchanged from M2's original (defaults resolve to cpu, no pinning, no warn_only relaxation).
    """
    import torch

    device = resolve_device(device)
    use_cuda = device.type == "cuda"
    if use_cuda:
        # ResNet-50 backward hits nondeterministic CUDA ops; warn instead of raising (DECISIONS).
        torch.use_deterministic_algorithms(True, warn_only=True)
    model = model.to(device).train()
    if use_cuda and channels_last:
        model = model.to(memory_format=torch.channels_last)
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(y), dtype=torch.long)
    if use_cuda:
        X = X.pin_memory()  # page-locked host memory -> overlap H2D copy with compute
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = torch.nn.NLLLoss()
    n = X.shape[0]
    n_batches = max(1, n // batch_size)
    epoch_losses = []
    for _ in range(epochs):
        perm = torch.randperm(n)
        batch_losses = []
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            xb = X[idx].to(device, non_blocking=use_cuda)
            yb = y[idx].to(device, non_blocking=use_cuda)
            if use_cuda and channels_last:
                xb = xb.to(memory_format=torch.channels_last)
            opt.zero_grad()
            loss = loss_func(model(xb), yb)
            loss.backward()
            opt.step()
            batch_losses.append(float(loss.detach()))
        epoch_losses.append(float(np.mean(batch_losses)))
    return epoch_losses if return_losses else model


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


# --------------------------------------------------------------------------- metrics
def ai_alone_accuracy(scores, y_true, *, threshold=0.5) -> float:
    """AI-alone accuracy: predict spiral when P(spiral) >= threshold, compare to y_true."""
    pred = (np.asarray(scores) >= threshold).astype(int)
    return float((pred == np.asarray(y_true)).mean())


def bootstrap_ci(scores, y_true, *, threshold, n_boot, ci, seed):
    """Percentile bootstrap CI for AI-alone accuracy over the test set (REPRODUCIBILITY.md).

    Inline until the shared eval module exists (M6 to absorb; M2-PLAN §2). Resamples test instances
    with replacement; deterministic given `seed`.
    """
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    n = len(scores)
    rng = np.random.RandomState(seed)
    accs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        accs[i] = ai_alone_accuracy(scores[idx], y_true[idx], threshold=threshold)
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(accs, alpha)), float(np.quantile(accs, 1.0 - alpha))


# --------------------------------------------------------------------------- entry point
def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _vendored_sha(path="third_party/okati2021") -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True
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
    missing = [g for g in split["train"] + split["test"] if g not in labels]
    if missing:
        raise RuntimeError(f"{len(missing)} split GalaxyIDs lack a y_debiased label (join confound)")

    tr = cfg["train"]
    device = resolve_device(tr.get("device", "auto"))
    channels_last = bool(tr.get("channels_last", True))
    X_train = load_images_for_ids(split["train"], cfg["images_dir"])
    y_train = np.array([labels[g] for g in split["train"]], dtype="int64")

    model = build_model()
    train_model(
        model, X_train, y_train,
        epochs=int(tr["epochs"]), batch_size=int(tr["batch_size"]),
        lr=float(tr["lr"]), weight_decay=float(tr.get("weight_decay", 0.0)),
        device=device, channels_last=channels_last,
    )

    X_test = load_images_for_ids(split["test"], cfg["images_dir"])
    y_test = np.array([labels[g] for g in split["test"]], dtype="int64")
    scores = predict_spiral_proba(
        model, X_test, batch_size=int(tr["batch_size"]), channels_last=channels_last
    )

    df = export_scores(split["test"], scores)
    Path(cfg["export_scores_path"]).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg["export_scores_path"], index=False)

    thr = float(cfg["threshold_default"])
    acc = ai_alone_accuracy(scores, y_test, threshold=thr)
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
        "n_test": len(split["test"]),
        "threshold": thr,
        "ai_alone_accuracy": acc,
        "ai_alone_accuracy_ci": [lo, hi],
        "ci_level": float(bs["ci"]),
        "n_boot": int(bs["n_boot"]),
        "artifact_path": cfg["artifact_path"],
        "export_scores_path": cfg["export_scores_path"],
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    Path(cfg["run_manifest_path"]).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/backbone.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    m = run(cfg)
    lo, hi = m["ai_alone_accuracy_ci"]
    print(
        f"backbone: trained on {m['n_train']} imgs, scored {m['n_test']} test imgs\n"
        f"AI-alone test accuracy @ thr {m['threshold']} = {m['ai_alone_accuracy']:.4f} "
        f"(95% CI [{lo:.4f}, {hi:.4f}])\n"
        f"scores -> {m['export_scores_path']}   artifact -> {m['artifact_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
