"""Deterministic frozen 70/15/15 split + manifest hashing (M1).

The split is generated once from seed=0, hashed, and committed as data/split_manifest.json;
`make verify-split` recomputes it and asserts the hash. One frozen split feeds every arm
(CLAUDE.md invariant #2). Pure functions only — file IO lives in build_split / verify_split.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

import numpy as np


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha256(obj) -> str:
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


def compute_split(
    galaxy_ids: Sequence[int], seed: int, ratios: Mapping[str, float]
) -> dict[str, list[int]]:
    """Shuffle the sorted GalaxyID list under ``seed`` and slice train/val/test.

    Sizes: train=floor(r_train*N), val=floor(r_val*N), test=N-train-val (remainder to test).
    Sorting first makes the result independent of input ordering; the permutation is the only
    randomness and it is fully determined by ``seed``.
    """
    ids_sorted = sorted(int(g) for g in galaxy_ids)
    n = len(ids_sorted)
    perm = np.random.default_rng(seed).permutation(n)
    shuffled = [ids_sorted[k] for k in perm]
    n_train = math.floor(ratios["train"] * n)
    n_val = math.floor(ratios["val"] * n)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def split_hashes(split: Mapping[str, Sequence[int]]) -> dict[str, str]:
    return {k: _sha256(list(map(int, v))) for k, v in split.items()}


def build_manifest(
    split: Mapping[str, Sequence[int]],
    seed: int,
    ratios: Mapping[str, float],
    extra: Mapping[str, object],
) -> dict:
    """Assemble the full committed manifest, including per-split and top-level content hashes."""
    split = {k: list(map(int, v)) for k, v in split.items()}
    final_n = sum(len(v) for v in split.values())
    core = {"seed": seed, "split": dict(ratios), "final_N": final_n, "splits": split}
    manifest = {
        "final_N": final_n,
        "seed": seed,
        "split": dict(ratios),
        "train": split["train"],
        "val": split["val"],
        "test": split["test"],
        "split_hashes": split_hashes(split),
        "content_hash": _sha256(core),
    }
    manifest.update(dict(extra))
    return manifest
