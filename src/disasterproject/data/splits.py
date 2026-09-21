"""
Frozen train / calibration / eval split helpers.

UID scheme matches create_frozen_eval_ids.py: sha1("<message>|<row_index>").
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from disasterproject.utils.config import CRITICAL_LABELS, TARGET_COLUMNS


def compute_uids(messages: Iterable) -> list[str]:
    """Compute stable UIDs from messages and their positional index."""
    uids: list[str] = []
    for idx, msg in enumerate(messages):
        text = "" if msg is None else str(msg)
        uid_src = f"{text}|{idx}"
        uids.append(hashlib.sha1(uid_src.encode("utf-8")).hexdigest())
    return uids


def load_uid_list(ids_path: str | Path, key: str) -> list[str]:
    """Load a UID list from JSON ({key: [...]}) or legacy CSV (uid column)."""
    path = Path(ids_path)
    if not path.is_file():
        raise FileNotFoundError(f"UID file not found: {path}")

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if key not in data:
            raise KeyError(f"Expected key {key!r} in {path}")
        return [str(uid) for uid in data[key]]

    frame = pd.read_csv(path)
    if "uid" not in frame.columns:
        raise KeyError(f"Expected 'uid' column in {path}")
    return frame["uid"].astype(str).tolist()


def positive_count_strata(Y: np.ndarray, max_count: int = 3) -> np.ndarray:
    """Stratification proxy used by frozen eval/cal creators."""
    pos_counts = Y.sum(axis=1)
    if isinstance(pos_counts, list):
        pos_counts = np.array(pos_counts)
    return np.clip(pos_counts, a_min=None, a_max=max_count)


def critical_label_positive_support(
    Y: np.ndarray,
    label_names: Sequence[str] | None = None,
    critical_labels: Iterable[str] | None = None,
) -> dict[str, int]:
    """Count positives per critical label in a label matrix."""
    names = list(label_names) if label_names is not None else list(TARGET_COLUMNS)
    critical = set(critical_labels) if critical_labels is not None else set(CRITICAL_LABELS)
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    support: dict[str, int] = {}
    for label in sorted(critical):
        idx = name_to_idx.get(label)
        if idx is None:
            support[label] = 0
        else:
            support[label] = int(np.sum(Y[:, idx]))
    return support


def assert_critical_support(
    support: Mapping[str, int],
    *,
    context: str = "calibration",
) -> None:
    """Raise ValueError if any critical label has zero positives."""
    zero = [label for label, count in sorted(support.items()) if count <= 0]
    if zero:
        raise ValueError(
            f"Zero positive support in {context} for critical labels: {', '.join(zero)}"
        )


def three_way_masks(
    uids: Sequence[str],
    eval_uids: Iterable[str],
    cal_uids: Iterable[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build boolean masks for train / cal / eval.

    Raises ValueError if cal and eval overlap or if any UID is unassigned.
    """
    uid_series = pd.Series(list(uids))
    eval_set = set(eval_uids)
    cal_set = set(cal_uids)

    overlap = eval_set & cal_set
    if overlap:
        raise ValueError(
            f"cal_ids and eval_ids overlap ({len(overlap)} UIDs); refusing split"
        )

    is_eval = uid_series.isin(eval_set).to_numpy()
    is_cal = uid_series.isin(cal_set).to_numpy()
    is_train = ~(is_eval | is_cal)

    if not bool(np.any(is_train)):
        raise ValueError("Train residual is empty after excluding cal ∪ eval")

    assigned = int(is_train.sum() + is_cal.sum() + is_eval.sum())
    if assigned != len(uids):
        raise ValueError(
            f"Incomplete assignment: assigned={assigned}, total={len(uids)}"
        )

    return is_train, is_cal, is_eval


def apply_masks(
    X,
    Y: np.ndarray,
    is_train: np.ndarray,
    is_cal: np.ndarray,
    is_eval: np.ndarray,
) -> dict[str, tuple]:
    """Slice X/Y into train, cal, and eval arrays."""
    return {
        "train": (X[is_train], Y[is_train]),
        "cal": (X[is_cal], Y[is_cal]),
        "eval": (X[is_eval], Y[is_eval]),
    }


def load_three_way_split(
    X,
    Y: np.ndarray,
    eval_ids_path: str | Path,
    cal_ids_path: str | Path,
) -> dict[str, tuple]:
    """
    Load frozen eval + cal UID files and return train/cal/eval slices.

    Returns dict with keys train, cal, eval mapping to (X_split, Y_split).
    """
    uids = compute_uids(X)
    eval_uids = load_uid_list(eval_ids_path, "eval_ids")
    cal_uids = load_uid_list(cal_ids_path, "cal_ids")
    is_train, is_cal, is_eval = three_way_masks(uids, eval_uids, cal_uids)
    return apply_masks(X, Y, is_train, is_cal, is_eval)


def assert_partition_invariants(
    all_uids: Sequence[str],
    train_uids: Iterable[str],
    cal_uids: Iterable[str],
    eval_uids: Iterable[str],
) -> None:
    """Assert pairwise disjointness and complete assignment over all_uids."""
    train_set = set(train_uids)
    cal_set = set(cal_uids)
    eval_set = set(eval_uids)
    all_set = set(all_uids)

    if train_set & cal_set:
        raise AssertionError("train ∩ cal is non-empty")
    if train_set & eval_set:
        raise AssertionError("train ∩ eval is non-empty")
    if cal_set & eval_set:
        raise AssertionError("cal ∩ eval is non-empty")
    if train_set | cal_set | eval_set != all_set:
        raise AssertionError("train ∪ cal ∪ eval does not equal all dataset UIDs")
