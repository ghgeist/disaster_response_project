"""Tests for train/cal/eval split invariants and threshold tune-vs-score separation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from disasterproject.data.splits import (
    assert_critical_support,
    assert_partition_invariants,
    compute_uids,
    critical_label_positive_support,
    load_three_way_split,
    three_way_masks,
)
from disasterproject.utils.config import CRITICAL_LABELS, TARGET_COLUMNS


def _write_uid_json(path: Path, key: str, uids: list[str], extra_meta: dict | None = None) -> None:
    payload = {"metadata": extra_meta or {}, key: uids}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compute_uids_stable():
    messages = ["help", "water needed", None]
    assert compute_uids(messages) == compute_uids(messages)
    assert len(set(compute_uids(messages))) == 3


def test_three_way_masks_partition_and_overlap():
    uids = ["a", "b", "c", "d", "e"]
    eval_uids = ["a", "b"]
    cal_uids = ["c"]
    is_train, is_cal, is_eval = three_way_masks(uids, eval_uids, cal_uids)
    assert int(is_train.sum()) == 2
    assert int(is_cal.sum()) == 1
    assert int(is_eval.sum()) == 2
    assert not np.any(is_train & is_cal)
    assert not np.any(is_train & is_eval)
    assert not np.any(is_cal & is_eval)

    with pytest.raises(ValueError, match="overlap"):
        three_way_masks(uids, ["a"], ["a", "b"])


def test_assert_partition_invariants():
    all_uids = ["a", "b", "c", "d"]
    assert_partition_invariants(all_uids, ["a", "b"], ["c"], ["d"])
    with pytest.raises(AssertionError):
        assert_partition_invariants(all_uids, ["a"], ["a"], ["d"])
    with pytest.raises(AssertionError):
        assert_partition_invariants(all_uids, ["a"], ["b"], ["c"])


def test_load_three_way_split_roundtrip(tmp_path: Path):
    rng = np.random.default_rng(0)
    messages = np.array([f"msg-{i}" for i in range(40)], dtype=object)
    Y = rng.integers(0, 2, size=(40, len(TARGET_COLUMNS)))
    # Ensure every critical label has positives in the full set.
    for label in CRITICAL_LABELS:
        idx = TARGET_COLUMNS.index(label)
        Y[idx % 40, idx] = 1
        Y[(idx + 10) % 40, idx] = 1

    uids = compute_uids(messages)
    eval_uids = uids[:8]
    cal_uids = uids[8:14]
    eval_path = tmp_path / "eval_ids.json"
    cal_path = tmp_path / "cal_ids.json"
    _write_uid_json(eval_path, "eval_ids", eval_uids)
    _write_uid_json(cal_path, "cal_ids", cal_uids)

    splits = load_three_way_split(messages, Y, eval_path, cal_path)
    X_train, Y_train = splits["train"]
    X_cal, Y_cal = splits["cal"]
    X_eval, Y_eval = splits["eval"]

    assert len(X_train) + len(X_cal) + len(X_eval) == len(messages)
    assert len(X_cal) == 6
    assert len(X_eval) == 8
    assert Y_train.shape[0] == len(X_train)

    train_set = set(uids) - set(eval_uids) - set(cal_uids)
    all_set = set(uids)
    assert_partition_invariants(uids, train_set, cal_uids, eval_uids)
    assert train_set | set(cal_uids) | set(eval_uids) == all_set
    _ = (Y_cal, Y_eval)  # exercised via load_three_way_split


def test_committed_cal_ids_critical_support():
    cal_path = Path("experiments/experimental_configs/eval_sets/cal_ids.json")
    if not cal_path.is_file():
        pytest.skip("cal_ids.json not present")
    data = json.loads(cal_path.read_text(encoding="utf-8"))
    support = data["metadata"]["critical_label_positive_support"]
    assert_critical_support(support, context="committed calibration artifact")
    for label in CRITICAL_LABELS:
        assert support[label] > 0


def test_critical_support_zero_fails():
    Y = np.zeros((5, len(TARGET_COLUMNS)), dtype=int)
    support = critical_label_positive_support(Y)
    assert all(v == 0 for v in support.values())
    with pytest.raises(ValueError, match="Zero positive support"):
        assert_critical_support(support, context="calibration")


def test_optimize_threshold_uses_cal_not_eval():
    """Thresholds come from cal labels; category_stats come from eval scores."""
    import importlib.util

    opt_path = Path("scripts/03_optimization/optimize_per_category_thresholds.py")
    spec = importlib.util.spec_from_file_location("opt_thresholds", opt_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    y_true_cal = np.array([1, 1, 0, 0, 1, 0])
    y_proba_cal = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.05])
    thresh = mod.optimize_threshold_for_category(y_true_cal, y_proba_cal, target_recall=0.65)
    assert 0.0 <= thresh <= 1.0

    thresholds = {label: 0.5 for label in TARGET_COLUMNS}
    thresholds["water"] = thresh

    Y_eval = np.zeros((4, len(TARGET_COLUMNS)), dtype=int)
    Y_cal = np.zeros((4, len(TARGET_COLUMNS)), dtype=int)
    water_idx = TARGET_COLUMNS.index("water")
    Y_eval[:, water_idx] = [1, 1, 0, 0]
    Y_cal[:, water_idx] = [1, 0, 1, 0]
    proba_eval = np.full((4, len(TARGET_COLUMNS)), 0.1)
    proba_cal = np.full((4, len(TARGET_COLUMNS)), 0.9)
    proba_eval[:, water_idx] = [0.9, 0.05, 0.9, 0.05]
    proba_cal[:, water_idx] = [0.9, 0.9, 0.9, 0.9]

    eval_stats = mod.build_category_stats(Y_eval, proba_eval, thresholds, 0.65, 0.60)
    cal_stats = mod.build_category_stats(Y_cal, proba_cal, thresholds, 0.65, 0.60)

    eval_water = next(s for s in eval_stats if s["category"] == "water")
    cal_water = next(s for s in cal_stats if s["category"] == "water")
    assert eval_water["actual_recall"] != cal_water["actual_recall"]
    assert eval_water["support"] == 2
    assert cal_water["support"] == 2
