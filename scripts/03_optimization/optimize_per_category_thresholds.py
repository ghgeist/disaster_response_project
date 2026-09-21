#!/usr/bin/env python3
"""
Optimize Per-Category Thresholds

Tunes per-label thresholds on a frozen calibration split carved from train,
then reports performance on the frozen eval set only.

Usage:
    python scripts/03_optimization/optimize_per_category_thresholds.py \
        --model-path <model.pkl> --output-dir <output>
"""

# Standard library imports
import argparse
import json
import logging
import os
import sys
from datetime import datetime

# Third-party imports
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_recall_curve

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Local imports
from disasterproject.data.loader import load_data
from disasterproject.data.splits import (
    assert_critical_support,
    critical_label_positive_support,
    load_three_way_split,
)
from disasterproject.utils.config import CRITICAL_LABELS, TARGET_COLUMNS, setup_logging


def get_proba_array(model, X):
    """Extract probability array from model predictions."""
    y_proba_list = model.predict_proba(X)
    n_samples = len(X)
    n_labels = len(y_proba_list)
    y_proba = np.zeros((n_samples, n_labels))

    clf = model.named_steps["clf"]

    for i, probs in enumerate(y_proba_list):
        if probs.ndim == 2 and probs.shape[1] == 2:
            y_proba[:, i] = probs[:, 1]
        elif probs.ndim == 2 and probs.shape[1] == 1:
            if hasattr(clf, "classes_") and i < len(clf.classes_):
                classes = clf.classes_[i]
                if len(classes) == 1 and classes[0] == 0:
                    y_proba[:, i] = 0.0
                elif len(classes) == 1 and classes[0] == 1:
                    y_proba[:, i] = 1.0
                else:
                    y_proba[:, i] = probs.ravel()
            else:
                y_proba[:, i] = probs.ravel()
        else:
            y_proba[:, i] = probs.ravel()

    return y_proba


def optimize_threshold_for_category(y_true, y_proba, target_recall=0.65):
    """Optimize threshold for a single category to achieve target recall."""
    if np.sum(y_true) == 0:
        return 0.5

    try:
        _precision, recall, thresh = precision_recall_curve(y_true, y_proba)
        recall_diff = np.abs(recall - target_recall)
        best_idx = int(np.argmin(recall_diff))
        if len(thresh) == 0:
            return 0.5
        return float(thresh[max(0, min(best_idx, len(thresh) - 1))])
    except ValueError as exc:
        logging.warning("Failed to optimize threshold: %s, using default", exc)
        return 0.5


def evaluate_with_thresholds(Y_true, Y_pred, category_names):
    """Evaluate predictions (matches training script calculation)."""
    all_metrics = []

    for i, _label in enumerate(category_names):
        report = classification_report(
            Y_true[:, i],
            Y_pred[:, i],
            output_dict=True,
            zero_division=0,
        )
        if "weighted avg" in report:
            all_metrics.append(report["weighted avg"]["f1-score"])

    f1_weighted = np.mean(all_metrics) if all_metrics else 0.0
    f1_micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)

    return {
        "f1_weighted": f1_weighted,
        "f1_micro": f1_micro,
    }


def _category_type(label: str) -> str:
    return "critical" if label in CRITICAL_LABELS else "non-critical"


def _target_recall_for_label(label: str, critical_recall: float, non_critical_recall: float) -> float:
    if label in CRITICAL_LABELS:
        return critical_recall
    return non_critical_recall


def build_category_stats(Y_true, y_proba, thresholds, critical_recall, non_critical_recall):
    """Per-category metrics for a scored split using frozen thresholds."""
    stats = []
    for i, label in enumerate(TARGET_COLUMNS):
        target_recall = _target_recall_for_label(label, critical_recall, non_critical_recall)
        threshold = thresholds[label]
        y_pred_label = (y_proba[:, i] >= threshold).astype(int)
        report = classification_report(
            Y_true[:, i],
            y_pred_label,
            output_dict=True,
            zero_division=0,
        )
        recall = report.get("1", {}).get("recall", 0.0) if "1" in report else 0.0
        precision = report.get("1", {}).get("precision", 0.0) if "1" in report else 0.0
        f1 = report.get("1", {}).get("f1-score", 0.0) if "1" in report else 0.0
        support = report.get("1", {}).get("support", 0) if "1" in report else 0
        stats.append(
            {
                "category": label,
                "type": _category_type(label),
                "threshold": threshold,
                "target_recall": target_recall,
                "actual_recall": recall,
                "precision": precision,
                "f1": f1,
                "support": support,
            }
        )
    return stats


def critical_recall_mean(category_stats):
    """Mean actual_recall across critical categories."""
    recalls = [
        float(stat["actual_recall"])
        for stat in category_stats
        if stat.get("type") == "critical"
    ]
    if not recalls:
        return 0.0
    return float(np.mean(recalls))


def apply_thresholds(y_proba, thresholds):
    """Apply per-label thresholds to probability matrix."""
    Y_pred = np.zeros((y_proba.shape[0], len(TARGET_COLUMNS)), dtype=int)
    for i, label in enumerate(TARGET_COLUMNS):
        Y_pred[:, i] = (y_proba[:, i] >= thresholds[label]).astype(int)
    return Y_pred


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Optimize thresholds on calibration split; report metrics on frozen eval only"
        )
    )
    parser.add_argument(
        "--model-path",
        default="experiments/experimental_runs/2025-11-04/lr_baseline_model.pkl",
        help="Path to trained model pickle file",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for thresholds (default: same directory as model)",
    )
    parser.add_argument(
        "--db-path",
        default="data/02_stg/stg_disaster_response.db",
        help="Path to database file",
    )
    parser.add_argument(
        "--eval-ids",
        default="experiments/experimental_configs/eval_sets/eval_ids.json",
        help="Path to frozen eval IDs file (report-only)",
    )
    parser.add_argument(
        "--cal-ids",
        default="experiments/experimental_configs/eval_sets/cal_ids.json",
        help="Path to calibration IDs file (threshold tuning)",
    )
    parser.add_argument(
        "--critical-recall",
        type=float,
        default=0.65,
        help="Target recall for critical categories (default: 0.65)",
    )
    parser.add_argument(
        "--non-critical-recall",
        type=float,
        default=0.60,
        help="Target recall for non-critical categories (default: 0.60)",
    )

    args = parser.parse_args()
    setup_logging()

    model_path = args.model_path
    db_path = args.db_path
    eval_ids_path = args.eval_ids
    cal_ids_path = args.cal_ids
    output_dir = args.output_dir or (os.path.dirname(model_path) or ".")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("THRESHOLD OPTIMIZATION (cal tune / eval report)")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Calibration IDs: {cal_ids_path}")
    print(f"Eval IDs (report-only): {eval_ids_path}")
    print(f"Critical Labels: {', '.join(sorted(CRITICAL_LABELS))}")
    print(
        f"Target Recall - Critical: {args.critical_recall:.0%}, "
        f"Non-Critical: {args.non_critical_recall:.0%}"
    )
    print("=" * 70 + "\n")

    print("Loading model...")
    model = joblib.load(model_path)
    print(f"✓ Model loaded: {type(model)}")

    print("Loading data...")
    X, Y = load_data(db_path)
    print(f"✓ Loaded {len(X)} samples with {Y.shape[1]} labels")

    print("Loading three-way train/cal/eval split...")
    try:
        splits = load_three_way_split(X, Y, eval_ids_path, cal_ids_path)
    except (OSError, KeyError, ValueError) as exc:
        logging.error("Failed to load three-way split: %s", exc)
        sys.exit(1)

    _X_train, _Y_train = splits["train"]
    X_cal, Y_cal = splits["cal"]
    X_eval, Y_eval = splits["eval"]

    if len(X_cal) == 0:
        logging.error("Calibration split is empty; cannot optimize thresholds.")
        sys.exit(1)

    cal_support = critical_label_positive_support(Y_cal)
    print("\ncal positive support:")
    for label in sorted(CRITICAL_LABELS):
        print(f"  {label}: {cal_support[label]}")
    try:
        assert_critical_support(cal_support, context="calibration")
    except ValueError as exc:
        logging.error("%s", exc)
        sys.exit(1)

    print(
        f"\nSplit: Train={len(_X_train)}, Cal={len(X_cal)}, Eval={len(X_eval)}"
    )

    print("\n" + "-" * 70)
    print("TUNING THRESHOLDS ON CALIBRATION SPLIT")
    print("-" * 70)
    y_proba_cal = get_proba_array(model, X_cal)
    print(f"✓ Calibration probability array shape: {y_proba_cal.shape}")

    all_thresholds = {}
    for i, label in enumerate(TARGET_COLUMNS):
        target_recall = _target_recall_for_label(
            label, args.critical_recall, args.non_critical_recall
        )
        all_thresholds[label] = optimize_threshold_for_category(
            Y_cal[:, i],
            y_proba_cal[:, i],
            target_recall=target_recall,
        )
        if i % 10 == 0:
            print(f"  Processed {i + 1}/{len(TARGET_COLUMNS)} categories...")

    print(f"\n✓ Optimized thresholds for all {len(TARGET_COLUMNS)} categories on cal")

    calibration_stats = build_category_stats(
        Y_cal,
        y_proba_cal,
        all_thresholds,
        args.critical_recall,
        args.non_critical_recall,
    )
    cal_critical_recall = critical_recall_mean(calibration_stats)
    print(f"Calibration critical recall (diagnostic): {cal_critical_recall:.4f}")

    print("\n" + "-" * 70)
    print("SCORING FROZEN THRESHOLDS ON EVAL (REPORT-ONLY)")
    print("-" * 70)

    Y_pred_baseline = model.predict(X_eval)
    baseline_metrics = evaluate_with_thresholds(
        Y_eval, Y_pred_baseline, TARGET_COLUMNS
    )
    print(
        f"Baseline F1-Weighted: {baseline_metrics['f1_weighted']:.4f} "
        f"(default 0.5 thresholds on eval)"
    )

    y_proba_eval = get_proba_array(model, X_eval)
    Y_pred_optimized = apply_thresholds(y_proba_eval, all_thresholds)
    optimized_metrics = evaluate_with_thresholds(
        Y_eval, Y_pred_optimized, TARGET_COLUMNS
    )
    category_stats = build_category_stats(
        Y_eval,
        y_proba_eval,
        all_thresholds,
        args.critical_recall,
        args.non_critical_recall,
    )
    eval_critical_recall = critical_recall_mean(category_stats)

    print(f"Optimized F1-Weighted: {optimized_metrics['f1_weighted']:.4f}")
    print(f"Optimized F1-Micro: {optimized_metrics['f1_micro']:.4f}")
    print(f"Eval critical recall (reported): {eval_critical_recall:.4f}")

    f1_change = optimized_metrics["f1_weighted"] - baseline_metrics["f1_weighted"]
    f1_change_pct = (f1_change / baseline_metrics["f1_weighted"]) * 100
    print(f"F1-Weighted Change: {f1_change:+.4f} ({f1_change_pct:+.2f}%)")

    stats_df = pd.DataFrame(category_stats)
    print("\n" + "-" * 70)
    print("EVAL CATEGORY STATISTICS (reported)")
    print("-" * 70)
    print("\nCritical Categories:")
    critical_df = stats_df[stats_df["type"] == "critical"].sort_values("threshold")
    print(
        critical_df[
            ["category", "threshold", "actual_recall", "precision", "f1"]
        ].to_string(index=False)
    )

    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    threshold_output_standard = os.path.join(output_dir, f"{model_stem}_thresholds.json")
    threshold_output_legacy = os.path.join(output_dir, "optimized_all_thresholds.json")
    threshold_data = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "model": model_path,
            "critical_target_recall": float(args.critical_recall),
            "non_critical_target_recall": float(args.non_critical_recall),
            "optimization_method": "precision_recall_curve",
            "optimization_split": "calibration",
            "reporting_split": "frozen_eval",
            "calibration_ids": cal_ids_path,
            "eval_ids": eval_ids_path,
            "calibration_critical_recall": cal_critical_recall,
            "eval_critical_recall": eval_critical_recall,
            "critical_label_positive_support_cal": cal_support,
        },
        "thresholds": all_thresholds,
        "calibration_stats": calibration_stats,
        "category_stats": category_stats,
        "performance": {
            "baseline": {
                "f1_weighted": float(baseline_metrics["f1_weighted"]),
                "f1_micro": float(baseline_metrics["f1_micro"]),
            },
            "optimized": {
                "f1_weighted": float(optimized_metrics["f1_weighted"]),
                "f1_micro": float(optimized_metrics["f1_micro"]),
                "critical_recall": eval_critical_recall,
            },
            "delta": {
                "f1_weighted": float(f1_change),
                "f1_weighted_pct": float(f1_change_pct),
            },
        },
    }

    with open(threshold_output_standard, "w", encoding="utf-8") as handle:
        json.dump(threshold_data, handle, indent=2)
    print(f"\n✓ Optimized thresholds saved to: {threshold_output_standard}")

    with open(threshold_output_legacy, "w", encoding="utf-8") as handle:
        json.dump(threshold_data, handle, indent=2)
    print(f"  (Also saved as: {os.path.basename(threshold_output_legacy)} for compatibility)")

    print("\n" + "=" * 70)
    if optimized_metrics["f1_weighted"] >= 0.90 and f1_change_pct >= -5.0:
        print("✅ THRESHOLD OPTIMIZATION SUCCESSFUL")
        print("   Thresholds tuned on cal; metrics reported on frozen eval")
    elif f1_change_pct >= -5.0:
        print("⚠️ THRESHOLD OPTIMIZATION PARTIAL SUCCESS")
        print("   F1 maintained but below 0.90 target on eval")
    else:
        print("❌ THRESHOLD OPTIMIZATION FAILED")
        print("   Eval F1 dropped too much (>5%)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
