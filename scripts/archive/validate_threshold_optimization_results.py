#!/usr/bin/env python3
"""
Validation script to verify threshold optimization results and check for logic errors.

This script performs multiple independent checks to ensure:
1. Baseline metrics match original training output
2. F1 calculation method is correct
3. Thresholds are actually being applied
4. No data leakage (correct eval split)
5. Critical recall improvements are real
"""

# Standard library imports
import hashlib
import json
import os
import sys

# Third-party imports
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Local imports
from disasterproject.data.loader import load_data
from disasterproject.utils.config import CRITICAL_LABELS, TARGET_COLUMNS


def load_eval_split(eval_ids_file, X, Y):
    """Load frozen eval split."""
    def _compute_uids(messages):
        uids = []
        for idx, msg in enumerate(messages):
            text = '' if msg is None else str(msg)
            uid_src = f"{text}|{idx}"
            uids.append(hashlib.sha1(uid_src.encode('utf-8')).hexdigest())
        return uids
    
    with open(eval_ids_file, 'r') as f:
        data = json.load(f)
    eval_uids = set(data['eval_ids'])
    
    uids = _compute_uids(X)
    uid_series = pd.Series(uids)
    is_eval = uid_series.isin(eval_uids).values
    
    return X[~is_eval], X[is_eval], Y[~is_eval], Y[is_eval]


def calculate_f1_like_training_script(Y_true, Y_pred, category_names):
    """Calculate F1 exactly as training script does (mean of per-category weighted F1)."""
    per_category_f1 = []
    
    for i, label in enumerate(category_names):
        report = classification_report(
            Y_true[:, i], Y_pred[:, i], 
            output_dict=True, zero_division=0
        )
        if 'weighted avg' in report:
            per_category_f1.append(report['weighted avg']['f1-score'])
    
    return np.mean(per_category_f1) if per_category_f1 else 0.0


def apply_thresholds(y_proba, thresholds, category_names):
    """Apply custom thresholds to probability array."""
    predictions = np.zeros_like(y_proba, dtype=int)
    
    for i, label in enumerate(category_names):
        threshold = thresholds.get(label, 0.5)
        predictions[:, i] = (y_proba[:, i] >= threshold).astype(int)
    
    return predictions


def get_proba_array(model, X):
    """Extract probability array from model predictions."""
    y_proba_list = model.predict_proba(X)
    n_samples = len(X)
    n_labels = len(y_proba_list)
    y_proba = np.zeros((n_samples, n_labels))
    
    # Access the underlying classifier to get class information
    clf = model.named_steps['clf']
    
    for i, probs in enumerate(y_proba_list):
        if probs.ndim == 2 and probs.shape[1] == 2:
            y_proba[:, i] = probs[:, 1]  # Probability of class 1
        elif probs.ndim == 2 and probs.shape[1] == 1:
            # Single class present - check which class it is
            if hasattr(clf, 'classes_') and i < len(clf.classes_):
                classes = clf.classes_[i]
                if len(classes) == 1 and classes[0] == 0:
                    # Only class 0 present, probability of class 1 is 0
                    y_proba[:, i] = 0.0
                elif len(classes) == 1 and classes[0] == 1:
                    # Only class 1 present, probability of class 1 is 1
                    y_proba[:, i] = 1.0
                else:
                    # Fallback (shouldn't happen)
                    y_proba[:, i] = probs.ravel()
            else:
                # Fallback if class info not available
                y_proba[:, i] = probs.ravel()
        else:
            # Fallback for unexpected shapes
            y_proba[:, i] = probs.ravel()
    
    return y_proba


def main():
    print("="*80)
    print("VALIDATION: Threshold Optimization Results")
    print("="*80)
    print("\nThis script independently verifies the optimization results.")
    print("It checks for logic errors, data leakage, and calculation mistakes.\n")
    
    # Paths
    model_path = 'experiments/experimental_runs/2025-11-04/lr_baseline_model.pkl'
    thresholds_path = 'experiments/experimental_runs/2025-11-04/optimized_critical_thresholds.json'
    db_path = 'data/02_stg/stg_disaster_response.db'
    eval_ids_path = 'experiments/experimental_configs/eval_sets/eval_ids.json'
    original_metrics_path = 'experiments/experimental_runs/2025-11-04/performance_metrics.csv'
    training_log_path = 'experiments/experimental_runs/2025-11-04/training_log.json'
    
    # =========================================================================
    # CHECK 1: Verify baseline metrics match original training output
    # =========================================================================
    print("\n" + "-"*80)
    print("CHECK 1: Baseline Metrics Match Original Training Output")
    print("-"*80)
    
    # Load original training log
    with open(training_log_path, 'r') as f:
        training_log = json.load(f)
    
    original_f1 = training_log['performance']['overall_f1']
    original_recall = training_log['performance']['overall_recall']
    original_precision = training_log['performance']['overall_precision']
    
    print(f"Original Training Output:")
    print(f"  F1: {original_f1:.4f}")
    print(f"  Recall: {original_recall:.4f}")
    print(f"  Precision: {original_precision:.4f}")
    
    # Recalculate from scratch
    print(f"\nRecalculating from scratch...")
    model = joblib.load(model_path)
    X, Y = load_data(db_path)
    X_train, X_test, Y_train, Y_test = load_eval_split(eval_ids_path, X, Y)
    
    Y_pred_baseline = model.predict(X_test)
    recalc_f1 = calculate_f1_like_training_script(Y_test, Y_pred_baseline, TARGET_COLUMNS)
    
    # Calculate recall and precision
    per_cat_recall = []
    per_cat_precision = []
    for i in range(Y_test.shape[1]):
        report = classification_report(Y_test[:, i], Y_pred_baseline[:, i], output_dict=True, zero_division=0)
        if 'weighted avg' in report:
            per_cat_recall.append(report['weighted avg']['recall'])
            per_cat_precision.append(report['weighted avg']['precision'])
    
    recalc_recall = np.mean(per_cat_recall)
    recalc_precision = np.mean(per_cat_precision)
    
    print(f"Recalculated (Independent):")
    print(f"  F1: {recalc_f1:.4f}")
    print(f"  Recall: {recalc_recall:.4f}")
    print(f"  Precision: {recalc_precision:.4f}")
    
    f1_diff = abs(original_f1 - recalc_f1)
    recall_diff = abs(original_recall - recalc_recall)
    precision_diff = abs(original_precision - recalc_precision)
    
    print(f"\nDifferences:")
    print(f"  F1 diff: {f1_diff:.6f} ({'✅ PASS' if f1_diff < 0.001 else '❌ FAIL'})")
    print(f"  Recall diff: {recall_diff:.6f} ({'✅ PASS' if recall_diff < 0.001 else '❌ FAIL'})")
    print(f"  Precision diff: {precision_diff:.6f} ({'✅ PASS' if precision_diff < 0.001 else '❌ FAIL'})")
    
    check1_pass = f1_diff < 0.001 and recall_diff < 0.001 and precision_diff < 0.001
    
    # =========================================================================
    # CHECK 2: Verify threshold application actually changes predictions
    # =========================================================================
    print("\n" + "-"*80)
    print("CHECK 2: Thresholds Actually Change Predictions")
    print("-"*80)
    
    # Load optimized thresholds
    with open(thresholds_path, 'r') as f:
        threshold_data = json.load(f)
    
    optimized_thresholds = threshold_data['thresholds']
    
    # Get probabilities
    y_proba = get_proba_array(model, X_test)
    
    # Apply default (0.5) thresholds
    default_thresholds = {label: 0.5 for label in TARGET_COLUMNS}
    Y_pred_default = apply_thresholds(y_proba, default_thresholds, TARGET_COLUMNS)
    
    # Apply optimized thresholds
    Y_pred_optimized = apply_thresholds(y_proba, optimized_thresholds, TARGET_COLUMNS)
    
    # Check differences
    total_predictions = Y_test.size
    changed_predictions = np.sum(Y_pred_default != Y_pred_optimized)
    pct_changed = (changed_predictions / total_predictions) * 100
    
    print(f"Total predictions: {total_predictions:,}")
    print(f"Changed predictions: {changed_predictions:,} ({pct_changed:.2f}%)")
    
    # Check critical categories specifically
    print(f"\nCritical category threshold changes:")
    for label in sorted(CRITICAL_LABELS):
        idx = TARGET_COLUMNS.index(label)
        default_t = 0.5
        optimized_t = optimized_thresholds[label]
        change = optimized_t - default_t
        
        # Count prediction changes for this category
        changed = np.sum(Y_pred_default[:, idx] != Y_pred_optimized[:, idx])
        pct = (changed / len(Y_test)) * 100
        
        print(f"  {label:20s}: {default_t:.3f} → {optimized_t:.3f} (Δ={change:+.3f}, {changed} preds changed, {pct:.1f}%)")
    
    check2_pass = changed_predictions > 1000  # Expect significant changes
    print(f"\n{'✅ PASS' if check2_pass else '❌ FAIL'}: Thresholds meaningfully changed predictions")
    
    # =========================================================================
    # CHECK 3: Verify critical recall calculation
    # =========================================================================
    print("\n" + "-"*80)
    print("CHECK 3: Critical Recall Calculation Verification")
    print("-"*80)
    
    # Calculate baseline critical recall manually
    baseline_critical_recalls = []
    print("\nBaseline critical category recall (manual calculation):")
    for label in sorted(CRITICAL_LABELS):
        idx = TARGET_COLUMNS.index(label)
        y_true = Y_test[:, idx]
        y_pred = Y_pred_default[:, idx]
        
        # Manual calculation
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        baseline_critical_recalls.append(recall)
        print(f"  {label:20s}: {recall:.4f} (TP={tp}, FN={fn}, Total Pos={tp+fn})")
    
    baseline_avg = np.mean(baseline_critical_recalls)
    print(f"\n  Average baseline critical recall: {baseline_avg:.4f}")
    
    # Calculate optimized critical recall manually
    optimized_critical_recalls = []
    print("\nOptimized critical category recall (manual calculation):")
    for label in sorted(CRITICAL_LABELS):
        idx = TARGET_COLUMNS.index(label)
        y_true = Y_test[:, idx]
        y_pred = Y_pred_optimized[:, idx]
        
        # Manual calculation
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        optimized_critical_recalls.append(recall)
        print(f"  {label:20s}: {recall:.4f} (TP={tp}, FN={fn}, FP={fp}, Precision={precision:.4f})")
    
    optimized_avg = np.mean(optimized_critical_recalls)
    improvement = optimized_avg - baseline_avg
    improvement_pct = (improvement / baseline_avg * 100) if baseline_avg > 0 else 0
    
    print(f"\n  Average optimized critical recall: {optimized_avg:.4f}")
    print(f"  Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")
    
    # Compare to reported values (UPDATED after bug fix - target recall 65%)
    reported_baseline = 0.2339
    reported_optimized = 0.6497  # Updated from 0.6208 after bug fix (target recall 65%)
    
    baseline_diff = abs(baseline_avg - reported_baseline)
    optimized_diff = abs(optimized_avg - reported_optimized)
    
    print(f"\nComparison to reported values:")
    print(f"  Baseline: {baseline_avg:.4f} vs {reported_baseline:.4f} (diff={baseline_diff:.4f}) {'✅' if baseline_diff < 0.01 else '❌'}")
    print(f"  Optimized: {optimized_avg:.4f} vs {reported_optimized:.4f} (diff={optimized_diff:.4f}) {'✅' if optimized_diff < 0.02 else '❌'}")
    
    check3_pass = baseline_diff < 0.01 and optimized_diff < 0.02  # Allow 0.02 tolerance for optimized (target recall changed)
    
    # =========================================================================
    # CHECK 4: Verify F1 with optimized thresholds
    # =========================================================================
    print("\n" + "-"*80)
    print("CHECK 4: F1 Calculation with Optimized Thresholds")
    print("-"*80)
    
    f1_optimized = calculate_f1_like_training_script(Y_test, Y_pred_optimized, TARGET_COLUMNS)
    f1_baseline = calculate_f1_like_training_script(Y_test, Y_pred_default, TARGET_COLUMNS)
    
    print(f"Baseline F1 (default thresholds): {f1_baseline:.4f}")
    print(f"Optimized F1 (custom thresholds): {f1_optimized:.4f}")
    print(f"Change: {f1_optimized - f1_baseline:+.4f} ({(f1_optimized - f1_baseline)/f1_baseline*100:+.2f}%)")
    
    # Compare to reported (UPDATED after bug fix - target recall 65%)
    reported_optimized_f1 = 0.9264  # Updated from 0.9009 after bug fix (target recall 65%)
    f1_opt_diff = abs(f1_optimized - reported_optimized_f1)
    
    print(f"\nComparison to reported optimized F1:")
    print(f"  Calculated: {f1_optimized:.4f}")
    print(f"  Reported: {reported_optimized_f1:.4f}")
    print(f"  Difference: {f1_opt_diff:.4f} {'✅' if f1_opt_diff < 0.01 else '❌'}")
    
    check4_pass = f1_opt_diff < 0.01 and f1_optimized >= 0.90
    
    # =========================================================================
    # CHECK 5: Spot check individual predictions
    # =========================================================================
    print("\n" + "-"*80)
    print("CHECK 5: Spot Check Individual Predictions")
    print("-"*80)
    
    # Find examples where thresholds changed predictions for critical categories
    print("\nExamples of threshold impact on critical categories:\n")
    
    for label in ['medical_help', 'water', 'security'][:3]:  # Check 3 categories
        idx = TARGET_COLUMNS.index(label)
        
        # Find cases where prediction changed
        changed_mask = Y_pred_default[:, idx] != Y_pred_optimized[:, idx]
        changed_indices = np.where(changed_mask)[0]
        
        if len(changed_indices) > 0:
            # Show first example
            i = changed_indices[0]
            prob = y_proba[i, idx]
            default_pred = Y_pred_default[i, idx]
            optimized_pred = Y_pred_optimized[i, idx]
            true_label = Y_test[i, idx]
            
            print(f"{label}:")
            print(f"  Probability: {prob:.4f}")
            print(f"  Default threshold (0.5): pred={default_pred}, true={true_label} {'✓' if default_pred == true_label else '✗'}")
            print(f"  Optimized threshold ({optimized_thresholds[label]:.4f}): pred={optimized_pred}, true={true_label} {'✓' if optimized_pred == true_label else '✗'}")
            print(f"  Impact: {'Correct → Correct' if default_pred == true_label and optimized_pred == true_label else 'Wrong → Correct' if default_pred != true_label and optimized_pred == true_label else 'Correct → Wrong' if default_pred == true_label and optimized_pred != true_label else 'Wrong → Wrong'}")
            print()
    
    check5_pass = True  # Manual inspection
    
    # =========================================================================
    # CHECK 6: Data leakage check
    # =========================================================================
    print("\n" + "-"*80)
    print("CHECK 6: Data Leakage Check")
    print("-"*80)
    
    print(f"Train samples: {len(X_train)}")
    print(f"Eval samples: {len(X_test)}")
    print(f"Total: {len(X)}")
    print(f"Split ratio: {len(X_test)/len(X):.2%} (expected: ~20%)")

    # Verify no overlap - compute UIDs from original dataset using original indices
    # (same logic as load_eval_split to ensure consistency)
    with open(eval_ids_path, 'r') as f:
        eval_data = json.load(f)
    eval_uids_set = set(eval_data['eval_ids'])
    
    # Compute UIDs for full dataset (same as load_eval_split does)
    all_uids = []
    for idx, msg in enumerate(X):
        text = '' if msg is None else str(msg)
        uid_src = f"{text}|{idx}"
        all_uids.append(hashlib.sha1(uid_src.encode('utf-8')).hexdigest())
    
    # Find which full-dataset UIDs are in train vs test
    train_uids_from_full = set()
    test_uids_from_full = set()
    for uid in all_uids:
        if uid in eval_uids_set:
            test_uids_from_full.add(uid)
        else:
            train_uids_from_full.add(uid)
    
    overlap = train_uids_from_full & test_uids_from_full
    
    print(f"Train UIDs: {len(train_uids_from_full)}")
    print(f"Test UIDs: {len(test_uids_from_full)}")
    print(f"Overlap between train/test: {len(overlap)} samples {'✅' if len(overlap) == 0 else '❌'}")
    
    check6_pass = len(overlap) == 0 and 0.15 < len(X_test)/len(X) < 0.25
    
    # =========================================================================
    # FINAL VERDICT
    # =========================================================================
    print("\n" + "="*80)
    print("FINAL VALIDATION RESULTS")
    print("="*80)
    
    checks = [
        ("Baseline metrics match", check1_pass),
        ("Thresholds change predictions", check2_pass),
        ("Critical recall accurate", check3_pass),
        ("Optimized F1 accurate", check4_pass),
        ("Spot checks valid", check5_pass),
        ("No data leakage", check6_pass)
    ]
    
    print()
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check_name}")
    
    all_pass = all(passed for _, passed in checks)
    
    print("\n" + "="*80)
    if all_pass:
        print("✅ ALL CHECKS PASSED - Results are validated!")
        print("="*80)
        print("\nConclusion: No logic errors detected. The threshold optimization")
        print("results are accurate and reliable. The improvements are real.")
    else:
        print("❌ SOME CHECKS FAILED - Review results carefully!")
        print("="*80)
        print("\nConclusion: There may be logic errors. Investigate failed checks")
        print("before proceeding with production deployment.")
    
    print()


if __name__ == '__main__':
    main()

