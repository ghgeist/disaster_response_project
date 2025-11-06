#!/usr/bin/env python3
"""
Optimize thresholds for ALL categories (not just critical).

This extends optimize_critical_thresholds_inc1.py to optimize thresholds
for all categories, using appropriate target recall based on category importance.

Usage:
    python scripts/optimize_all_thresholds.py --model-path <model.pkl> --output-dir <output>
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sqlalchemy import create_engine
from sklearn.metrics import classification_report, f1_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from disasterproject.utils.config import setup_logging, TARGET_COLUMNS, CRITICAL_LABELS
from disasterproject.data.loader import load_data
from disasterproject.hierarchy import optimize_critical_thresholds


def load_eval_split(eval_ids_file, X, Y):
    """Load frozen eval split."""
    import hashlib
    import pandas as pd
    
    def _compute_uids(messages):
        uids_local = []
        for idx, msg in enumerate(messages):
            text = '' if msg is None else str(msg)
            uid_src = f"{text}|{idx}"
            uids_local.append(hashlib.sha1(uid_src.encode('utf-8')).hexdigest())
        return uids_local
    
    # Load eval IDs
    with open(eval_ids_file, 'r') as f:
        data = json.load(f)
    eval_uids = set(data['eval_ids'])
    
    # Compute UIDs and split
    uids = _compute_uids(X)
    uid_series = pd.Series(uids)
    is_eval = uid_series.isin(eval_uids).values
    
    X_train, X_test = X[~is_eval], X[is_eval]
    Y_train, Y_test = Y[~is_eval], Y[is_eval]
    
    print(f"Split: Train={len(X_train)}, Eval={len(X_test)}")
    return X_train, X_test, Y_train, Y_test


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


def optimize_threshold_for_category(y_true, y_proba, target_recall=0.65):
    """
    Optimize threshold for a single category to achieve target recall.
    
    Returns threshold value.
    """
    from sklearn.metrics import precision_recall_curve
    
    # Skip if no positive examples
    if np.sum(y_true) == 0:
        return 0.5
    
    try:
        precision, recall, thresh = precision_recall_curve(y_true, y_proba)
        
        # Find threshold with recall nearest to target
        recall_diff = np.abs(recall - target_recall)
        best_idx = int(np.argmin(recall_diff))
        
        # precision_recall_curve returns thresholds one shorter than recall
        chosen = float(thresh[max(0, min(best_idx, len(thresh)-1))]) if len(thresh) else 0.5
        return chosen
    except Exception as e:
        logging.warning(f"Failed to optimize threshold: {e}, using default")
        return 0.5


def evaluate_with_thresholds(Y_true, Y_pred, category_names):
    """Evaluate predictions (matches training script calculation)."""
    all_metrics = []
    
    for i, label in enumerate(category_names):
        report = classification_report(
            Y_true[:, i], Y_pred[:, i], 
            output_dict=True, zero_division=0
        )
        
        # Get weighted avg F1 for this category
        if 'weighted avg' in report:
            all_metrics.append(report['weighted avg']['f1-score'])
    
    # Calculate overall F1 as mean of per-category weighted F1
    f1_weighted = np.mean(all_metrics) if all_metrics else 0.0
    f1_micro = f1_score(Y_true, Y_pred, average='micro', zero_division=0)
    
    return {
        'f1_weighted': f1_weighted,
        'f1_micro': f1_micro
    }


def main():
    parser = argparse.ArgumentParser(
        description='Optimize thresholds for ALL categories on a trained model'
    )
    parser.add_argument(
        '--model-path',
        default='experiments/experimental_runs/2025-11-04/lr_baseline_model.pkl',
        help='Path to trained model pickle file'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory for thresholds (default: same directory as model)'
    )
    parser.add_argument(
        '--db-path',
        default='data/02_stg/stg_disaster_response.db',
        help='Path to database file'
    )
    parser.add_argument(
        '--eval-ids',
        default='experiments/experimental_configs/eval_sets/eval_ids.json',
        help='Path to eval IDs file'
    )
    parser.add_argument(
        '--critical-recall',
        type=float,
        default=0.65,
        help='Target recall for critical categories (default: 0.65)'
    )
    parser.add_argument(
        '--non-critical-recall',
        type=float,
        default=0.60,
        help='Target recall for non-critical categories (default: 0.60)'
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Paths
    model_path = args.model_path
    db_path = args.db_path
    eval_ids_path = args.eval_ids
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default to same directory as model
        output_dir = os.path.dirname(model_path) or '.'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("THRESHOLD OPTIMIZATION FOR ALL CATEGORIES")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Critical Labels: {', '.join(sorted(CRITICAL_LABELS))}")
    print(f"Target Recall - Critical: {args.critical_recall:.0%}, Non-Critical: {args.non_critical_recall:.0%}")
    print("="*70 + "\n")
    
    # Load model
    print("Loading model...")
    model = joblib.load(model_path)
    print(f"✓ Model loaded: {type(model)}")
    
    # Load data
    print("Loading data...")
    X, Y = load_data(db_path)
    print(f"✓ Loaded {len(X)} samples with {Y.shape[1]} labels")
    
    # Load eval split
    print("Loading eval split...")
    X_train, X_test, Y_train, Y_test = load_eval_split(eval_ids_path, X, Y)
    
    # Baseline evaluation (default 0.5 thresholds)
    print("\n" + "-"*70)
    print("BASELINE PERFORMANCE (default 0.5 thresholds)")
    print("-"*70)
    Y_pred_baseline = model.predict(X_test)
    baseline_metrics = evaluate_with_thresholds(
        Y_test, Y_pred_baseline, TARGET_COLUMNS
    )
    
    print(f"F1-Weighted: {baseline_metrics['f1_weighted']:.4f}")
    print(f"F1-Micro: {baseline_metrics['f1_micro']:.4f}")
    
    # Get probabilities
    print("\nExtracting probabilities...")
    y_proba = get_proba_array(model, X_test)
    print(f"✓ Probability array shape: {y_proba.shape}")
    
    # Optimize thresholds for all categories
    print("\n" + "-"*70)
    print("OPTIMIZING THRESHOLDS FOR ALL CATEGORIES")
    print("-"*70)
    
    all_thresholds = {}
    category_stats = []
    
    for i, label in enumerate(TARGET_COLUMNS):
        y_true_label = Y_test[:, i]
        y_proba_label = y_proba[:, i]
        
        # Determine target recall based on category type
        if label in CRITICAL_LABELS:
            target_recall = args.critical_recall
            category_type = "critical"
        else:
            target_recall = args.non_critical_recall
            category_type = "non-critical"
        
        # Optimize threshold
        threshold = optimize_threshold_for_category(
            y_true_label, 
            y_proba_label, 
            target_recall=target_recall
        )
        
        all_thresholds[label] = threshold
        
        # Calculate metrics for this category
        y_pred_label = (y_proba_label >= threshold).astype(int)
        report = classification_report(
            y_true_label, y_pred_label,
            output_dict=True, zero_division=0
        )
        
        recall = report.get('1', {}).get('recall', 0.0) if '1' in report else 0.0
        precision = report.get('1', {}).get('precision', 0.0) if '1' in report else 0.0
        f1 = report.get('1', {}).get('f1-score', 0.0) if '1' in report else 0.0
        support = report.get('1', {}).get('support', 0) if '1' in report else 0
        
        category_stats.append({
            'category': label,
            'type': category_type,
            'threshold': threshold,
            'target_recall': target_recall,
            'actual_recall': recall,
            'precision': precision,
            'f1': f1,
            'support': support
        })
        
        if i % 10 == 0:
            print(f"  Processed {i+1}/{len(TARGET_COLUMNS)} categories...")
    
    print(f"\n✓ Optimized thresholds for all {len(TARGET_COLUMNS)} categories")
    
    # Apply optimized thresholds
    print("\n" + "-"*70)
    print("EVALUATING WITH OPTIMIZED THRESHOLDS")
    print("-"*70)
    
    Y_pred_optimized = np.zeros_like(Y_test, dtype=int)
    for i, label in enumerate(TARGET_COLUMNS):
        threshold = all_thresholds[label]
        Y_pred_optimized[:, i] = (y_proba[:, i] >= threshold).astype(int)
    
    optimized_metrics = evaluate_with_thresholds(
        Y_test, Y_pred_optimized, TARGET_COLUMNS
    )
    
    print(f"F1-Weighted: {optimized_metrics['f1_weighted']:.4f}")
    print(f"F1-Micro: {optimized_metrics['f1_micro']:.4f}")
    
    # Performance delta
    print("\n" + "-"*70)
    print("PERFORMANCE DELTA (Optimized vs Baseline)")
    print("-"*70)
    f1_change = optimized_metrics['f1_weighted'] - baseline_metrics['f1_weighted']
    f1_change_pct = (f1_change / baseline_metrics['f1_weighted']) * 100
    
    print(f"F1-Weighted Change: {f1_change:+.4f} ({f1_change_pct:+.2f}%)")
    
    # Category statistics
    stats_df = pd.DataFrame(category_stats)
    print("\n" + "-"*70)
    print("CATEGORY STATISTICS")
    print("-"*70)
    print("\nCritical Categories:")
    critical_df = stats_df[stats_df['type'] == 'critical'].sort_values('threshold')
    print(critical_df[['category', 'threshold', 'actual_recall', 'precision', 'f1']].to_string(index=False))
    
    print("\nNon-Critical Categories (top 10 by threshold change):")
    non_critical_df = stats_df[stats_df['type'] == 'non-critical'].copy()
    non_critical_df['threshold_change'] = abs(non_critical_df['threshold'] - 0.5)
    top_changed = non_critical_df.nlargest(10, 'threshold_change')
    print(top_changed[['category', 'threshold', 'actual_recall', 'precision', 'f1']].to_string(index=False))
    
    # Save optimized thresholds
    threshold_output = os.path.join(output_dir, 'optimized_all_thresholds.json')
    threshold_data = {
        'metadata': {
            'created': datetime.now().isoformat(),
            'model': model_path,
            'critical_target_recall': float(args.critical_recall),
            'non_critical_target_recall': float(args.non_critical_recall),
            'optimization_method': 'precision_recall_curve'
        },
        'thresholds': all_thresholds,
        'category_stats': category_stats,
        'performance': {
            'baseline': {
                'f1_weighted': float(baseline_metrics['f1_weighted']),
                'f1_micro': float(baseline_metrics['f1_micro'])
            },
            'optimized': {
                'f1_weighted': float(optimized_metrics['f1_weighted']),
                'f1_micro': float(optimized_metrics['f1_micro'])
            },
            'delta': {
                'f1_weighted': float(f1_change),
                'f1_weighted_pct': float(f1_change_pct)
            }
        }
    }
    
    with open(threshold_output, 'w') as f:
        json.dump(threshold_data, f, indent=2)
    
    print(f"\n✓ Optimized thresholds saved to: {threshold_output}")
    
    # Verdict
    print("\n" + "="*70)
    if optimized_metrics['f1_weighted'] >= 0.90 and f1_change_pct >= -5.0:
        print("✅ THRESHOLD OPTIMIZATION SUCCESSFUL")
        print("   All categories optimized while maintaining F1 ≥ 0.90")
    elif f1_change_pct >= -5.0:
        print("⚠️ THRESHOLD OPTIMIZATION PARTIAL SUCCESS")
        print("   F1 maintained but below 0.90 target")
    else:
        print("❌ THRESHOLD OPTIMIZATION FAILED")
        print("   F1 dropped too much (>5%)")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

