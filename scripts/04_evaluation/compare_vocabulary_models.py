#!/usr/bin/env python3
"""
Compare vocabulary-optimized models and generate comprehensive comparison report.

Analyzes all vocabulary-limited models, compares performance metrics,
model sizes, and generates a markdown report.
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from disasterproject.utils.config import setup_logging, TARGET_COLUMNS, CRITICAL_LABELS
from disasterproject.data.loader import load_data
from sklearn.metrics import classification_report, f1_score
import hashlib

def load_eval_split(eval_ids_file, X, Y):
    """Load frozen eval split."""
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
    
    return X_train, X_test, Y_train, Y_test

def predict_with_thresholds(model, X, thresholds, category_names):
    """Apply custom thresholds to model predictions."""
    # Get probabilities
    y_proba = model.predict_proba(X)
    
    # Convert list of arrays to 2D array
    n_samples = len(X)
    n_labels = len(category_names)
    proba_array = np.zeros((n_samples, n_labels))
    
    # Access the underlying classifier to get class information
    clf = model.named_steps['clf']
    
    for i, probs in enumerate(y_proba):
        if probs.ndim == 2 and probs.shape[1] == 2:
            proba_array[:, i] = probs[:, 1]  # Probability of class 1
        elif probs.ndim == 2 and probs.shape[1] == 1:
            # Single class present - check which class it is
            if hasattr(clf, 'classes_') and i < len(clf.classes_):
                classes = clf.classes_[i]
                if len(classes) == 1 and classes[0] == 0:
                    # Only class 0 present, probability of class 1 is 0
                    proba_array[:, i] = 0.0
                elif len(classes) == 1 and classes[0] == 1:
                    # Only class 1 present, probability of class 1 is 1
                    proba_array[:, i] = 1.0
                else:
                    # Fallback (shouldn't happen)
                    proba_array[:, i] = probs.ravel()
            else:
                # Fallback if class info not available
                proba_array[:, i] = probs.ravel()
        else:
            # Fallback for unexpected shapes
            proba_array[:, i] = probs.ravel()
    
    # Apply thresholds
    predictions = np.zeros((n_samples, n_labels), dtype=int)
    for i, label in enumerate(category_names):
        threshold = thresholds.get(label, 0.5)
        predictions[:, i] = (proba_array[:, i] >= threshold).astype(int)
    
    return predictions, proba_array

def load_model_metrics(model_path, thresholds_path=None):
    """Load model and calculate metrics."""
    # Load model
    model = joblib.load(model_path)
    
    # Get model size
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    # Load training log if available
    model_dir = os.path.dirname(model_path)
    training_log_path = os.path.join(model_dir, 'training_log.json')
    training_time = None
    baseline_f1 = None
    baseline_recall = None
    baseline_precision = None
    
    if os.path.exists(training_log_path):
        with open(training_log_path, 'r') as f:
            log = json.load(f)
            training_time = log.get('training_time_seconds')
            perf = log.get('performance', {})
            baseline_f1 = perf.get('overall_f1')
            baseline_recall = perf.get('overall_recall')
            baseline_precision = perf.get('overall_precision')
    
    # Load optimized thresholds if available
    optimized_f1 = None
    optimized_critical_recall = None
    thresholds = None
    
    if thresholds_path and os.path.exists(thresholds_path):
        with open(thresholds_path, 'r') as f:
            thresh_data = json.load(f)
            perf = thresh_data.get('performance', {})
            optimized_f1 = perf.get('optimized', {}).get('f1_weighted')
            optimized_critical_recall = perf.get('optimized', {}).get('critical_recall_mean')
            thresholds = thresh_data.get('thresholds')
    
    return {
        'model': model,
        'model_size_mb': model_size_mb,
        'training_time': training_time,
        'baseline_f1': baseline_f1,
        'baseline_recall': baseline_recall,
        'baseline_precision': baseline_precision,
        'optimized_f1': optimized_f1,
        'optimized_critical_recall': optimized_critical_recall,
        'thresholds': thresholds
    }

def calculate_per_category_metrics(model, X_test, Y_test, thresholds, category_names):
    """Calculate per-category metrics."""
    Y_pred, _ = predict_with_thresholds(model, X_test, thresholds, category_names)
    
    per_category = []
    for i, label in enumerate(category_names):
        report = classification_report(
            Y_test[:, i], Y_pred[:, i],
            output_dict=True, zero_division=0
        )
        
        if '1' in report:
            per_category.append({
                'category': label,
                'recall': report['1']['recall'],
                'precision': report['1']['precision'],
                'f1': report['1']['f1-score'],
                'support': report['1']['support'],
                'is_critical': label in CRITICAL_LABELS
            })
        else:
            per_category.append({
                'category': label,
                'recall': 0.0,
                'precision': 0.0,
                'f1': 0.0,
                'support': 0.0,
                'is_critical': label in CRITICAL_LABELS
            })
    
    return pd.DataFrame(per_category)

def main():
    parser = argparse.ArgumentParser(
        description='Compare vocabulary-optimized models'
    )
    parser.add_argument(
        '--base-model',
        default='experiments/experimental_runs/2025-11-04/lr_baseline_model.pkl',
        help='Baseline model path (unlimited vocabulary)'
    )
    parser.add_argument(
        '--base-thresholds',
        default='experiments/experimental_runs/2025-11-04/optimized_critical_thresholds.json',
        help='Baseline thresholds path'
    )
    parser.add_argument(
        '--output',
        default='experiments/experimental_runs/2025-11-06/vocabulary_comparison_report.md',
        help='Output markdown report path'
    )
    parser.add_argument(
        '--db-path',
        default='data/02_stg/stg_disaster_response.db',
        help='Database path'
    )
    parser.add_argument(
        '--eval-ids',
        default='experiments/experimental_configs/eval_sets/eval_ids.json',
        help='Eval IDs path'
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    print("="*80)
    print("VOCABULARY MODEL COMPARISON")
    print("="*80)
    print(f"Base model: {args.base_model}")
    print(f"Output: {args.output}")
    print("="*80 + "\n")
    
    # Load data
    print("Loading data...")
    X, Y = load_data(args.db_path)
    X_train, X_test, Y_train, Y_test = load_eval_split(args.eval_ids, X, Y)
    print(f"✓ Loaded data: Train={len(X_train)}, Eval={len(X_test)}\n")
    
    # Define models to compare
    models_to_compare = [
        {
            'name': 'Baseline (Unlimited)',
            'model_path': args.base_model,
            'thresholds_path': args.base_thresholds,
            'vocab_config': 'Unlimited (230K features)',
            'vocab_params': {'max_features': None, 'min_df': 1, 'max_df': 1.0}
        },
        {
            'name': 'Baseline Filters',
            'model_path': 'experiments/experimental_runs/2025-11-06/lr_vocab_baseline_filters_model.pkl',
            'thresholds_path': 'experiments/experimental_runs/2025-11-06/vocab_baseline_filters/optimized_critical_thresholds.json',
            'vocab_config': 'Filters only (min_df=2, max_df=0.95)',
            'vocab_params': {'max_features': None, 'min_df': 2, 'max_df': 0.95}
        },
        {
            'name': '30K Features',
            'model_path': 'experiments/experimental_runs/2025-11-06/lr_vocab30k_model.pkl',
            'thresholds_path': 'experiments/experimental_runs/2025-11-06/vocab30k/optimized_critical_thresholds.json',
            'vocab_config': '30K features + filters',
            'vocab_params': {'max_features': 30000, 'min_df': 2, 'max_df': 0.95}
        },
        {
            'name': '25K Features',
            'model_path': 'experiments/experimental_runs/2025-11-06/lr_vocab25k_model.pkl',
            'thresholds_path': 'experiments/experimental_runs/2025-11-06/vocab25k/optimized_critical_thresholds.json',
            'vocab_config': '25K features + filters',
            'vocab_params': {'max_features': 25000, 'min_df': 2, 'max_df': 0.95}
        },
        {
            'name': '20K Features',
            'model_path': 'experiments/experimental_runs/2025-11-06/lr_vocab20k_model.pkl',
            'thresholds_path': 'experiments/experimental_runs/2025-11-06/vocab20k/optimized_critical_thresholds.json',
            'vocab_config': '20K features + filters',
            'vocab_params': {'max_features': 20000, 'min_df': 2, 'max_df': 0.95}
        },
        {
            'name': '15K Features',
            'model_path': 'experiments/experimental_runs/2025-11-06/lr_vocab15k_model.pkl',
            'thresholds_path': 'experiments/experimental_runs/2025-11-06/vocab15k/optimized_critical_thresholds.json',
            'vocab_config': '15K features + aggressive filters',
            'vocab_params': {'max_features': 15000, 'min_df': 3, 'max_df': 0.90}
        }
    ]
    
    # Collect results
    results = []
    
    for model_info in models_to_compare:
        print(f"\nProcessing: {model_info['name']}")
        print("-" * 80)
        
        if not os.path.exists(model_info['model_path']):
            print(f"⚠ Model not found: {model_info['model_path']}")
            continue
        
        try:
            metrics = load_model_metrics(
                model_info['model_path'],
                model_info.get('thresholds_path')
            )
            
            # Calculate per-category metrics if thresholds available
            per_category = None
            if metrics['thresholds']:
                print("  Calculating per-category metrics...")
                per_category = calculate_per_category_metrics(
                    metrics['model'],
                    X_test,
                    Y_test,
                    metrics['thresholds'],
                    TARGET_COLUMNS
                )
            
            result = {
                'name': model_info['name'],
                'vocab_config': model_info['vocab_config'],
                'vocab_params': model_info['vocab_params'],
                'model_size_mb': metrics['model_size_mb'],
                'training_time': metrics['training_time'],
                'baseline_f1': metrics['baseline_f1'],
                'baseline_recall': metrics['baseline_recall'],
                'baseline_precision': metrics['baseline_precision'],
                'optimized_f1': metrics['optimized_f1'],
                'optimized_critical_recall': metrics['optimized_critical_recall'],
                'per_category': per_category,
                'model_path': model_info['model_path'],
                'thresholds_path': model_info.get('thresholds_path')
            }
            
            results.append(result)
            print(f"  ✓ Model size: {metrics['model_size_mb']:.2f} MB")
            print(f"  ✓ Baseline F1: {metrics['baseline_f1']:.4f}" if metrics['baseline_f1'] else "  ✓ Baseline F1: N/A")
            print(f"  ✓ Optimized F1: {metrics['optimized_f1']:.4f}" if metrics['optimized_f1'] else "  ✓ Optimized F1: N/A")
            print(f"  ✓ Critical Recall: {metrics['optimized_critical_recall']:.4f}" if metrics['optimized_critical_recall'] else "  ✓ Critical Recall: N/A")
            
        except Exception as e:
            print(f"  ✗ Error processing model: {e}")
            logging.error(f"Error processing {model_info['name']}: {e}", exc_info=True)
    
    # Generate comparison report
    print("\n" + "="*80)
    print("GENERATING COMPARISON REPORT")
    print("="*80)
    
    # Find baseline for comparison
    baseline_result = next((r for r in results if r['name'] == 'Baseline (Unlimited)'), None)
    baseline_size = baseline_result['model_size_mb'] if baseline_result else 67.69
    baseline_f1 = baseline_result['optimized_f1'] if baseline_result and baseline_result['optimized_f1'] else baseline_result['baseline_f1'] if baseline_result else 0.9264
    
    # Create markdown report
    report_lines = [
        "# Vocabulary Size Optimization Comparison Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        f"This report compares {len(results)} vocabulary-optimized models against the baseline unlimited vocabulary model.",
        "",
        "## Overall Performance Comparison",
        "",
        "| Model | Vocabulary Config | Model Size | Size Reduction | Baseline F1 | Optimized F1 | F1 Change | Critical Recall | Training Time |",
        "|-------|------------------|------------|----------------|-------------|--------------|-----------|-----------------|---------------|"
    ]
    
    for result in results:
        size_reduction = ((baseline_size - result['model_size_mb']) / baseline_size * 100) if baseline_size > 0 else 0
        f1_change = (result['optimized_f1'] - baseline_f1) if result['optimized_f1'] else (result['baseline_f1'] - baseline_f1) if result['baseline_f1'] else None
        f1_change_str = f"{f1_change:+.4f}" if f1_change is not None else "N/A"
        
        baseline_f1_str = f"{result['baseline_f1']:.4f}" if result['baseline_f1'] else "N/A"
        optimized_f1_str = f"{result['optimized_f1']:.4f}" if result['optimized_f1'] else "N/A"
        critical_recall_str = f"{result['optimized_critical_recall']:.4f}" if result['optimized_critical_recall'] else "N/A"
        training_time_str = f"{result['training_time']:.1f}s" if result['training_time'] else "N/A"
        
        report_lines.append(
            f"| {result['name']} | {result['vocab_config']} | "
            f"{result['model_size_mb']:.2f} MB | {size_reduction:.1f}% | "
            f"{baseline_f1_str} | {optimized_f1_str} | "
            f"{f1_change_str} | {critical_recall_str} | "
            f"{training_time_str} |"
        )
    
    report_lines.extend([
        "",
        "## Critical Category Performance",
        ""
    ])
    
    # Per-category comparison for models with thresholds
    models_with_thresholds = [r for r in results if r['per_category'] is not None]
    if models_with_thresholds:
        # Critical categories only
        critical_cats = sorted(CRITICAL_LABELS)
        
        report_lines.append("### Critical Categories (with Optimized Thresholds)")
        report_lines.append("")
        report_lines.append("| Category | " + " | ".join([r['name'] for r in models_with_thresholds]) + " |")
        report_lines.append("|----------|" + "|".join(["----------" for _ in models_with_thresholds]) + "|")
        
        for cat in critical_cats:
            row = [f"**{cat}**"]
            for result in models_with_thresholds:
                cat_data = result['per_category'][result['per_category']['category'] == cat]
                if len(cat_data) > 0:
                    recall = cat_data.iloc[0]['recall']
                    row.append(f"{recall:.1%}")
                else:
                    row.append("N/A")
            report_lines.append("| " + " | ".join(row) + " |")
    
    report_lines.extend([
        "",
        "## Recommendations",
        ""
    ])
    
    # Find best model (smallest size with F1 ≥ 92.0% and Critical Recall ≥ 64%)
    valid_models = [
        r for r in results
        if r['optimized_f1'] and r['optimized_f1'] >= 0.92
        and r['optimized_critical_recall'] and r['optimized_critical_recall'] >= 0.64
    ]
    
    if valid_models:
        best_model = min(valid_models, key=lambda x: x['model_size_mb'])
        report_lines.extend([
            f"### Best Model: **{best_model['name']}**",
            "",
            f"- **Model Size**: {best_model['model_size_mb']:.2f} MB ({((baseline_size - best_model['model_size_mb']) / baseline_size * 100):.1f}% reduction)",
            f"- **F1-Weighted**: {best_model['optimized_f1']:.4f}",
            f"- **Critical Recall**: {best_model['optimized_critical_recall']:.4f}",
            f"- **Training Time**: {best_model['training_time']:.1f}s",
            "",
            f"**Model Path**: `{best_model['model_path']}`",
            f"**Thresholds Path**: `{best_model['thresholds_path']}`",
            "",
            "### All Valid Models",
            "",
            "The following models meet production gates (F1 ≥ 92.0%, Critical Recall ≥ 64%):",
            ""
        ])
        
        for model in sorted(valid_models, key=lambda x: x['model_size_mb']):
            size_red = ((baseline_size - model['model_size_mb']) / baseline_size * 100)
            report_lines.append(
                f"- **{model['name']}**: {model['model_size_mb']:.2f} MB "
                f"({size_red:.1f}% reduction), F1={model['optimized_f1']:.4f}, "
                f"Critical Recall={model['optimized_critical_recall']:.4f}"
            )
    else:
        report_lines.append("⚠ No models meet all production gates. Review results carefully.")
    
    report_lines.extend([
        "",
        "## Vocabulary Parameters",
        "",
        "| Model | max_features | min_df | max_df |",
        "|-------|--------------|--------|--------|"
    ])
    
    for result in results:
        params = result['vocab_params']
        report_lines.append(
            f"| {result['name']} | "
            f"{params['max_features'] if params['max_features'] else 'None'} | "
            f"{params['min_df']} | {params['max_df']} |"
        )
    
    # Save report
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Comparison report saved to: {args.output}")
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()

