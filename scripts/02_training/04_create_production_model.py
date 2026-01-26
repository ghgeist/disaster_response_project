#!/usr/bin/env python3
"""
Create a production disaster response classification model.

⚠️ IMPORTANT (2026-01-22): This script requires --params and --class-weights arguments.
The default files (model/parameters.json, model/class_weights.json) were removed.
These defaults were for RandomForest models only.

Current production models use LogisticRegression and are created via:
  scripts/02_training/03_create_experimental_model.py --algorithm logistic_regression

This script creates a production model and saves results in a clean, obvious structure:
- model/disaster_rf_v1-2-0_prod_2025-09-11.pkl (the current production model)
- model/performance_metrics.csv (current model performance)
- model/training_log.json (training metadata)

Usage:
    python scripts/02_training/04_create_production_model.py \
      --params experiments/model_candidates/vocab_15k.json \
      --class-weights experiments/model_candidates/class_weights.json
"""

import argparse
import os
import sys
import logging
import json
from datetime import datetime
import hashlib
from time import time

# Import from installed package (requires: pip install -e .)
# Alternative: set PYTHONPATH to include src directory

from disasterproject.utils.config import setup_logging, TARGET_COLUMNS, DEFAULT_TEST_SIZE, DEFAULT_RANDOM_SEED, TAXONOMY, CRITICAL_LABELS, EXCLUDE_FROM_CONSTRAINTS, HIERARCHY_CRITICAL_THRESHOLD_REDUCTION
from disasterproject.utils.json_io import load_model_parameters
from disasterproject.data.loader import load_data
from disasterproject.models.pipeline import (
    create_pipeline, 
    create_pipeline_with_custom_weights,
    build_model
)
from disasterproject.models.samplers import get_multilabel_class_weights
from disasterproject.evaluation.metrics import evaluate_model, save_model
from disasterproject.hierarchy import apply_hierarchy, count_violations
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import joblib


def load_class_weights_config(file_path):
    """Load class weights configuration from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logging.error(f'Class weights file not found: {file_path}')
        return None
    except json.JSONDecodeError:
        logging.error(f'Invalid JSON in class weights file: {file_path}')
        return None
    except Exception as e:
        logging.error(f'Error loading class weights config: {e}')
        return None


def evaluate_model_to_model_folder(model, X_test, Y_test, category_names, model_dir="model"):
    """
    Evaluate model with both baseline and hierarchy-corrected predictions.

    Args:
        model: Trained model
        X_test: Test features
        Y_test: Test labels
        category_names: List of category names
        model_dir: Directory to save results (default: "model")

    Returns:
        dict: Performance summary including hierarchy comparison
    """
    try:
        # Make baseline predictions
        Y_pred_baseline = model.predict(X_test)

        # Get probabilities for hierarchy processing
        try:
            proba_list = model.predict_proba(X_test)
        except Exception as e:
            logging.warning("predict_proba failed (%s); hierarchy evaluation skipped", e)
            proba_list = None

        results = []
        hierarchy_results = []

        # Baseline evaluation
        for i, col in enumerate(category_names):
            report = classification_report(
                Y_test[:, i], Y_pred_baseline[:, i], output_dict=True, zero_division=0
            )
            for output_class, metrics in report.items():
                if isinstance(metrics, dict):
                    temp = metrics.copy()
                    temp["output_class"] = output_class
                    temp["category"] = col
                    temp["evaluation_type"] = "baseline"
                    results.append(temp)

        # Hierarchy-corrected evaluation (if probabilities available)
        if proba_list is not None:
            logging.info("Performing hierarchy-corrected evaluation...")

            # Convert probabilities to dict format expected by apply_hierarchy
            n_samples = len(X_test)
            Y_pred_hierarchy = np.zeros_like(Y_pred_baseline)

            violations_before = 0
            violations_after = 0
            edges_before = 0
            edges_after = 0
            skipped_samples_missing_proba = 0

            def _count_edges(prob_map, taxonomy, exclude) -> int:
                total = 0
                for parent, children in taxonomy.items():
                    if parent == "related" or parent in exclude:
                        continue
                    if parent not in prob_map:
                        continue
                    for child in children:
                        if child in exclude or child not in prob_map:
                            continue
                        total += 1
                return total

            # Prepare thresholds used (apply critical-label reduction once)
            base_thresholds = {name: 0.5 for name in category_names}
            thresholds_used = base_thresholds.copy()
            if HIERARCHY_CRITICAL_THRESHOLD_REDUCTION > 0:
                for lbl in CRITICAL_LABELS:
                    if lbl in thresholds_used:
                        thresholds_used[lbl] = max(
                            0.0, thresholds_used[lbl] - HIERARCHY_CRITICAL_THRESHOLD_REDUCTION
                        )

            # Process each sample
            for sample_idx in range(n_samples):
                # Build probability dict for this sample
                probs = {}
                proba_complete = True
                for label_idx, label_name in enumerate(category_names):
                    try:
                        proba_array = proba_list[label_idx]
                        if proba_array.ndim == 2 and proba_array.shape[1] == 2:
                            prob = proba_array[sample_idx, 1]  # Positive class probability
                        elif proba_array.ndim == 2 and proba_array.shape[1] == 1:
                            # Single class present - check which class it is
                            clf = model.named_steps['clf']
                            if hasattr(clf, 'classes_') and label_idx < len(clf.classes_):
                                classes = clf.classes_[label_idx]
                                if len(classes) == 1 and classes[0] == 0:
                                    # Only class 0 present, probability of class 1 is 0
                                    prob = 0.0
                                elif len(classes) == 1 and classes[0] == 1:
                                    # Only class 1 present, probability of class 1 is 1
                                    prob = 1.0
                                else:
                                    # Fallback
                                    prob = proba_array[sample_idx, 0]
                            else:
                                # Fallback if class info not available
                                prob = proba_array[sample_idx, 0]
                        else:
                            # Handle other shapes (1D or unexpected)
                            prob = proba_array[sample_idx]
                        probs[label_name] = float(prob)
                    except Exception:
                        # Mark as incomplete and avoid mixing hard labels with probabilities
                        proba_complete = False
                        break

                if not proba_complete:
                    # Fall back to baseline for this sample, do not include in hierarchy metric counts
                    skipped_samples_missing_proba += 1
                    Y_pred_hierarchy[sample_idx, :] = Y_pred_baseline[sample_idx, :]
                    continue

                # Count violations before hierarchy (only when probabilities are complete)
                violations_before += count_violations(probs, TAXONOMY, EXCLUDE_FROM_CONSTRAINTS)
                edges_before += _count_edges(probs, TAXONOMY, EXCLUDE_FROM_CONSTRAINTS)

                # Apply hierarchy correction
                adjusted_probs, binary_predictions = apply_hierarchy(
                    probs=probs,
                    thresholds=thresholds_used,
                    taxonomy=TAXONOMY,
                    critical_labels=CRITICAL_LABELS,
                    exclude=EXCLUDE_FROM_CONSTRAINTS,
                    critical_threshold_reduction=0.0,
                )

                # Count violations after hierarchy
                violations_after += count_violations(adjusted_probs, TAXONOMY, EXCLUDE_FROM_CONSTRAINTS)
                edges_after += _count_edges(adjusted_probs, TAXONOMY, EXCLUDE_FROM_CONSTRAINTS)

                # Store hierarchy-corrected predictions
                for label_idx, label_name in enumerate(category_names):
                    Y_pred_hierarchy[sample_idx, label_idx] = binary_predictions.get(label_name, Y_pred_baseline[sample_idx, label_idx])

            # Calculate violation rates per 1k edges
            violations_per_1k_before = (violations_before / edges_before * 1000) if edges_before > 0 else 0.0
            violations_per_1k_after = (violations_after / edges_after * 1000) if edges_after > 0 else 0.0

            logging.info(
                f"Violations per 1k edges - Before: {violations_per_1k_before:.1f}, After: {violations_per_1k_after:.1f}"
            )
            logging.info("Note: 'violations per 1k' is edge-normalized (per parent→child edge).")
            if skipped_samples_missing_proba > 0:
                logging.info("Hierarchy: %d samples used baseline only due to missing probabilities; excluded from edge metrics.", skipped_samples_missing_proba)

            # Hierarchy-corrected evaluation
            for i, col in enumerate(category_names):
                report = classification_report(
                    Y_test[:, i], Y_pred_hierarchy[:, i], output_dict=True, zero_division=0
                )
                for output_class, metrics in report.items():
                    if isinstance(metrics, dict):
                        temp = metrics.copy()
                        temp["output_class"] = output_class
                        temp["category"] = col
                        temp["evaluation_type"] = "hierarchy_corrected"
                        hierarchy_results.append(temp)

        # Combine results
        all_results = results + hierarchy_results
        results_df = pd.DataFrame(all_results)
        results_df = results_df[
            ["category", "evaluation_type", "output_class", "precision", "recall", "f1-score", "support"]
        ]

        # Save to clean model folder location
        os.makedirs(model_dir, exist_ok=True)
        results_file_path = os.path.join(model_dir, "performance_metrics.csv")
        results_df.to_csv(results_file_path, index=False)
        logging.info("Performance metrics saved to: %s", results_file_path)

        # Persist thresholds used for hierarchy (for reproducibility)
        try:
            thresholds_out_path = os.path.join(model_dir, "thresholds_used_hierarchy.json")
            with open(thresholds_out_path, "w", encoding="utf-8") as f:
                json.dump(thresholds_used, f, indent=2)
            logging.info("Hierarchy thresholds saved to: %s", thresholds_out_path)
        except Exception as e:
            logging.warning("Failed to persist hierarchy thresholds: %s", e)

        # Calculate summary statistics with clear across-label metrics
        baseline_df = results_df[results_df['evaluation_type'] == 'baseline']

        # Helper: compute across-label Macro/Weighted scores using positive class ('1') rows
        def _across_label_scores(df: pd.DataFrame):
            pos_rows = df[df['output_class'] == '1']
            # Ensure only known categories are considered
            pos_rows = pos_rows[pos_rows['category'].isin(category_names)]
            # Macro over labels: simple mean of label-wise F1/precision/recall
            macro_precision = pos_rows['precision'].mean() if not pos_rows.empty else 0.0
            macro_recall = pos_rows['recall'].mean() if not pos_rows.empty else 0.0
            macro_f1 = pos_rows['f1-score'].mean() if not pos_rows.empty else 0.0
            # Weighted over labels: weights by positive support per label
            weights = pos_rows['support'].astype(float)
            total_w = float(weights.sum()) if not pos_rows.empty else 0.0
            if total_w > 0:
                weighted_precision = float((pos_rows['precision'] * weights).sum() / total_w)
                weighted_recall = float((pos_rows['recall'] * weights).sum() / total_w)
                weighted_f1 = float((pos_rows['f1-score'] * weights).sum() / total_w)
            else:
                weighted_precision = weighted_recall = weighted_f1 = 0.0
            return {
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1,
                'weighted_precision': weighted_precision,
                'weighted_recall': weighted_recall,
                'weighted_f1': weighted_f1,
            }

        baseline_across = _across_label_scores(baseline_df)

        summary = {
            # Legacy overall rows (per-label weighted avg across classes, then mean) kept for continuity
            'total_categories': len(category_names),
            'test_samples': len(Y_test),
            # Across-label metrics (primary/secondary gates)
            'macro_precision_baseline': baseline_across['macro_precision'],
            'macro_recall_baseline': baseline_across['macro_recall'],
            'macro_f1_baseline': baseline_across['macro_f1'],
            'weighted_precision_baseline': baseline_across['weighted_precision'],
            'weighted_recall_baseline': baseline_across['weighted_recall'],
            'weighted_f1_baseline': baseline_across['weighted_f1'],
        }

        # Add hierarchy comparison if available
        if proba_list is not None and hierarchy_results:
            hierarchy_df = results_df[results_df['evaluation_type'] == 'hierarchy_corrected']
            hierarchy_across = _across_label_scores(hierarchy_df)

            # Calculate Safety Recall (average recall on critical labels)
            critical_recalls_baseline = []
            critical_recalls_hierarchy = []

            for label in CRITICAL_LABELS:
                if label in category_names:
                    baseline_recall = baseline_df[(baseline_df['category'] == label) &
                                                (baseline_df['output_class'] == '1')]['recall']
                    hierarchy_recall = hierarchy_df[(hierarchy_df['category'] == label) &
                                                   (hierarchy_df['output_class'] == '1')]['recall']

                    if not baseline_recall.empty:
                        critical_recalls_baseline.append(baseline_recall.iloc[0])
                    if not hierarchy_recall.empty:
                        critical_recalls_hierarchy.append(hierarchy_recall.iloc[0])

            safety_recall_baseline = np.mean(critical_recalls_baseline) if critical_recalls_baseline else 0.0
            safety_recall_hierarchy = np.mean(critical_recalls_hierarchy) if critical_recalls_hierarchy else 0.0

            summary.update({
                # Across-label metrics after hierarchy
                'macro_precision_hierarchy': hierarchy_across['macro_precision'],
                'macro_recall_hierarchy': hierarchy_across['macro_recall'],
                'macro_f1_hierarchy': hierarchy_across['macro_f1'],
                'weighted_precision_hierarchy': hierarchy_across['weighted_precision'],
                'weighted_recall_hierarchy': hierarchy_across['weighted_recall'],
                'weighted_f1_hierarchy': hierarchy_across['weighted_f1'],
                # Deltas
                'macro_f1_change': hierarchy_across['macro_f1'] - baseline_across['macro_f1'],
                'weighted_f1_change': hierarchy_across['weighted_f1'] - baseline_across['weighted_f1'],
                # Safety + violations
                'violations_per_1k_before': violations_per_1k_before,
                'violations_per_1k_after': violations_per_1k_after,
                'safety_recall_baseline': safety_recall_baseline,
                'safety_recall_hierarchy': safety_recall_hierarchy,
                'safety_recall_improvement': safety_recall_hierarchy - safety_recall_baseline,
            })

            logging.info(
                f"Safety Recall: {safety_recall_baseline:.3f} → {safety_recall_hierarchy:.3f} (Δ{(safety_recall_hierarchy - safety_recall_baseline):+.3f})"
            )
            logging.info(
                f"Macro F1 (across labels) Change: {summary['macro_f1_change']:+.3f}"
            )

        return summary

    except Exception as e:
        logging.error("Error evaluating model: %s", e)
        return {}


def save_training_log(model_dir, config, performance_summary, training_time, model_path):
    """Save training metadata to clean JSON log."""
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'training_time_seconds': training_time,
        'configuration': config,
        'performance': performance_summary,
        'version': '1.0',
        'status': 'production_ready'
    }
    
    log_path = os.path.join(model_dir, 'training_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2)
    
    logging.info("Training log saved to: %s", log_path)
    return log_path


def _compute_f2_thresholds_for_labels(model, X_eval, Y_eval, labels, all_category_names):
    try:
        proba_list = model.predict_proba(X_eval)
    except Exception as e:
        logging.warning("predict_proba failed (%s); returning default thresholds=0.5", e)
        return {name: 0.5 for name in labels}, {name: "default" for name in labels}

    thresholds = {}
    sources = {}
    beta = 2.0
    eps = 1e-12
    name_to_idx = {name: i for i, name in enumerate(all_category_names)}
    for name in labels:
        idx = name_to_idx.get(name)
        if idx is None:
            thresholds[name] = 0.5
            sources[name] = "default"
            continue
        y_true = Y_eval[:, idx]
        if np.sum(y_true) == 0:
            thresholds[name] = 0.5
            sources[name] = "default"
            continue
        try:
            probs = proba_list[idx]
            if probs.ndim == 2 and probs.shape[1] == 2:
                # Normal binary classifier with both classes
                p = probs[:, 1]
            elif probs.ndim == 2 and probs.shape[1] == 1:
                # Single class present - check which class it is
                # Access the underlying classifier to get class information
                clf = model.named_steps['clf']
                if hasattr(clf, 'classes_') and idx < len(clf.classes_):
                    classes = clf.classes_[idx]
                    if len(classes) == 1 and classes[0] == 0:
                        # Only class 0 present, probability of class 1 is 0
                        p = np.zeros(probs.shape[0])
                    elif len(classes) == 1 and classes[0] == 1:
                        # Only class 1 present, probability of class 1 is 1
                        p = np.ones(probs.shape[0])
                    else:
                        # Fallback (shouldn't happen)
                        p = probs.ravel()
                else:
                    # Fallback if class info not available
                    p = probs.ravel()
            else:
                # Fallback for unexpected shapes
                p = probs.ravel()
        except Exception:
            thresholds[name] = 0.5
            sources[name] = "default"
            continue

        best_t = 0.5
        best_f = -1.0
        candidates = np.unique(np.clip(p, 0.0, 1.0))
        if candidates.size > 200:
            q = np.linspace(0.05, 0.95, 19)
            candidates = np.unique(np.concatenate([np.quantile(p, q), [0.5]]))
        else:
            candidates = np.unique(np.concatenate([candidates, [0.5]]))
        for t in candidates:
            y_pred = (p >= float(t)).astype(int)
            tp = float(np.sum((y_pred == 1) & (y_true == 1)))
            fp = float(np.sum((y_pred == 1) & (y_true == 0)))
            fn = float(np.sum((y_pred == 0) & (y_true == 1)))
            prec = tp / (tp + fp + eps)
            rec = tp / (tp + fn + eps)
            f = (1 + beta ** 2) * (prec * rec) / (beta ** 2 * prec + rec + eps)
            if f > best_f:
                best_f = f
                best_t = float(t)
        if best_f <= 0:
            thresholds[name] = 0.5
            sources[name] = "default"
        else:
            thresholds[name] = round(best_t, 4)
            sources[name] = "optimized"
    return thresholds, sources


def _json_safe(obj):
    import collections.abc
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, collections.abc.Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in list(obj)]
    if callable(obj):
        return f"<function:{obj.__name__}>" if hasattr(obj, '__name__') else "<function:unknown>"
    return str(obj)


def main():
    parser = argparse.ArgumentParser(
        description='Create production disaster response classification model with clean results structure.'
    )

    parser.add_argument('--db', dest='database_filepath', 
                       default='data/02_stg/stg_disaster_response.db',
                       help='Path to SQLite database (default: data/02_stg/stg_disaster_response.db)')
    parser.add_argument('--params', dest='params_path', 
                       default='model/parameters.json',
                       help='Path to hyperparameters JSON (default: model/parameters.json)')
    parser.add_argument('--class-weights', dest='class_weights_path',
                       default='model/class_weights.json', 
                       help='Path to class weights JSON (default: model/class_weights.json)')
    parser.add_argument('--output', dest='model_out', 
                       default='model/disaster_rf_v1-2-0_prod_2025-09-11.pkl',
                       help='Output model path (default: model/disaster_rf_v1-2-0_prod_2025-09-11.pkl)')
    parser.add_argument('--test-size', dest='test_size', type=float, default=DEFAULT_TEST_SIZE,
                       help=f'Test size fraction (default: {DEFAULT_TEST_SIZE})')
    parser.add_argument('--seed', dest='seed', type=int, default=DEFAULT_RANDOM_SEED,
                       help=f'Random seed (default: {DEFAULT_RANDOM_SEED})')
    parser.add_argument('--eval-ids', dest='eval_ids_path', default=None,
                       help='Path to eval UIDs file (JSON or CSV); if not provided, defaults to experiments/experimental_configs/eval_sets/eval_ids.json if present')
    parser.add_argument('--no-frozen-eval', dest='no_frozen_eval', action='store_true',
                       help='Force random split even if an eval IDs file exists')

    args = parser.parse_args()

    setup_logging()
    
    print(f"\nCreating Production Disaster Response Model")
    print(f"{'='*60}")
    print(f"Database: {args.database_filepath}")
    print(f"Hyperparameters: {args.params_path}")
    print(f"Class weights: {args.class_weights_path}")
    print(f"Output: {args.model_out}")
    print(f"Results will be saved to model/ directory for clarity")
    print(f"{'='*60}")

    # Load data
    logging.info('Loading data...')
    X, Y = load_data(args.database_filepath)
    if X is None or Y is None:
        logging.error('Failed to load data. Exiting.')
        sys.exit(1)

    logging.info(f'Loaded {len(X)} samples with {Y.shape[1]} labels')

    # Determine split mode (frozen eval vs random)
    eval_ids_file = None
    if not args.no_frozen_eval:
        # Prefer explicit path; otherwise default to conventional location
        candidate = args.eval_ids_path or os.path.join('experiments', 'experimental_configs', 'eval_sets', 'eval_ids.json')
        if os.path.isfile(candidate):
            eval_ids_file = candidate

    def _compute_uids(messages):
        uids_local = []
        for idx, msg in enumerate(messages):
            text = '' if msg is None else str(msg)
            uid_src = f"{text}|{idx}"
            uids_local.append(hashlib.sha1(uid_src.encode('utf-8')).hexdigest())
        return uids_local

    if eval_ids_file:
        logging.info('Using frozen eval set from %s', eval_ids_file)
        try:
            # Support both JSON and legacy CSV formats
            if eval_ids_file.endswith('.json'):
                with open(eval_ids_file, 'r') as f:
                    data = json.load(f)
                eval_uids = set(data['eval_ids'])
            else:
                # Legacy CSV format
                eval_df = pd.read_csv(eval_ids_file)
                eval_uids = set(eval_df['uid'].astype(str).tolist())
        except Exception as e:
            logging.error('Failed to read eval IDs file: %s', e)
            sys.exit(1)

        uids = _compute_uids(X)
        uid_series = pd.Series(uids)
        is_eval = uid_series.isin(eval_uids).values

        match_count = int(is_eval.sum())
        expected_eval = int(len(X) * args.test_size)
        if match_count == 0 or match_count < max(1, int(0.5 * expected_eval)):
            logging.error('Eval IDs coverage too low (matched %d, expected around %d). Aborting.', match_count, expected_eval)
            sys.exit(1)

        X_train, X_test = X[~is_eval], X[is_eval]
        Y_train, Y_test = Y[~is_eval], Y[is_eval]
        logging.info('Split via frozen eval set. Train: %d, Eval: %d', len(X_train), len(X_test))
    else:
        # Random split fallback
        logging.info(f'Splitting data randomly (test_size={args.test_size}, seed={args.seed})...')
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=args.test_size, random_state=args.seed
        )

    # Load hyperparameters
    logging.info(f'Loading hyperparameters from {args.params_path}')
    parameters = load_model_parameters(args.params_path)
    if parameters is None:
        logging.error(f'Failed to load hyperparameters from {args.params_path}.')
        if args.params_path == 'model/parameters.json':
            logging.error('')
            logging.error('Note: model/parameters.json was removed on 2026-01-22.')
            logging.error('Please provide --params argument, e.g.:')
            logging.error('  --params experiments/model_candidates/vocab_15k.json')
            logging.error('')
            logging.error('For LogisticRegression models, use scripts/03_create_experimental_model.py instead.')
        sys.exit(1)

    # Load class weights configuration
    logging.info(f'Loading class weights configuration from {args.class_weights_path}')
    class_weights_config = load_class_weights_config(args.class_weights_path)
    if class_weights_config is None:
        logging.error(f'Failed to load class weights config from {args.class_weights_path}.')
        if args.class_weights_path == 'model/class_weights.json':
            logging.error('')
            logging.error('Note: model/class_weights.json was removed on 2026-01-22.')
            logging.error('Please provide --class-weights argument, e.g.:')
            logging.error('  --class-weights experiments/model_candidates/class_weights.json')
            logging.error('')
            logging.error('For LogisticRegression models, use scripts/03_create_experimental_model.py instead.')
        sys.exit(1)

    # Determine if class weighting is enabled
    class_weights_enabled = class_weights_config.get('class_weights', {}).get('enabled', False)
    
    # Create pipeline based on class weights configuration
    if class_weights_enabled:
        logging.info('Creating pipeline with class weighting enabled...')
        
        # Calculate class weights
        class_weights = get_multilabel_class_weights(Y_train, strategy='balanced')
        if class_weights:
            logging.info(f'Calculated class weights for {len(class_weights)} labels')
        
        pipeline = create_pipeline_with_custom_weights()
    else:
        logging.info('Creating pipeline without class weighting (default)...')
        pipeline = create_pipeline(use_class_weights=False)
    
    if pipeline is None:
        logging.error('Failed to create pipeline. Exiting.')
        sys.exit(1)

    # Build model
    logging.info('Building model with hyperparameters...')
    model = build_model(pipeline, parameters)
    if model is None:
        logging.error('Failed to build model. Exiting.')
        sys.exit(1)

    # Train model
    logging.info('Training model...')
    train_start = time()
    
    model.fit(X_train, Y_train)
    
    train_time = time() - train_start
    logging.info(f'Model training completed in {train_time:.2f} seconds')

    # Guardrail: if artifact would be >200MB, refit with max_leaf_nodes=10000
    try:
        tmp_path = os.path.join('model', '_tmp_size_check.pkl')
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        save_model(model, tmp_path)
        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        os.remove(tmp_path)
        if size_mb > 200:
            logging.info('Model size %.1f MB exceeds 200MB; refitting with max_leaf_nodes=10000', size_mb)
            try:
                model.set_params(clf__estimator__max_leaf_nodes=10000)
                train_start = time()
                model.fit(X_train, Y_train)
                train_time = time() - train_start
                logging.info('Refit completed in %.2f seconds', train_time)
            except Exception as refit_exc:
                logging.warning('Refit with max_leaf_nodes failed: %s', refit_exc)
    except Exception as size_exc:
        logging.warning('Size guardrail check failed: %s', size_exc)

    # Evaluate model and save to model folder
    logging.info('Evaluating model and saving results to model/ directory...')
    model_dir = os.path.dirname(args.model_out) or "model"
    performance_summary = evaluate_model_to_model_folder(
        model, X_test, Y_test, TARGET_COLUMNS, model_dir
    )

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    logging.info(f'Saving model to {args.model_out}')
    save_model(model, args.model_out)

    # Compute cold-load time and model size for summary
    model_size_mb = 0.0
    cold_load_s = None
    try:
        model_size_mb = os.path.getsize(args.model_out) / (1024 * 1024)
    except Exception:
        pass
    try:
        t0 = time()
        _ = joblib.load(args.model_out)
        cold_load_s = time() - t0
    except Exception:
        pass

    # Compute thresholds for selected labels and save artifacts
    selected_labels = ['medical_help', 'search_and_rescue', 'water', 'food', 'shelter', 'hospitals', 'security', 'weather_related']
    thresholds_map, threshold_sources = _compute_f2_thresholds_for_labels(model, X_test, Y_test, selected_labels, TARGET_COLUMNS)
    label_order = list(TARGET_COLUMNS)
    try:
        with open(os.path.join(model_dir, 'thresholds.json'), 'w', encoding='utf-8') as f:
            json.dump(thresholds_map, f, indent=2)
        with open(os.path.join(model_dir, 'label_order.json'), 'w', encoding='utf-8') as f:
            json.dump(label_order, f, indent=2)
        info = {
            'sha256': hashlib.sha256(open(args.model_out, 'rb').read()).hexdigest() if os.path.isfile(args.model_out) else None,
            'rf_params': _json_safe(getattr(model.named_steps.get('clf').estimator, 'get_params', lambda: {})()),
            'vectorizer_params': _json_safe(getattr(model.named_steps.get('vect'), 'get_params', lambda: {})()),
            'label_order_hash': hashlib.sha1(json.dumps(label_order).encode('utf-8')).hexdigest(),
            'fit_time_seconds': float(train_time) if train_time is not None else None,
            'model_size_mb': float(model_size_mb) if model_size_mb is not None else None,
            'cold_load_seconds': float(cold_load_s) if cold_load_s is not None else None,
            'threshold_sources': _json_safe(threshold_sources),
        }
        with open(os.path.join(model_dir, 'MODEL_INFO.json'), 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)
    except Exception as e:
        logging.warning("Failed to write model artifacts (thresholds/label_order/MODEL_INFO): %s", e)

    # Create comprehensive config for logging
    comprehensive_config = {
        'hyperparameters': parameters,
        'class_weighting': {
            'enabled': class_weights_enabled,
            'strategy': 'balanced' if class_weights_enabled else None
        },
        'data_split': {
            'test_size': args.test_size,
            'random_seed': args.seed,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'mode': 'frozen_eval' if eval_ids_file else 'random_split',
            'eval_ids_file': eval_ids_file
        },
        'target_labels': len(TARGET_COLUMNS)
    }

    # Save training log
    training_log_path = save_training_log(
        model_dir, comprehensive_config, performance_summary, train_time, args.model_out
    )

    # Success summary
    print(f'\nProduction Model Created Successfully!')
    print(f"{'='*60}")
    print(f'Model: {args.model_out}')
    print(f'Performance: {model_dir}/performance_metrics.csv')
    print(f'Training Log: {training_log_path}')
    print(f'Training Time: {train_time:.2f} seconds')
    print(f"{'='*60}")
    
    print(f'\nPerformance Summary:')
    print(f'   Overall F1-Score: {performance_summary.get("overall_f1", 0):.4f}')
    print(f'   Overall Recall: {performance_summary.get("overall_recall", 0):.4f}')  
    print(f'   Overall Precision: {performance_summary.get("overall_precision", 0):.4f}')
    print(f'   Positive Class F1: {performance_summary.get("positive_class_f1", 0):.4f}')
    
    class_weighting_status = "enabled" if class_weights_enabled else "disabled"
    print(f'\nClass Weighting: {class_weighting_status}')
    if class_weights_enabled:
        print('   Model uses balanced class weights for improved minority class detection')
    
    print(f'\nResults Structure:')
    print(f'   model/disaster_rf_v1-2-0_prod_2025-09-11.pkl          <- Current production model')
    print(f'   model/performance_metrics.csv <- Current model performance')  
    print(f'   model/training_log.json       <- Training metadata & config')
    print(f'\nThis clear structure makes it easy to find current model results!')


if __name__ == '__main__':
    main()
