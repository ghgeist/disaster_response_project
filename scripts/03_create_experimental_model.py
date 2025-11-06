#!/usr/bin/env python3
"""
Create an experimental disaster response classification model.

This script mirrors the production model creation but uses experimental
candidate configurations and stores outputs under the experiments folder:

- experiments/model_candidates/parameters.json (hyperparameters)
- experiments/model_candidates/class_weights.json (class weights config)
- experiments/results/ (outputs)
    - classifier.pkl
    - performance_metrics.csv
    - training_log.json

Usage:
    python scripts/03_create_experimental_model.py
    python scripts/03_create_experimental_model.py --output experiments/results/experimental_classifier.pkl
"""

import argparse
import os
import sys
import logging
import json
from datetime import datetime
from time import time
import hashlib
import shutil

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from disasterproject.utils.config import setup_logging, TARGET_COLUMNS
from disasterproject.utils.json_io import load_model_parameters
from disasterproject.utils.experimental_paths import ExperimentalPathManager
from disasterproject.data.loader import load_data
from disasterproject.models.pipeline import (
    create_pipeline,
    create_pipeline_with_custom_weights,
    build_model
)
from disasterproject.models.samplers import get_multilabel_class_weights
from disasterproject.evaluation.metrics import evaluate_model, save_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import pandas as pd
import numpy as np
import joblib


def generate_model_filename(params_file_path):
    """
    Generate model filename based on parameters file following naming convention.

    Example:
    Input:  "2025-09-16-comprehensive-grid-search-optimized-hyperparameters.json"
    Output: "2025-09-16-comprehensive-grid-search-optimized-model.pkl"
    """
    params_filename = os.path.basename(params_file_path)
    base_name = os.path.splitext(params_filename)[0]  # Remove .json

    # Replace "optimized-hyperparameters" with "optimized-model"
    if "optimized-hyperparameters" in base_name:
        model_name = base_name.replace("optimized-hyperparameters", "optimized-model")
    else:
        # Fallback for other naming patterns
        model_name = f"{base_name}-model"

    return f"{model_name}.pkl"


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


def evaluate_model_to_experiment_folder(model, X_test, Y_test, category_names, experiment_dir):
    """
    Evaluate model and save results to dated experiment folder.

    Args:
        model: Trained model
        X_test: Test features
        Y_test: Test labels
        category_names: List of category names
        experiment_dir: Experiment directory to save results

    Returns:
        dict: Performance summary
    """
    try:
        # Make predictions
        Y_pred = model.predict(X_test)
        results = []

        # Generate detailed classification report for each category
        for i, col in enumerate(category_names):
            report = classification_report(
                Y_test[:, i], Y_pred[:, i], output_dict=True, zero_division=0
            )
            for output_class, metrics in report.items():
                if isinstance(metrics, dict):
                    temp = metrics.copy()
                    temp["output_class"] = output_class
                    temp["category"] = col
                    results.append(temp)

        results_df = pd.DataFrame(results)
        results_df = results_df[
            ["category", "output_class", "precision", "recall", "f1-score", "support"]
        ]

        # Save to experiment folder
        os.makedirs(experiment_dir, exist_ok=True)
        results_file_path = os.path.join(experiment_dir, "performance_metrics.csv")
        results_df.to_csv(results_file_path, index=False)
        logging.info("Performance metrics saved to: %s", results_file_path)

        # Calculate summary statistics
        weighted_avg = results_df[results_df['output_class'] == 'weighted avg']
        positive_class = results_df[results_df['output_class'] == '1']

        # True across-label metrics (micro/samples) for promotion gating
        try:
            micro_f1 = float(f1_score(Y_test, Y_pred, average="micro", zero_division=0))
        except Exception:
            micro_f1 = None
        try:
            samples_f1 = float(f1_score(Y_test, Y_pred, average="samples", zero_division=0))
        except Exception:
            samples_f1 = None

        summary = {
            'overall_precision': weighted_avg['precision'].mean(),
            'overall_recall': weighted_avg['recall'].mean(),
            'overall_f1': weighted_avg['f1-score'].mean(),
            'positive_class_precision': positive_class['precision'].mean(),
            'positive_class_recall': positive_class['recall'].mean(),
            'positive_class_f1': positive_class['f1-score'].mean(),
            'micro_f1': micro_f1,
            'samples_f1': samples_f1,
            'total_categories': len(category_names),
            'test_samples': len(Y_test)
        }

        return summary

    except Exception as e:
        logging.error("Error evaluating model: %s", e)
        return {}


def save_training_log(experiment_dir, config, performance_summary, training_time, model_path):
    """Save training metadata to experiment folder."""
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'training_time_seconds': training_time,
        'configuration': config,
        'performance': performance_summary,
        'version': '0.1-exp',
        'status': 'experimental'
    }

    log_path = os.path.join(experiment_dir, 'training_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2)

    logging.info(f"Training log saved to: {log_path}")
    return log_path


def _compute_f2_thresholds_for_labels(model, X_eval, Y_eval, labels, all_category_names):
    """
    Compute F2-optimized thresholds for the selected labels; fallback to 0.5 when unreliable.
    
    Note: This uses F2 score (beta=2.0) which emphasizes recall over precision.
    For production use, consider using optimize_critical_thresholds() which uses
    precision_recall_curve with target recall instead (see optimize_critical_thresholds_inc1.py).
    """
    try:
        proba_list = model.predict_proba(X_eval)
    except Exception as e:
        logging.warning(f"predict_proba failed ({e}); returning default thresholds=0.5")
        return {name: 0.5 for name in labels}, {name: "default" for name in labels}

    thresholds = {}
    sources = {}
    beta = 2.0
    eps = 1e-12
    # map category to index
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
        # MultiOutputClassifier returns list of arrays, one per label
        try:
            probs = proba_list[idx]
            # shape (n_samples, 2) -> class 1
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
        # candidate thresholds: unique probs clipped plus a grid
        candidates = np.unique(np.clip(p, 0.0, 1.0))
        if candidates.size > 200:
            # subsample to keep compute reasonable
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
    """Convert objects to JSON-serializable forms, stringifying as needed."""
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
        description='Create experimental disaster response classification model with clean results structure.'
    )

    parser.add_argument('--db', dest='database_filepath',
                       default='data/02_stg/stg_disaster_response.db',
                       help='Path to SQLite database (default: data/02_stg/stg_disaster_response.db)')
    parser.add_argument('--params', dest='params_path',
                       default='experiments/model_candidates/parameters.json',
                       help='Path to hyperparameters JSON (default: experiments/model_candidates/parameters.json)')
    parser.add_argument('--class-weights', dest='class_weights_path',
                       default='experiments/model_candidates/class_weights.json',
                       help='Path to class weights JSON (default: experiments/model_candidates/class_weights.json)')
    parser.add_argument('--output', dest='model_out', default=None,
                       help='Output model path (default: auto-generated from params filename)')
    parser.add_argument('--test-size', dest='test_size', type=float, default=0.2,
                       help='Test size fraction (default: 0.2)')
    parser.add_argument('--seed', dest='seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--eval-ids', dest='eval_ids_path', default=None,
                       help='Path to eval UIDs file (JSON or CSV); if not provided, defaults to experiments/experimental_configs/eval_sets/eval_ids.json if present')
    parser.add_argument('--no-frozen-eval', dest='no_frozen_eval', action='store_true',
                       help='Force random split even if an eval IDs file exists')
    parser.add_argument('--algorithm', dest='algorithm',
                       choices=['random_forest', 'logistic_regression'],
                       default='random_forest',
                       help='Algorithm to use (default: random_forest)')

    args = parser.parse_args()

    # Auto-generate output filename if not provided
    if args.model_out is None:
        model_filename = generate_model_filename(args.params_path)
        path_manager = ExperimentalPathManager()
        output_dir = path_manager.get_output_directory()
        args.model_out = os.path.join(output_dir, model_filename)

    setup_logging()

    # Get today's date for experiment folder naming
    date_str = datetime.now().strftime('%Y-%m-%d')
    experiment_dir = os.path.join('experiments', 'experimental_runs', date_str)

    print(f"\nCreating Experimental Disaster Response Model")
    print(f"{'='*60}")
    print(f"Database: {args.database_filepath}")
    print(f"Hyperparameters: {args.params_path}")
    print(f"Class weights: {args.class_weights_path}")
    print(f"Output: {args.model_out}")
    print(f"Experiment results will be saved to: {experiment_dir}")
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
        print(f"Using frozen eval set from {eval_ids_file} (eval samples: {len(X_test)})")
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
        logging.error(f'Failed to load hyperparameters from {args.params_path}. Exiting.')
        sys.exit(1)

    # Filter parameters for LogisticRegression (remove RF-specific params)
    if args.algorithm == 'logistic_regression':
        rf_params = [
            'clf__estimator__n_estimators', 
            'clf__estimator__max_depth',
            'clf__estimator__min_samples_leaf', 
            'clf__estimator__min_samples_split'
        ]
        original_params = parameters.copy()
        parameters = {k: v for k, v in parameters.items() if k not in rf_params}
        removed = [k for k in original_params if k not in parameters]
        if removed:
            logging.info(f'Filtered RF-specific params for LR: {removed}')

    # Load class weights configuration
    logging.info(f'Loading class weights configuration from {args.class_weights_path}')
    class_weights_config = load_class_weights_config(args.class_weights_path)
    if class_weights_config is None:
        logging.error(f'Failed to load class weights config from {args.class_weights_path}. Exiting.')
        sys.exit(1)

    # Determine if class weighting is enabled
    class_weights_enabled = class_weights_config.get('class_weights', {}).get('enabled', False)

    # Import appropriate pipeline functions
    from disasterproject.models.pipeline import (
        create_pipeline_logistic_regression,
        create_pipeline_logistic_regression_weighted
    )

    # Create pipeline based on algorithm and class weights configuration
    if args.algorithm == 'logistic_regression':
        if class_weights_enabled:
            logging.info('Creating weighted LogisticRegression pipeline...')
            # Calculate class weights
            class_weights_list = get_multilabel_class_weights(Y_train, strategy='balanced')
            if class_weights_list:
                logging.info(f'Calculated class weights for {len(class_weights_list)} labels')
            pipeline = create_pipeline_logistic_regression_weighted(class_weights_list=class_weights_list)
        else:
            logging.info('Creating LogisticRegression pipeline (no class weights)...')
            pipeline = create_pipeline_logistic_regression()
    else:
        # RandomForest (original behavior)
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
        tmp_path = os.path.join('experiments', 'results', '_tmp_size_check.pkl')
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

    # Evaluate model and save to experiment folder
    logging.info(f'Evaluating model and saving results to {experiment_dir}...')
    performance_summary = evaluate_model_to_experiment_folder(
        model, X_test, Y_test, TARGET_COLUMNS, experiment_dir
    )

    # Snapshot the eval IDs used for this run (traceability)
    if eval_ids_file:
        try:
            # Copy eval IDs file to experiment directory with appropriate extension
            if eval_ids_file.endswith('.json'):
                target_file = os.path.join(experiment_dir, 'eval_ids_used.json')
            else:
                target_file = os.path.join(experiment_dir, 'eval_ids_used.csv')
            shutil.copyfile(eval_ids_file, target_file)
        except Exception as e:
            logging.warning('Could not snapshot eval IDs file: %s', e)

    # Save model to new experimental structure
    path_manager = ExperimentalPathManager()
    results_dir = os.path.dirname(args.model_out) or path_manager.get_output_directory()
    os.makedirs(results_dir, exist_ok=True)
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

    # Compute thresholds for selected labels and save artifacts to experiment folder
    selected_labels = ['medical_help', 'search_and_rescue', 'water', 'food', 'shelter', 'hospitals', 'security', 'weather_related']
    thresholds_map, threshold_sources = _compute_f2_thresholds_for_labels(model, X_test, Y_test, selected_labels, TARGET_COLUMNS)
    label_order = list(TARGET_COLUMNS)
    try:
        with open(os.path.join(experiment_dir, 'thresholds.json'), 'w', encoding='utf-8') as f:
            json.dump(thresholds_map, f, indent=2)
        with open(os.path.join(experiment_dir, 'label_order.json'), 'w', encoding='utf-8') as f:
            json.dump(label_order, f, indent=2)
        # MODEL_INFO.json for artifact hygiene
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
        with open(os.path.join(experiment_dir, 'MODEL_INFO.json'), 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)
    except Exception as e:
        logging.warning(f"Failed to write model artifacts (thresholds/label_order/MODEL_INFO): {e}")

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
            'eval_ids_file': eval_ids_file,
            'eval_uid_count': int(len(X_test))
        },
        'target_labels': len(TARGET_COLUMNS)
    }

    # Save training log
    training_log_path = save_training_log(
        experiment_dir, comprehensive_config, performance_summary, train_time, args.model_out
    )

    # Success summary
    print(f'\nExperimental Model Created Successfully!')
    print(f"{'='*60}")
    print(f'Model: {args.model_out}')
    print(f'Experiment: {experiment_dir}')
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
        print('   Model uses balanced class weights for experimental evaluation')

    print(f'\nExperiment Structure:')
    print(f'   Experimental Model:')
    print(f'     {args.model_out}')
    print(f'   ')
    print(f'   Experiment Results ({experiment_dir}):')
    print(f'     performance_metrics.csv    <- Detailed classification metrics')
    print(f'     training_log.json         <- Training metadata & configuration')
    print(f'     thresholds.json           <- Optimized F2 thresholds')
    print(f'     label_order.json          <- Category label order')
    print(f'     MODEL_INFO.json           <- Model metadata & info')
    if eval_ids_file:
        eval_file_ext = 'json' if eval_ids_file.endswith('.json') else 'csv'
        print(f'     eval_ids_used.{eval_file_ext}         <- Evaluation set identifiers')
    print(f'\nThis structure organizes all experiment artifacts by date for easy tracking!')


if __name__ == '__main__':
    main()

