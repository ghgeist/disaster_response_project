#!/usr/bin/env python3
"""
Create a rollback disaster response classification model using TF-IDF + LogisticRegression.

This script creates a lightweight, fast-loading model as a fallback when RandomForest
models fail performance gates. Uses TF-IDF vectorization with LogisticRegression
for better recall on minority classes.

Usage:
    python scripts/05_create_rollback_model.py
    python scripts/05_create_rollback_model.py --output experiments/results/rollback_classifier.pkl
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
from disasterproject.data.loader import load_data
from disasterproject.models.pipeline import (
    create_logistic_regression_pipeline,
    build_logistic_model
)
from disasterproject.evaluation.metrics import evaluate_model, save_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import joblib


def load_logistic_parameters(file_path):
    """Load LogisticRegression parameters from JSON file."""
    try:
        with open(file_path, 'r') as f:
            params = json.load(f)
        return params
    except Exception as e:
        logging.error(f'Failed to load parameters from {file_path}: {e}')
        return None


def compute_f2_thresholds(model, X_test, Y_test, target_labels=None):
    """
    Compute F2-optimized thresholds for specified labels.
    
    Args:
        model: Trained model
        X_test: Test features
        Y_test: Test labels
        target_labels: List of label names to optimize thresholds for
        
    Returns:
        dict: Mapping of label names to optimal thresholds
    """
    if target_labels is None:
        target_labels = [
            'medical_help', 'search_and_rescue', 'water', 'food', 
            'shelter', 'hospitals', 'security', 'weather_related'
        ]
    
    # Get prediction probabilities
    y_pred_proba = model.predict_proba(X_test)
    
    thresholds = {}
    
    for i, label in enumerate(TARGET_COLUMNS):
        if label not in target_labels:
            continue
            
        # Get probabilities for positive class
        if hasattr(y_pred_proba[i], '__len__') and len(y_pred_proba[i]) > 1:
            proba_pos = y_pred_proba[i][:, 1]
        else:
            continue
            
        y_true = Y_test[:, i]
        
        # Find optimal threshold using F2 score
        best_f2 = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.01, 0.99, 0.01):
            y_pred = (proba_pos >= threshold).astype(int)
            
            # Calculate F2 score (weights recall more than precision)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            if tp + fp == 0:
                precision = 0
            else:
                precision = tp / (tp + fp)
                
            if tp + fn == 0:
                recall = 0
            else:
                recall = tp / (tp + fn)
            
            if precision + recall == 0:
                f2 = 0
            else:
                # F2 score weights recall 4x more than precision
                f2 = 5 * precision * recall / (4 * precision + recall)
            
            if f2 > best_f2:
                best_f2 = f2
                best_threshold = threshold
        
        thresholds[label] = round(best_threshold, 4)
        logging.info(f"Optimal threshold for {label}: {best_threshold:.4f} (F2: {best_f2:.4f})")
    
    return thresholds


def _compute_uids(messages):
    """Compute stable UIDs for frozen evaluation."""
    uids = []
    for idx, msg in enumerate(messages):
        text = '' if msg is None else str(msg)
        uid_src = f"{text}|{idx}"
        uids.append(hashlib.sha1(uid_src.encode('utf-8')).hexdigest())
    return uids


def _json_safe(obj):
    """Convert object to JSON-safe format."""
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return {k: _json_safe(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return str(obj)


def main():
    parser = argparse.ArgumentParser(
        description='Create rollback disaster response classification model with TF-IDF + LogisticRegression.'
    )

    parser.add_argument('--db', dest='database_filepath',
                       default='data/02_stg/stg_disaster_response.db',
                       help='Path to SQLite database (default: data/02_stg/stg_disaster_response.db)')
    parser.add_argument('--params', dest='params_path',
                       default='experiments/model_candidates/logistic_regression_parameters.json',
                       help='Path to LogisticRegression parameters JSON')
    parser.add_argument('--output', dest='model_out',
                       default='experiments/results/rollback_classifier.pkl',
                       help='Output model path')
    parser.add_argument('--test-size', dest='test_size', type=float, default=0.2,
                       help='Test size fraction (default: 0.2)')
    parser.add_argument('--seed', dest='seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--eval-ids', dest='eval_ids_path', default=None,
                       help='Path to CSV of eval UIDs for frozen evaluation')

    args = parser.parse_args()

    setup_logging()

    print(f"\nCreating Rollback Disaster Response Model (TF-IDF + LogisticRegression)")
    print(f"{'='*70}")
    print(f"Database: {args.database_filepath}")
    print(f"Parameters: {args.params_path}")
    print(f"Output: {args.model_out}")

    # Load data
    logging.info('Loading data from %s', args.database_filepath)
    X, Y = load_data(args.database_filepath)
    if X is None or Y is None:
        logging.error('Failed to load data. Exiting.')
        sys.exit(1)

    logging.info('Loaded %d samples with %d features', len(X), Y.shape[1])

    # Handle frozen evaluation split
    eval_ids_file = args.eval_ids_path or 'data/04_fct/eval_ids.csv'
    
    if os.path.exists(eval_ids_file):
        logging.info('Using frozen eval set from %s', eval_ids_file)
        try:
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

    # Load parameters
    logging.info(f'Loading LogisticRegression parameters from {args.params_path}')
    parameters = load_logistic_parameters(args.params_path)
    
    # Create pipeline
    logging.info('Creating TF-IDF + LogisticRegression pipeline...')
    pipeline = create_logistic_regression_pipeline(use_class_weights=True)
    if pipeline is None:
        logging.error('Failed to create pipeline. Exiting.')
        sys.exit(1)

    # Build model
    logging.info('Building LogisticRegression model...')
    model = build_logistic_model(pipeline, parameters)
    if model is None:
        logging.error('Failed to build model. Exiting.')
        sys.exit(1)

    # Train model
    logging.info('Training model...')
    start_time = time()
    model.fit(X_train, Y_train)
    fit_time = time() - start_time
    logging.info(f'Training completed in {fit_time:.2f} seconds')

    # Save model
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    save_model(model, args.model_out)
    
    # Get model size
    model_size_mb = os.path.getsize(args.model_out) / (1024 * 1024)
    logging.info(f'Model saved to {args.model_out} (size: {model_size_mb:.1f} MB)')

    # Test cold load time
    start_time = time()
    _ = joblib.load(args.model_out)
    cold_load_time = time() - start_time
    logging.info(f'Cold load time: {cold_load_time:.3f} seconds')

    # Evaluate model
    logging.info('Evaluating model...')
    evaluate_model(model, 'rollback_logistic', X_test, Y_test, TARGET_COLUMNS)

    # Compute F2-optimized thresholds for high-impact labels
    logging.info('Computing F2-optimized thresholds...')
    thresholds = compute_f2_thresholds(model, X_test, Y_test)
    
    # Save results to experiments/results/
    results_dir = os.path.dirname(args.model_out)
    
    # Save thresholds
    thresholds_path = os.path.join(results_dir, 'thresholds.json')
    with open(thresholds_path, 'w') as f:
        json.dump(thresholds, f, indent=2)
    logging.info(f'Thresholds saved to {thresholds_path}')
    
    # Save label order
    label_order_path = os.path.join(results_dir, 'label_order.json')
    with open(label_order_path, 'w') as f:
        json.dump(TARGET_COLUMNS, f, indent=2)
    logging.info(f'Label order saved to {label_order_path}')
    
    # Save model info
    model_info = {
        'sha256': hashlib.sha256(open(args.model_out, 'rb').read()).hexdigest(),
        'lr_params': _json_safe(model.get_params()['clf__estimator']),
        'vectorizer_params': _json_safe(model.get_params()['tfidf']),
        'label_order_hash': hashlib.sha1(json.dumps(TARGET_COLUMNS).encode()).hexdigest(),
        'fit_time_seconds': fit_time,
        'model_size_mb': model_size_mb,
        'cold_load_seconds': cold_load_time,
        'threshold_sources': {label: 'optimized' for label in thresholds.keys()}
    }
    
    model_info_path = os.path.join(results_dir, 'MODEL_INFO.json')
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    logging.info(f'Model info saved to {model_info_path}')

    print(f"\n✅ Rollback Model Creation Complete!")
    print(f"Model: {args.model_out}")
    print(f"Size: {model_size_mb:.1f} MB")
    print(f"Training time: {fit_time:.1f}s")
    print(f"Cold load time: {cold_load_time:.3f}s")
    print(f"Thresholds optimized for {len(thresholds)} labels")


if __name__ == '__main__':
    main()
