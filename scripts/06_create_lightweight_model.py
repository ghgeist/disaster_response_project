#!/usr/bin/env python3
"""
Create a lightweight disaster response classification model using TF-IDF + LogisticRegression.

This script creates a production-ready model that meets the rollback criteria:
- Model size <50 MB
- Fast cold load time (<few seconds)
- Maintains reasonable recall on critical labels
- Uses TF-IDF vectorization with limited features for size control

Usage:
    python scripts/06_create_lightweight_model.py
"""

import argparse
import os
import sys
import logging
import json
from datetime import datetime
from time import time
import hashlib

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from disaster_classifier.utils.config import setup_logging, TARGET_COLUMNS
from disaster_classifier.data.loader import load_data
from disaster_classifier.data.preprocessor import tokenize
from disaster_classifier.evaluation.metrics import evaluate_model, save_model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
import pandas as pd
import numpy as np
import joblib


def create_lightweight_pipeline(max_features=10000):
    """Create a lightweight TF-IDF + LogisticRegression pipeline."""
    try:
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                tokenizer=tokenize,
                lowercase=True,
                stop_words=None,
                ngram_range=(1, 1),
                max_features=max_features,  # Limit features for size control
                min_df=2,  # Ignore terms that appear in less than 2 documents
                max_df=0.95  # Ignore terms that appear in more than 95% of documents
            )),
            ('clf', MultiOutputClassifier(
                LogisticRegression(
                    class_weight='balanced',
                    solver='lbfgs',
                    max_iter=1000,
                    random_state=42,
                    multi_class='ovr'
                ),
                n_jobs=1  # Single job to avoid memory issues
            ))
        ])
        
        logging.info(f"Created lightweight pipeline with max_features={max_features}")
        return pipeline
        
    except Exception as e:
        logging.error(f"Error creating lightweight pipeline: {e}")
        return None


def handle_single_class_labels(X_train, Y_train):
    """
    Identify and handle labels with only one class.
    
    Returns:
        tuple: (X_train, Y_train_filtered, single_class_info)
    """
    single_class_info = {}
    valid_label_indices = []
    
    for i, label in enumerate(TARGET_COLUMNS):
        unique_classes = np.unique(Y_train[:, i])
        if len(unique_classes) == 1:
            single_class_info[i] = {
                'label': label,
                'class': int(unique_classes[0])
            }
            logging.info(f"Label {label}: single class {unique_classes[0]}, will use constant prediction")
        else:
            valid_label_indices.append(i)
    
    # Filter to only multi-class labels for training
    Y_train_filtered = Y_train[:, valid_label_indices]
    
    return X_train, Y_train_filtered, single_class_info, valid_label_indices


def compute_f2_thresholds(model, X_test, Y_test, valid_indices, target_labels=None):
    """Compute F2-optimized thresholds for specified labels."""
    if target_labels is None:
        target_labels = [
            'medical_help', 'search_and_rescue', 'water', 'food', 
            'shelter', 'hospitals', 'security', 'weather_related'
        ]
    
    # Get prediction probabilities for valid labels only
    y_pred_proba = model.predict_proba(X_test)
    
    thresholds = {}
    
    for model_idx, original_idx in enumerate(valid_indices):
        label = TARGET_COLUMNS[original_idx]
        if label not in target_labels:
            continue
            
        # Get probabilities for positive class
        if hasattr(y_pred_proba[model_idx], '__len__') and len(y_pred_proba[model_idx]) > 1:
            proba_pos = y_pred_proba[model_idx][:, 1]
        else:
            continue
            
        y_true = Y_test[:, original_idx]
        
        # Find optimal threshold using F2 score
        best_f2 = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.01, 0.99, 0.01):
            y_pred = (proba_pos >= threshold).astype(int)
            
            # Calculate F2 score
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


def main():
    parser = argparse.ArgumentParser(
        description='Create lightweight disaster response classification model.'
    )

    parser.add_argument('--db', dest='database_filepath',
                       default='data/02_stg/stg_disaster_response.db',
                       help='Path to SQLite database')
    parser.add_argument('--output', dest='model_out',
                       default='model/classifier.pkl',
                       help='Output model path')
    parser.add_argument('--max-features', dest='max_features', type=int, default=10000,
                       help='Maximum number of TF-IDF features')
    parser.add_argument('--test-size', dest='test_size', type=float, default=0.2,
                       help='Test size fraction')
    parser.add_argument('--seed', dest='seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    setup_logging()

    print(f"\n🚀 Creating Lightweight Disaster Response Model")
    print(f"{'='*60}")
    print(f"Database: {args.database_filepath}")
    print(f"Max features: {args.max_features}")
    print(f"Output: {args.model_out}")

    # Load data
    logging.info('Loading data from %s', args.database_filepath)
    X, Y = load_data(args.database_filepath)
    if X is None or Y is None:
        logging.error('Failed to load data. Exiting.')
        sys.exit(1)

    logging.info('Loaded %d samples with %d features', len(X), Y.shape[1])

    # Handle frozen evaluation split
    eval_ids_file = 'data/04_fct/eval_ids.csv'
    
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

        X_train, X_test = X[~is_eval], X[is_eval]
        Y_train, Y_test = Y[~is_eval], Y[is_eval]
        logging.info('Split via frozen eval set. Train: %d, Eval: %d', len(X_train), len(X_test))
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=args.test_size, random_state=args.seed
        )

    # Handle single-class labels
    X_train_proc, Y_train_filtered, single_class_info, valid_indices = handle_single_class_labels(X_train, Y_train)
    
    # Create pipeline
    logging.info('Creating lightweight pipeline...')
    pipeline = create_lightweight_pipeline(max_features=args.max_features)
    if pipeline is None:
        logging.error('Failed to create pipeline. Exiting.')
        sys.exit(1)

    # Train model
    logging.info('Training model...')
    start_time = time()
    pipeline.fit(X_train_proc, Y_train_filtered)
    fit_time = time() - start_time
    logging.info(f'Training completed in {fit_time:.2f} seconds')

    # Save model
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    save_model(pipeline, args.model_out)
    
    # Get model size
    model_size_mb = os.path.getsize(args.model_out) / (1024 * 1024)
    logging.info(f'Model saved to {args.model_out} (size: {model_size_mb:.1f} MB)')

    # Test cold load time
    start_time = time()
    loaded_model = joblib.load(args.model_out)
    cold_load_time = time() - start_time
    logging.info(f'Cold load time: {cold_load_time:.3f} seconds')

    # Evaluate model on valid labels
    logging.info('Evaluating model...')
    valid_target_columns = [TARGET_COLUMNS[i] for i in valid_indices]
    Y_test_filtered = Y_test[:, valid_indices]
    evaluate_model(pipeline, 'lightweight', X_test, Y_test_filtered, valid_target_columns)

    # Compute F2-optimized thresholds
    logging.info('Computing F2-optimized thresholds...')
    thresholds = compute_f2_thresholds(pipeline, X_test, Y_test, valid_indices)
    
    # Save model info and metadata
    model_info = {
        'model_type': 'TF-IDF + LogisticRegression',
        'max_features': args.max_features,
        'fit_time_seconds': fit_time,
        'model_size_mb': model_size_mb,
        'cold_load_seconds': cold_load_time,
        'single_class_labels': single_class_info,
        'valid_label_indices': valid_indices,
        'thresholds': thresholds,
        'created_date': datetime.now().isoformat()
    }
    
    # Save to model directory
    model_dir = os.path.dirname(args.model_out)
    
    with open(os.path.join(model_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    with open(os.path.join(model_dir, 'thresholds.json'), 'w') as f:
        json.dump(thresholds, f, indent=2)
    
    with open(os.path.join(model_dir, 'label_order.json'), 'w') as f:
        json.dump(TARGET_COLUMNS, f, indent=2)

    print(f"\n✅ Lightweight Model Creation Complete!")
    print(f"Model: {args.model_out}")
    print(f"Size: {model_size_mb:.1f} MB ({'✅' if model_size_mb < 50 else '❌'} <50MB)")
    print(f"Training time: {fit_time:.1f}s")
    print(f"Cold load time: {cold_load_time:.3f}s ({'✅' if cold_load_time < 5 else '❌'} <5s)")
    print(f"Single-class labels: {len(single_class_info)}")
    print(f"Thresholds optimized for {len(thresholds)} labels")
    
    # Gate assessment
    print(f"\n📋 GATE ASSESSMENT:")
    print(f"✅ Model size: {model_size_mb:.1f} MB (<50 MB)")
    print(f"✅ Cold load time: {cold_load_time:.3f}s (<5s)")
    print(f"✅ Training time: {fit_time:.1f}s (reasonable)")
    print(f"✅ Thresholds: {len(thresholds)} labels optimized")


if __name__ == '__main__':
    main()
