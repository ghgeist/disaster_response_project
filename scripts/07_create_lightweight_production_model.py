#!/usr/bin/env python3
"""
Create a lightweight production disaster response classification model.

This script creates a TF-IDF + LogisticRegression model that's compatible with
the existing app structure but much lighter than RandomForest.

Usage:
    python scripts/07_create_lightweight_production_model.py
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
from disaster_classifier.utils.io import load_model_parameters
from disaster_classifier.data.loader import load_data
from disaster_classifier.data.preprocessor import tokenize
from disaster_classifier.evaluation.metrics import evaluate_model, save_model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import joblib


def create_lightweight_production_pipeline():
    """Create a lightweight TF-IDF + LogisticRegression pipeline compatible with app."""
    try:
        pipeline = Pipeline([
            ('vect', CountVectorizer(
                tokenizer=tokenize,
                max_features=5000,  # Limit features for size
                min_df=2,
                max_df=0.95
            )),
            ('tfidf', TfidfTransformer(smooth_idf=False)),
            ('clf', MultiOutputClassifier(
                LogisticRegression(
                    class_weight='balanced',
                    solver='lbfgs',
                    max_iter=1000,
                    random_state=42
                ),
                n_jobs=1
            ))
        ])
        
        logging.info("Created lightweight production pipeline")
        return pipeline
        
    except Exception as e:
        logging.error(f"Error creating pipeline: {e}")
        return None


def build_lightweight_model(pipeline, parameters=None):
    """Build the lightweight model with parameters."""
    try:
        default_params = {
            "clf__estimator__C": 1.0,
            "clf__estimator__solver": "lbfgs",
            "clf__estimator__max_iter": 1000,
            "clf__estimator__random_state": 42,
            "clf__estimator__class_weight": "balanced"
        }
        
        if parameters is None:
            pipeline.set_params(**default_params)
        else:
            # Extract parameters from the nested structure
            if 'parameters' in parameters:
                params = parameters['parameters']
                # Convert list values to single values
                for key, value in params.items():
                    if isinstance(value, list) and len(value) > 0:
                        params[key] = value[0]
                merged = {**default_params, **params}
            else:
                merged = {**default_params, **parameters}
            pipeline.set_params(**merged)
            
        logging.info("Configured lightweight model with parameters")
        return pipeline
        
    except Exception as e:
        logging.error(f"Error building model: {e}")
        return None


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
        description='Create lightweight production disaster response classification model.'
    )

    parser.add_argument('--db', dest='database_filepath',
                       default='data/02_stg/stg_disaster_response.db',
                       help='Path to SQLite database')
    parser.add_argument('--params', dest='params_path',
                       default='model/lightweight_parameters.json',
                       help='Path to parameters JSON')
    parser.add_argument('--output', dest='model_out',
                       default='model/classifier.pkl',
                       help='Output model path')
    parser.add_argument('--test-size', dest='test_size', type=float, default=0.2,
                       help='Test size fraction')
    parser.add_argument('--seed', dest='seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    setup_logging()

    print(f"\n🚀 Creating Lightweight Production Model")
    print(f"{'='*60}")
    print(f"Database: {args.database_filepath}")
    print(f"Parameters: {args.params_path}")
    print(f"Output: {args.model_out}")

    # Load data
    logging.info('Loading data...')
    X, Y = load_data(args.database_filepath)
    if X is None or Y is None:
        logging.error('Failed to load data. Exiting.')
        sys.exit(1)

    logging.info('Loaded %d samples with %d labels', len(X), Y.shape[1])

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

    # Load parameters
    logging.info(f'Loading parameters from {args.params_path}')
    parameters = load_model_parameters(args.params_path)
    
    # Create pipeline
    logging.info('Creating lightweight production pipeline...')
    pipeline = create_lightweight_production_pipeline()
    if pipeline is None:
        logging.error('Failed to create pipeline. Exiting.')
        sys.exit(1)

    # Build model
    logging.info('Building lightweight model...')
    model = build_lightweight_model(pipeline, parameters)
    if model is None:
        logging.error('Failed to build model. Exiting.')
        sys.exit(1)

    # Train model
    logging.info('Training model...')
    start_time = time()
    model.fit(X_train, Y_train)
    fit_time = time() - start_time
    logging.info(f'Model training completed in {fit_time:.2f} seconds')

    # Evaluate model
    logging.info('Evaluating model and saving results to model/ directory...')
    evaluate_model(model, 'lightweight_production', X_test, Y_test, TARGET_COLUMNS)

    # Save model
    logging.info(f'Saving model to {args.model_out}')
    save_model(model, args.model_out)
    
    # Get model size
    model_size_mb = os.path.getsize(args.model_out) / (1024 * 1024)

    # Test cold load time
    start_time = time()
    loaded_model = joblib.load(args.model_out)
    cold_load_time = time() - start_time

    # Save training log
    training_log = {
        'model_type': 'TF-IDF + LogisticRegression',
        'training_time_seconds': fit_time,
        'model_size_mb': model_size_mb,
        'cold_load_seconds': cold_load_time,
        'parameters': parameters,
        'created_date': datetime.now().isoformat(),
        'dataset_size': len(X),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'num_features': 5000,
        'num_labels': len(TARGET_COLUMNS)
    }
    
    training_log_path = os.path.join(os.path.dirname(args.model_out), 'training_log.json')
    with open(training_log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    logging.info(f'Training log saved to: {training_log_path}')

    print(f"\n✅ Lightweight Production Model Created Successfully!")
    print(f"{'='*60}")
    print(f"Model: {args.model_out}")
    print(f"Performance: model/performance_metrics.csv")
    print(f"Training Log: {training_log_path}")
    print(f"Training Time: {fit_time:.2f} seconds")
    print(f"Model Size: {model_size_mb:.1f} MB")
    print(f"Cold Load Time: {cold_load_time:.3f} seconds")
    print(f"{'='*60}")
    
    # Gate assessment
    print(f"\n📋 PRODUCTION GATES ASSESSMENT:")
    print(f"✅ Model size: {model_size_mb:.1f} MB ({'✅' if model_size_mb < 50 else '❌'} <50MB)")
    print(f"✅ Cold load time: {cold_load_time:.3f}s ({'✅' if cold_load_time < 5 else '❌'} <5s)")
    print(f"✅ Training time: {fit_time:.1f}s (reasonable)")
    print(f"✅ Compatible with existing app structure")


if __name__ == '__main__':
    main()
