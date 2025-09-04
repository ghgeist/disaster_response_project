#!/usr/bin/env python3
"""
Create a disaster response classification model with class weighting.

This script creates a model with class weighting to handle multi-label class imbalance.
This is the recommended approach for production deployment as it improves minority
class detection without data duplication.

For baseline models, use create_baseline_model.py instead.
For sampling experiments, use test_sampling_strategies.py instead.
"""

import argparse
import os
import sys
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from disaster_classifier.utils.config import setup_logging, TARGET_COLUMNS
from disaster_classifier.utils.io import load_model_parameters
from disaster_classifier.data.loader import load_data
from disaster_classifier.models.pipeline import (
    create_pipeline, 
    create_pipeline_with_custom_weights,
    build_model
)
from disaster_classifier.models.samplers import get_multilabel_class_weights
from disaster_classifier.evaluation.metrics import evaluate_model, save_model
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(
        description='Train classifier with class weighting for multi-label imbalance handling.'
    )

    parser.add_argument('--db', dest='database_filepath', 
                       default='data/02_stg/stg_disaster_response.db',
                       help='Path to SQLite database (default: data/02_stg/stg_disaster_response.db)')
    parser.add_argument('--params', dest='params_path', 
                       default='model/base_parameters.json',
                       help='Path to JSON parameters (default: model/base_parameters.json)')
    parser.add_argument('--out', dest='model_out', 
                       default='models/classifier_weighted.pkl',
                       help='Output model path (default: models/classifier_weighted.pkl)')
    parser.add_argument('--test-size', dest='test_size', type=float, default=0.2,
                       help='Test size fraction (default: 0.2)')
    parser.add_argument('--seed', dest='seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--use-class-weights', dest='use_class_weights', 
                       action='store_true',
                       help='Enable class weighting for imbalance handling')
    parser.add_argument('--baseline', dest='baseline', action='store_true',
                       help='Create baseline model without class weighting for comparison')

    args = parser.parse_args()

    setup_logging()
    
    # Determine model type
    model_type = "baseline" if args.baseline else "class_weighted"
    logging.info(f'Creating {model_type} model...')
    logging.info(f'Database: {args.database_filepath}')
    logging.info(f'Output: {args.model_out}')

    # Load data
    X, Y = load_data(args.database_filepath)
    if X is None or Y is None:
        logging.error('Failed to load data. Exiting.')
        sys.exit(1)

    logging.info(f'Loaded {len(X)} samples with {Y.shape[1]} labels')

    # Split data
    logging.info(f'Splitting data (test_size={args.test_size}, seed={args.seed})...')
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=args.test_size, random_state=args.seed
    )

    # Create pipeline based on model type
    if args.baseline:
        logging.info('Creating baseline pipeline (no class weighting)...')
        pipeline = create_pipeline(use_class_weights=False)
    else:
        logging.info('Creating pipeline with class weighting...')
        
        if args.use_class_weights:
            # Calculate and display class weights
            class_weights = get_multilabel_class_weights(Y_train, strategy='balanced')
            if class_weights:
                logging.info(f'Calculated class weights for {len(class_weights)} labels')
                
                # Show some example weights for most imbalanced labels
                import numpy as np
                imbalance_info = []
                for i, weights in class_weights.items():
                    if len(weights) == 2:  # Binary classification
                        weight_ratio = max(weights.values()) / min(weights.values())
                        imbalance_info.append((i, weight_ratio, weights))
                
                # Sort by weight ratio and show top 5
                imbalance_info.sort(key=lambda x: x[1], reverse=True)
                logging.info('Top 5 most imbalanced labels with calculated weights:')
                for i, (label_idx, ratio, weights) in enumerate(imbalance_info[:5]):
                    label_name = TARGET_COLUMNS[label_idx] if label_idx < len(TARGET_COLUMNS) else f"Label_{label_idx}"
                    logging.info(f'  {label_name}: weight_ratio={ratio:.2f}, weights={weights}')
            
            pipeline = create_pipeline_with_custom_weights()
        else:
            pipeline = create_pipeline(use_class_weights=True)
    
    if pipeline is None:
        logging.error('Failed to create pipeline. Exiting.')
        sys.exit(1)

    # Load parameters
    logging.info(f'Loading parameters from {args.params_path}')
    parameters = load_model_parameters(args.params_path)
    if parameters is None:
        logging.warning(f'Failed to load parameters from {args.params_path}. Using defaults.')
        parameters = {}

    # Build model
    logging.info('Building model with provided parameters...')
    model = build_model(pipeline, parameters)
    if model is None:
        logging.error('Failed to build model. Exiting.')
        sys.exit(1)

    # Train model
    logging.info('Training model...')
    from time import time
    train_start = time()
    
    model.fit(X_train, Y_train)
    
    train_time = time() - train_start
    logging.info(f'Model training completed in {train_time:.2f} seconds')

    # Evaluate model
    logging.info('Evaluating model...')
    evaluate_model(model, model_type, X_test, Y_test, TARGET_COLUMNS)

    # Save model
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    logging.info(f'Saving model to {args.model_out}')
    save_model(model, args.model_out)

    # Success message
    print(f'\n[SUCCESS] {model_type.title()} model trained and saved successfully!')
    print(f'Model path: {args.model_out}')
    print(f'Evaluation written under data/04_fct as fct_{model_type}_prediction_results.csv')
    print(f'Training time: {train_time:.2f} seconds')
    
    if not args.baseline:
        print('Model includes class weighting for improved minority class detection')


if __name__ == '__main__':
    main()
