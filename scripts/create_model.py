#!/usr/bin/env python3
"""
Create a production disaster response classification model.

This script creates a production model by reading configuration from:
- model/parameters.json (hyperparameters)
- model/class_weights.json (class weighting configuration)

The model uses the current preprocessing pipeline with fixed tokenization
and can optionally apply class weighting based on the configuration.

Usage:
    python scripts/create_model.py
    python scripts/create_model.py --output model/disaster_response_classifier.pkl
    python scripts/create_model.py --params model/parameters.json --class-weights model/class_weights.json
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
import json


def load_class_weights_config(file_path):
    """
    Load class weights configuration from JSON file.
    
    Args:
        file_path (str): Path to the class weights JSON file
        
    Returns:
        dict: Class weights configuration, or None if error
    """
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


def main():
    parser = argparse.ArgumentParser(
        description='Create production disaster response classification model.'
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
                       default='model/classifier.pkl',
                       help='Output model path (default: model/classifier.pkl)')
    parser.add_argument('--test-size', dest='test_size', type=float, default=0.2,
                       help='Test size fraction (default: 0.2)')
    parser.add_argument('--seed', dest='seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    setup_logging()
    logging.info('Creating production disaster response classification model...')
    logging.info(f'Database: {args.database_filepath}')
    logging.info(f'Hyperparameters: {args.params_path}')
    logging.info(f'Class weights: {args.class_weights_path}')
    logging.info(f'Output: {args.model_out}')

    # Load data
    logging.info('Loading data...')
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

    # Load hyperparameters
    logging.info(f'Loading hyperparameters from {args.params_path}')
    parameters = load_model_parameters(args.params_path)
    if parameters is None:
        logging.error(f'Failed to load hyperparameters from {args.params_path}. Exiting.')
        sys.exit(1)

    # Load class weights configuration
    logging.info(f'Loading class weights configuration from {args.class_weights_path}')
    class_weights_config = load_class_weights_config(args.class_weights_path)
    if class_weights_config is None:
        logging.error(f'Failed to load class weights config from {args.class_weights_path}. Exiting.')
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
    from time import time
    train_start = time()
    
    model.fit(X_train, Y_train)
    
    train_time = time() - train_start
    logging.info(f'Model training completed in {train_time:.2f} seconds')

    # Evaluate model
    logging.info('Evaluating model...')
    model_name = "production_model"
    evaluate_model(model, model_name, X_test, Y_test, TARGET_COLUMNS)

    # Save model
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    logging.info(f'Saving model to {args.model_out}')
    save_model(model, args.model_out)

    # Success message
    class_weighting_status = "enabled" if class_weights_enabled else "disabled"
    print(f'\n✅ Production model created successfully!')
    print(f'📁 Model path: {args.model_out}')
    print(f'⚖️  Class weighting: {class_weighting_status}')
    print(f'📊 Evaluation results: data/04_fct/fct_{model_name}_prediction_results.csv')
    print(f'⏱️  Training time: {train_time:.2f} seconds')
    
    if class_weights_enabled:
        print('🎯 Model includes class weighting for improved minority class detection')


if __name__ == '__main__':
    main()
