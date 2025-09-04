#!/usr/bin/env python3
"""
Create a baseline disaster response classification model.

This script creates a baseline model using default parameters without any
class imbalance handling or sampling methods. Perfect for:
- Establishing baseline performance metrics
- Creating reference models for comparison
- Quick model builds without experiment overhead

For models with class weighting, use create_weighted_model.py instead.
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
from disaster_classifier.models.pipeline import create_pipeline, build_model
from disaster_classifier.evaluation.metrics import evaluate_model, save_model
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description='Train classifier from JSON parameters.')

    parser.add_argument('--db', dest='database_filepath', default='data/02_stg/stg_disaster_response.db',
                        help='Path to SQLite database (default: data/02_stg/stg_disaster_response.db)')
    parser.add_argument('--params', dest='params_path', default='model/base_parameters.json',
                        help='Path to JSON parameters (default: model/base_parameters.json)')
    parser.add_argument('--out', dest='model_out', default='models/classifier.pkl',
                        help='Output model path (default: models/classifier.pkl)')
    parser.add_argument('--test-size', dest='test_size', type=float, default=0.2,
                        help='Test size fraction (default: 0.2)')
    parser.add_argument('--seed', dest='seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    setup_logging()
    logging.info('Loading data from %s', args.database_filepath)

    X, Y = load_data(args.database_filepath)
    if X is None or Y is None:
        logging.error('Failed to load data. Exiting.')
        sys.exit(1)

    logging.info('Splitting data (test_size=%s, seed=%s)...', args.test_size, args.seed)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=args.test_size, random_state=args.seed
    )

    logging.info('Creating pipeline...')
    pipeline = create_pipeline()
    if pipeline is None:
        logging.error('Failed to create pipeline. Exiting.')
        sys.exit(1)

    logging.info('Loading parameters from %s', args.params_path)
    parameters = load_model_parameters(args.params_path)
    if parameters is None:
        logging.error('Failed to load parameters from %s. Exiting.', args.params_path)
        sys.exit(1)

    logging.info('Building model with provided parameters...')
    model = build_model(pipeline, parameters)
    if model is None:
        logging.error('Failed to build model. Exiting.')
        sys.exit(1)

    logging.info('Training model...')
    model.fit(X_train, Y_train)

    logging.info('Evaluating model...')
    evaluate_model(model, 'base_model', X_test, Y_test, TARGET_COLUMNS)

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    logging.info('Saving model to %s', args.model_out)
    save_model(model, args.model_out)

    print('\n✅ Model trained and saved successfully!')
    print(f'📁 Model path: {args.model_out}')
    print('📊 Evaluation written under data/04_fct as fct_base_model_prediction_results.csv')


if __name__ == '__main__':
    main()


