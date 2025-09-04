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
from sklearn.metrics import classification_report
import pandas as pd


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


def evaluate_model_to_results_folder(model, X_test, Y_test, category_names, results_dir="experiments/results"):
    """
    Evaluate model and save results to the experimental results folder.

    Args:
        model: Trained model
        X_test: Test features
        Y_test: Test labels
        category_names: List of category names
        results_dir: Directory to save results (default: "experiments/results")

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

        # Save to experiments results location
        os.makedirs(results_dir, exist_ok=True)
        results_file_path = os.path.join(results_dir, "performance_metrics.csv")
        results_df.to_csv(results_file_path, index=False)
        logging.info("Performance metrics saved to: %s", results_file_path)

        # Calculate summary statistics
        weighted_avg = results_df[results_df['output_class'] == 'weighted avg']
        positive_class = results_df[results_df['output_class'] == '1']

        summary = {
            'overall_precision': weighted_avg['precision'].mean(),
            'overall_recall': weighted_avg['recall'].mean(),
            'overall_f1': weighted_avg['f1-score'].mean(),
            'positive_class_precision': positive_class['precision'].mean(),
            'positive_class_recall': positive_class['recall'].mean(),
            'positive_class_f1': positive_class['f1-score'].mean(),
            'total_categories': len(category_names),
            'test_samples': len(Y_test)
        }

        return summary

    except Exception as e:
        logging.error("Error evaluating model: %s", e)
        return {}


def save_training_log(results_dir, config, performance_summary, training_time, model_path):
    """Save training metadata to experimental JSON log."""
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'training_time_seconds': training_time,
        'configuration': config,
        'performance': performance_summary,
        'version': '0.1-exp',
        'status': 'experimental'
    }

    log_path = os.path.join(results_dir, 'training_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2)

    logging.info(f"Training log saved to: {log_path}")
    return log_path


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
    parser.add_argument('--output', dest='model_out',
                       default='experiments/results/experimental_classifier.pkl',
                       help='Output model path (default: experiments/results/experimental_classifier.pkl)')
    parser.add_argument('--test-size', dest='test_size', type=float, default=0.2,
                       help='Test size fraction (default: 0.2)')
    parser.add_argument('--seed', dest='seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    setup_logging()

    print(f"\n🧪 Creating Experimental Disaster Response Model")
    print(f"{'='*60}")
    print(f"📁 Database: {args.database_filepath}")
    print(f"⚙️  Hyperparameters: {args.params_path}")
    print(f"⚖️  Class weights: {args.class_weights_path}")
    print(f"💾 Output: {args.model_out}")
    print(f"🎯 Results will be saved to experiments/results/ directory for clarity")
    print(f"{'='*60}")

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

    # Evaluate model and save to experiments results directory
    logging.info('Evaluating model and saving results to experiments/results directory...')
    results_dir = os.path.dirname(args.model_out) or "experiments/results"
    performance_summary = evaluate_model_to_results_folder(
        model, X_test, Y_test, TARGET_COLUMNS, results_dir
    )

    # Save model
    os.makedirs(results_dir, exist_ok=True)
    logging.info(f'Saving model to {args.model_out}')
    save_model(model, args.model_out)

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
            'test_samples': len(X_test)
        },
        'target_labels': len(TARGET_COLUMNS)
    }

    # Save training log
    training_log_path = save_training_log(
        results_dir, comprehensive_config, performance_summary, train_time, args.model_out
    )

    # Success summary
    print(f'\n✅ Experimental Model Created Successfully!')
    print(f"{'='*60}")
    print(f'📁 Model: {args.model_out}')
    print(f'📊 Performance: {results_dir}/performance_metrics.csv')
    print(f'📝 Training Log: {training_log_path}')
    print(f'⏱️  Training Time: {train_time:.2f} seconds')
    print(f"{'='*60}")

    print(f'\n📈 Performance Summary:')
    print(f'   Overall F1-Score: {performance_summary.get("overall_f1", 0):.4f}')
    print(f'   Overall Recall: {performance_summary.get("overall_recall", 0):.4f}')
    print(f'   Overall Precision: {performance_summary.get("overall_precision", 0):.4f}')
    print(f'   Positive Class F1: {performance_summary.get("positive_class_f1", 0):.4f}')

    class_weighting_status = "enabled" if class_weights_enabled else "disabled"
    print(f'\n⚖️  Class Weighting: {class_weighting_status}')
    if class_weights_enabled:
        print('   🧪 Model uses balanced class weights for experimental evaluation')

    print(f'\n📁 Clean Results Structure:')
    print(f'   experiments/results/experimental_classifier.pkl <- Experimental model artifact')
    print(f'   experiments/results/performance_metrics.csv     <- Experimental performance')
    print(f'   experiments/results/training_log.json           <- Training metadata & config')
    print(f'\n💡 Use this clean structure to organize experimental runs!')


if __name__ == '__main__':
    main()


