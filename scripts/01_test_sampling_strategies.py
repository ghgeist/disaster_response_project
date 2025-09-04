#!/usr/bin/env python3
"""
Test different sampling strategies for disaster response classification.

This script provides an interactive menu to test different sampling methods
for handling class imbalance in multi-label classification. It uses the production
hyperparameters from model/parameters.json to ensure consistency with the deployed model.

Use this script when you want to:
- Test sampling strategies (baseline, SMOTE, ADASYN, conservative)
- Compare different approaches to class imbalance
- Run individual sampling experiments with production parameters

For batch runs of multiple experiments, use run_batch_experiments.py instead.

Usage:
    python scripts/test_sampling_strategies.py data/02_stg/stg_disaster_response.db [model_output.pkl]
    
The script will prompt you to select from available sampling strategies and handle all the
training, evaluation, and result storage automatically.
"""

import sys
import os
import logging
import json
import argparse
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from disaster_classifier.utils.config import setup_logging, TARGET_COLUMNS
from disaster_classifier.data.loader import load_data
from disaster_classifier.models.pipeline import create_pipeline, build_model
from disaster_classifier.models.samplers import apply_multi_label_aware_sampling
from disaster_classifier.evaluation.metrics import evaluate_model, save_model
from disaster_classifier.utils.io import load_model_parameters
from disaster_classifier.utils.experiment_tracker import ExperimentTracker, create_experiment_name, build_slug
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


DEFAULT_STRATEGIES_DIR = os.path.join('experiments', 'experimental_configs', 'sampling_strategies')
ALLOWED_SAMPLING_METHODS = {"baseline", "smote", "adasyn", "conservative"}


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments with backward-compatible positional args.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Train and evaluate sampling strategy experiments'
    )
    parser.add_argument('database_filepath', help='Path to SQLite database file')
    parser.add_argument('model_filepath', nargs='?', default=None,
                        help='Optional path to save the trained model')
    parser.add_argument('--strategies-dir', default=os.getenv('STRATEGIES_DIR', DEFAULT_STRATEGIES_DIR),
                        help=f'Directory containing sampling strategy JSON files (default: {DEFAULT_STRATEGIES_DIR})')
    return parser.parse_args()


def is_disabled_filename(filename: str) -> bool:
    """Return True if filename should be ignored by convention."""
    return (
        filename.startswith('_') or
        filename.endswith('.disabled.json')
    )


def load_json_file(file_path: str) -> Optional[dict]:
    """
    Load a JSON file safely.

    Returns None if the file cannot be read or parsed.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logging.warning("Skipping %s due to read/parse error: %s", file_path, e)
        return None


def validate_strategy_payload(payload: dict) -> Tuple[bool, Optional[str]]:
    """
    Minimal schema validation for strategy payloads.

    Required:
      - 'config' dict containing at least 'sampling_method' (str)

    Optional:
      - 'order' (int), 'display_name' (str), 'description' (str), 'experiment_name' (str)
    """
    if not isinstance(payload, dict):
        return False, 'payload is not a JSON object'
    config_obj = payload.get('config')
    if not isinstance(config_obj, dict):
        return False, "missing 'config' object"
    method = config_obj.get('sampling_method')
    if not isinstance(method, str) or not method.strip():
        return False, "missing or invalid 'config.sampling_method'"
    if method not in ALLOWED_SAMPLING_METHODS:
        return False, f"unsupported sampling_method '{method}' (allowed: {sorted(ALLOWED_SAMPLING_METHODS)})"

    # Optional fields type checks
    order = payload.get('order')
    if order is not None and not isinstance(order, int):
        return False, "optional 'order' must be an integer"
    if isinstance(order, int) and order < 0:
        return False, "optional 'order' must be >= 0"

    for key in ('experiment_name', 'display_name', 'description'):
        if key in payload and payload[key] is not None and not isinstance(payload[key], str):
            return False, f"optional '{key}' must be a string"
    return True, None


def normalize_strategy_entry(file_path: str, payload: dict) -> Dict[str, object]:
    """
    Normalize raw payload into a menu-friendly dict.
    """
    filename = os.path.basename(file_path)
    name_from_file = os.path.splitext(filename)[0]

    config_obj = payload.get('config', {})
    sampling_method = config_obj.get('sampling_method', 'baseline')

    display_name = payload.get('display_name') or payload.get('experiment_name') or name_from_file
    description = payload.get('description') or f"Strategy: {sampling_method}"
    order = payload.get('order')

    return {
        'path': file_path,
        'filename': filename,
        'display_name': display_name,
        'description': description,
        'order': int(order) if isinstance(order, int) else None,
        'sampling_method': sampling_method,
        'experiment_name': payload.get('experiment_name'),
    }


def discover_strategies(strategies_dir: str) -> List[Dict[str, object]]:
    """
    Discover valid strategies in a directory.

    - Ignores files by convention: leading '_' or '*.disabled.json'
    - Validates minimal schema
    - Respects 'enabled: false'
    - Returns deterministically sorted list by (order, display_name, filename)
    """
    results: List[Dict[str, object]] = []

    if not os.path.isdir(strategies_dir):
        logging.warning("Strategies directory not found: %s", strategies_dir)
        return results

    for filename in os.listdir(strategies_dir):
        if not filename.endswith('.json'):
            continue
        if is_disabled_filename(filename):
            continue
        file_path = os.path.join(strategies_dir, filename)
        payload = load_json_file(file_path)
        if payload is None:
            continue
        is_valid, reason = validate_strategy_payload(payload)
        if not is_valid:
            logging.warning("Skipping %s due to invalid schema: %s", filename, reason)
            continue
        entry = normalize_strategy_entry(file_path, payload)
        results.append(entry)

    # Sorting: entries with explicit order first (ascending), then by display_name or filename
    def sort_key(e: Dict[str, object]):
        has_order = 0 if e['order'] is not None else 1
        name_key = str(e['display_name']).lower() if e.get('display_name') else str(e['filename']).lower()
        return (has_order, e['order'] if e['order'] is not None else 0, name_key)

    results.sort(key=sort_key)
    return results


def calculate_overall_metrics(metrics_file_path: str) -> dict:
    """
    Calculate overall performance metrics from detailed metrics CSV.
    
    Args:
        metrics_file_path: Path to the metrics CSV file
        
    Returns:
        Dictionary with overall metrics
    """
    try:
        df = pd.read_csv(metrics_file_path)
        
        # Filter to only include class 0 and 1 (binary classification results)
        binary_df = df[df['output_class'].isin([0, 1])]
        
        # Calculate macro averages (average across all categories)
        macro_precision = binary_df[binary_df['output_class'] == 1]['precision'].mean()
        macro_recall = binary_df[binary_df['output_class'] == 1]['recall'].mean()
        macro_f1 = binary_df[binary_df['output_class'] == 1]['f1-score'].mean()
        
        # Calculate weighted averages (weighted by support)
        weighted_precision = np.average(
            binary_df[binary_df['output_class'] == 1]['precision'], 
            weights=binary_df[binary_df['output_class'] == 1]['support']
        )
        weighted_recall = np.average(
            binary_df[binary_df['output_class'] == 1]['recall'], 
            weights=binary_df[binary_df['output_class'] == 1]['support']
        )
        weighted_f1 = np.average(
            binary_df[binary_df['output_class'] == 1]['f1-score'], 
            weights=binary_df[binary_df['output_class'] == 1]['support']
        )
        
        return {
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'total_categories': len(binary_df[binary_df['output_class'] == 1])
        }
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError, ValueError) as e:
        logging.error("Error calculating metrics for %s: %s", metrics_file_path, e)
        return None


def create_experiment_comparison(results_dir: str = "results") -> str:
    """
    Create a comparison report of all experiments.
    
    Args:
        results_dir: Directory containing experiment results
        
    Returns:
        Path to the comparison report
    """
    try:
        
        # Find all metrics files
        metrics_files = [f for f in os.listdir(results_dir) if f.endswith('_metrics.csv')]
        
        if not metrics_files:
            logging.warning("No metrics files found for comparison")
            return None
            
        comparison_data = []
        
        for metrics_file in metrics_files:
            # Extract experiment name from filename
            experiment_name = metrics_file.replace('_metrics.csv', '').split('_', 1)[1]  # Remove date prefix
            metrics_path = os.path.join(results_dir, metrics_file)
            
            # Calculate overall metrics
            metrics = calculate_overall_metrics(metrics_path)
            
            if metrics:
                comparison_data.append({
                    'experiment': experiment_name,
                    'macro_precision': metrics['macro_precision'],
                    'macro_recall': metrics['macro_recall'],
                    'macro_f1': metrics['macro_f1'],
                    'weighted_precision': metrics['weighted_precision'],
                    'weighted_recall': metrics['weighted_recall'],
                    'weighted_f1': metrics['weighted_f1'],
                    'total_categories': metrics['total_categories']
                })
        
        if not comparison_data:
            logging.warning("No valid metrics found for comparison")
            return None
            
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by weighted F1 score (most important metric for imbalanced data)
        comparison_df = comparison_df.sort_values('weighted_f1', ascending=False)
        
        # Save comparison report
        date_str = datetime.now().strftime("%Y-%m-%d")
        comparison_path = os.path.join(results_dir, f"{date_str}_experiment_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        
        # Print summary
        print(f"\n{'='*60}")
        print("EXPERIMENT COMPARISON RESULTS")
        print(f"{'='*60}")
        print(f"Best performing experiment: {comparison_df.iloc[0]['experiment']}")
        print(f"Best weighted F1 score: {comparison_df.iloc[0]['weighted_f1']:.4f}")
        print(f"\nFull comparison saved to: {comparison_path}")
        print(f"\nTop 3 experiments by weighted F1:")
        print(comparison_df[['experiment', 'weighted_f1', 'macro_f1', 'weighted_precision', 'weighted_recall']].head(3).to_string(index=False))
        print(f"{'='*60}")
        
        return comparison_path
        
    except (OSError, pd.errors.EmptyDataError, ValueError) as e:
        logging.error("Error creating experiment comparison: %s", e)
        return None


def train_experiment(experiment_name: str, sampling_method: str, 
                    database_filepath: str, model_filepath: str = None):
    """
    Train a model for a specific experiment configuration.
    
    Args:
        experiment_name: Name of the experiment
        sampling_method: Sampling method to use ('baseline', 'smote', 'adasyn', 'conservative')
        database_filepath: Path to the database file
        model_filepath: Path to save the model (optional, will use experiment tracker if not provided)
    """
    # Initialize experiment tracker
    tracker = ExperimentTracker()
    
    # Load data
    logging.info("Loading data for experiment: %s", experiment_name)
    X, Y = load_data(database_filepath)
    if X is None or Y is None:
        logging.error("Error loading data from database")
        return None
    
    # Split data
    logging.info("Splitting data into training and test sets...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Apply sampling if not baseline
    if sampling_method != 'baseline':
        logging.info("Applying %s sampling...", sampling_method)
        
        X_train, Y_train = apply_multi_label_aware_sampling(
            X_train, Y_train, 
            method=sampling_method
        )
    
    # Create pipeline
    logging.info("Creating ML pipeline...")
    pipeline = create_pipeline()
    
    # Load production parameters for config
    parameters_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'parameters.json')
    loaded_parameters = load_model_parameters(parameters_path)
    
    # Save experiment configuration
    config = {
        'sampling_method': sampling_method,
        'test_size': 0.2,
        'random_state': 42,
        'target_columns': TARGET_COLUMNS,
        'hyperparameters': loaded_parameters if loaded_parameters else 'default',
        'data_shape': {
            'X_train': X_train.shape,
            'X_test': X_test.shape,
            'Y_train': Y_train.shape,
            'Y_test': Y_test.shape
        }
    }
    # Build a run slug for flat saving
    slug = build_slug(sampling_method, version="v1")

    # Legacy nested config (kept small) + flat config
    try:
        tracker.save_experiment_config(experiment_name, config)
    except (OSError, IOError, ValueError) as e:
        logging.warning("Could not save nested experiment config: %s", e)
    tracker.save_experiment_config_flat(slug, experiment_name, config)
    
    # Train model with loaded parameters
    logging.info("Training model...")
    model = build_model(pipeline, loaded_parameters)
    model.fit(X_train, Y_train)
    
    # Ensure results directory exists for metrics
    os.makedirs('results', exist_ok=True)
    
    # Evaluate model (use experiment name for cleaner file naming)
    logging.info("Evaluating model...")
    evaluate_model(model, experiment_name, X_test, Y_test, TARGET_COLUMNS)
    
    # Save model
    saved_model_path = None
    if model_filepath is None:
        # Save to flat experiments bucket
        saved_model_path = tracker.save_model_flat(slug, model)
    else:
        # Respect provided path (backward compatible)
        save_model(model, model_filepath)
        saved_model_path = model_filepath
    
    # Save results summary
    results = {
        'experiment_name': experiment_name,
        'sampling_method': sampling_method,
        'slug': slug,
        'model_saved_to': saved_model_path,
        'evaluation_completed': True
    }
    tracker.save_results_flat(slug, results)
    
    logging.info("Experiment %s completed successfully!", experiment_name)
    print(f"SLUG: {slug}")
    print(f"MODEL_PATH: {saved_model_path}")
    return model


def main():
    """
    Main function with interactive training workflow.
    """
    # Set up logging
    setup_logging()

    # Parse arguments (backward compatible)
    try:
        args = parse_args()
    except SystemExit:
        # argparse already printed usage
        return

    database_filepath = args.database_filepath
    model_filepath = args.model_filepath
    strategies_dir = args.strategies_dir

    # Discover strategies
    strategies = discover_strategies(strategies_dir)

    # Interactive experiment selection
    print("\n=== Disaster Response Classification Training ===")
    print("Available experiments (auto-discovered):")
    if strategies:
        for idx, s in enumerate(strategies, start=1):
            print(f"{idx}. {s['display_name']} - {s['description']}")
        base_offset = len(strategies)
    else:
        print("(none found in directory)")
        base_offset = 0
    print(f"{base_offset + 1}. Custom experiment")
    print(f"{base_offset + 2}. Compare all experiments - Show performance comparison")

    choice = input(f"\nSelect option (1-{base_offset + 2}): ").strip()

    try:
        choice_num = int(choice)
    except ValueError:
        print("Invalid choice. Exiting.")
        return

    if 1 <= choice_num <= base_offset and strategies:
        selected = strategies[choice_num - 1]
        sampling_method = str(selected['sampling_method'])
        # Prefer experiment_name from file; otherwise generate one
        experiment_name = selected.get('experiment_name') or create_experiment_name(sampling_method)
    elif choice_num == base_offset + 1:
        experiment_name = input("Enter experiment name: ").strip()
        sampling_method = input("Enter sampling method (baseline/smote/adasyn/conservative): ").strip()
    elif choice_num == base_offset + 2:
        # Compare all experiments
        print("\n=== Comparing All Experiments ===")
        comparison_path = create_experiment_comparison("results")
        if comparison_path:
            print(f"\nComparison complete! Results saved to: {comparison_path}")
        else:
            print("\nNo experiments found to compare. Run some experiments first.")
        return
    else:
        print("Invalid choice. Exiting.")
        return

    # Train the experiment
    model = train_experiment(experiment_name, sampling_method, database_filepath, model_filepath)

    if model is not None:
        print(f"\n[SUCCESS] Experiment '{experiment_name}' completed successfully!")
        if model_filepath:
            print(f"Model saved to: {model_filepath}")
        print("Files saved:")
        date_str = datetime.now().strftime("%Y-%m-%d")
        print(f"  - Model: models/{date_str}_{experiment_name}.pkl")
        print(f"  - Config: results/{date_str}_{experiment_name}_config.json")
        print(f"  - Summary: results/{date_str}_{experiment_name}_summary.json")
        print(f"  - Metrics: results/{date_str}_{experiment_name}_metrics.csv")

        # Show quick comparison if other experiments exist
        print(f"\n{'='*50}")
        print("QUICK COMPARISON")
        print(f"{'='*50}")
        comparison_path = create_experiment_comparison("results")
        if comparison_path:
            print(f"\nFull comparison saved to: {comparison_path}")
        print(f"{'='*50}")
    else:
        print("\n[ERROR] Experiment failed. Check logs for details.")


if __name__ == "__main__":
    main()
