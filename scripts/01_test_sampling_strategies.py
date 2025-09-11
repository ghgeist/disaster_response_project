#!/usr/bin/env python3
"""
Test different sampling strategies for disaster response classification.

This script provides an interactive menu to test different sampling methods
for handling class imbalance in multi-label classification. It uses the production
hyperparameters from model/parameters.json to ensure consistency with the deployed model.

Use this script when you want to:
- Test sampling strategies (currently only baseline works with this dataset)
- Compare different approaches to class imbalance
- Run individual sampling experiments with production parameters

For batch runs of multiple experiments, use run_batch_experiments.py instead.

Usage:
    python scripts/test_sampling_strategies.py [data/02_stg/stg_disaster_response.db] [model_output.pkl]
    (DB path is optional; defaults to data/02_stg/stg_disaster_response.db)
    
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

from disasterproject.utils.config import setup_logging, TARGET_COLUMNS
from disasterproject.data.loader import load_data
from disasterproject.models.pipeline import create_pipeline, build_model
from disasterproject.models.samplers import apply_multi_label_aware_sampling
from disasterproject.evaluation.metrics import evaluate_model, save_model
from disasterproject.utils.json_io import load_model_parameters
from disasterproject.utils.experiment_tracker import create_experiment_name, build_slug
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


DEFAULT_STRATEGIES_DIR = os.path.join('experiments', 'experimental_configs', 'sampling_strategies')
ALLOWED_SAMPLING_METHODS = {"baseline"}  # Only baseline works with this dataset


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments with backward-compatible positional args.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Train and evaluate sampling strategy experiments'
    )
    parser.add_argument('database_filepath', nargs='?', default='data/02_stg/stg_disaster_response.db',
                        help='Path to SQLite database file (default: data/02_stg/stg_disaster_response.db)')
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
    Simple validation for strategy payloads.
    """
    if not isinstance(payload, dict):
        return False, 'payload is not a JSON object'
    
    config_obj = payload.get('config')
    if not isinstance(config_obj, dict):
        return False, "missing 'config' object"
    
    method = config_obj.get('sampling_method')
    if not method or method not in ALLOWED_SAMPLING_METHODS:
        return False, f"unsupported sampling_method '{method}' (allowed: {sorted(ALLOWED_SAMPLING_METHODS)})"
    
    return True, None


def normalize_strategy_entry(file_path: str, payload: dict) -> Dict[str, object]:
    """
    Normalize raw payload into a menu-friendly dict.
    """
    config_obj = payload.get('config', {})
    sampling_method = config_obj.get('sampling_method', 'baseline')
    experiment_name = payload.get('experiment_name', f"{sampling_method}_v1")
    
    return {
        'path': file_path,
        'display_name': experiment_name,
        'description': f"Strategy: {sampling_method}",
        'sampling_method': sampling_method,
        'experiment_name': experiment_name,
    }


def discover_strategies(strategies_dir: str) -> List[Dict[str, object]]:
    """
    Discover valid strategies in a directory.
    """
    results: List[Dict[str, object]] = []

    if not os.path.isdir(strategies_dir):
        logging.warning("Strategies directory not found: %s", strategies_dir)
        return results

    for filename in sorted(os.listdir(strategies_dir)):
        if not filename.endswith('.json') or is_disabled_filename(filename):
            continue
            
        file_path = os.path.join(strategies_dir, filename)
        payload = load_json_file(file_path)
        if payload is None:
            continue
            
        is_valid, reason = validate_strategy_payload(payload)
        if not is_valid:
            logging.warning("Skipping %s: %s", filename, reason)
            continue
            
        entry = normalize_strategy_entry(file_path, payload)
        results.append(entry)

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
        
        if binary_df.empty:
            logging.warning("No binary classification results found in %s", metrics_file_path)
            return None
        
        # Calculate macro averages (average across all categories)
        class_1_metrics = binary_df[binary_df['output_class'] == 1]
        
        if class_1_metrics.empty:
            logging.warning("No positive class metrics found in %s", metrics_file_path)
            return None
            
        macro_precision = class_1_metrics['precision'].fillna(0.0).mean()
        macro_recall = class_1_metrics['recall'].fillna(0.0).mean()
        macro_f1 = class_1_metrics['f1-score'].fillna(0.0).mean()
        
        # Calculate weighted averages (weighted by support)
        class_1_data = binary_df[binary_df['output_class'] == 1]
        weights = class_1_data['support'].values
        
        # Handle case where all weights are zero (no positive predictions)
        if np.sum(weights) == 0:
            weighted_precision = 0.0
            weighted_recall = 0.0
            weighted_f1 = 0.0
        else:
            # Fill NaN values with 0.0 before calculating weighted averages
            precision_values = class_1_data['precision'].fillna(0.0).values
            recall_values = class_1_data['recall'].fillna(0.0).values
            f1_values = class_1_data['f1-score'].fillna(0.0).values
            
            weighted_precision = np.average(precision_values, weights=weights)
            weighted_recall = np.average(recall_values, weights=weights)
            weighted_f1 = np.average(f1_values, weights=weights)
        
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


def create_experiment_comparison(results_dir: str = "experiments/results") -> str:
    """
    Create a comparison report of all experiments.
    """
    try:
        if not os.path.isdir(results_dir):
            logging.warning("Results directory not found: %s", results_dir)
            return None

        # Find metrics files
        found_files = []
        for f in os.listdir(results_dir):
            if f.endswith('_metrics.csv'):
                found_files.append((results_dir, f))
        
        if not found_files:
            logging.warning("No metrics files found for comparison in %s", results_dir)
            return None
        
        comparison_data = []
        
        for d, metrics_file in found_files:
            # Derive experiment name from filename
            experiment_name = metrics_file.replace('_metrics.csv', '').split('_', 1)[1] if '_' in metrics_file else metrics_file
            metrics_path = os.path.join(d, metrics_file)
            
            # Calculate overall metrics
            metrics = calculate_overall_metrics(metrics_path)
            
            if metrics is not None:
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
        print("\n" + "="*60)
        print("EXPERIMENT COMPARISON RESULTS")
        print("="*60)
        print(f"Best performing experiment: {comparison_df.iloc[0]['experiment']}")
        print(f"Best weighted F1 score: {comparison_df.iloc[0]['weighted_f1']:.4f}")
        print(f"\nFull comparison saved to: {comparison_path}")
        print("\nTop 3 experiments by weighted F1:")
        print(comparison_df[['experiment', 'weighted_f1', 'macro_f1', 'weighted_precision', 'weighted_recall']].head(3).to_string(index=False))
        print("="*60)
        
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
        
        try:
            X_train_sampled, Y_train_sampled = apply_multi_label_aware_sampling(
                X_train, Y_train, 
                method=sampling_method
            )
            
            # Check if sampling actually worked (data changed)
            if len(X_train_sampled) == len(X_train) and np.array_equal(Y_train_sampled, Y_train):
                logging.error("CRITICAL: %s sampling failed - no changes to training data", sampling_method)
                print(f"[EXPERIMENT FAILED] {sampling_method} sampling could not be applied.")
                print("Stopping experiment to prevent misleading results.")
                return None
            
            X_train, Y_train = X_train_sampled, Y_train_sampled
            
        except Exception as e:
            logging.error("CRITICAL: %s sampling failed with exception: %s", sampling_method, e)
            print(f"[EXPERIMENT FAILED] {sampling_method} sampling failed: {e}")
            print("Stopping experiment to prevent misleading results.")
            return None
    
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

    # Save config to results directory
    date_str = datetime.now().strftime("%Y-%m-%d")
    config_path = os.path.join('experiments', 'results', f"{date_str}_{experiment_name}_config.json")
    config_with_metadata = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'config': config
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_with_metadata, f, indent=2)
    
    # Train model with loaded parameters
    logging.info("Training model...")
    model = build_model(pipeline, loaded_parameters)
    model.fit(X_train, Y_train)
    
    # Evaluate model
    logging.info("Evaluating model...")
    evaluate_model(model, experiment_name, X_test, Y_test, TARGET_COLUMNS)
    
    # Save model
    saved_model_path = None
    if model_filepath is None:
        # Save to results directory
        date_str = datetime.now().strftime("%Y-%m-%d")
        saved_model_path = os.path.join('experiments', 'results', f"{date_str}_{experiment_name}.pkl")
        save_model(model, saved_model_path)
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
    # Save results summary to results directory
    results_path = os.path.join('experiments', 'results', f"{date_str}_{experiment_name}_summary.json")
    results_with_metadata = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_with_metadata, f, indent=2)
    
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
    print(f"{base_offset + 1}. Run all experiments")
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
        # Run all discovered experiments sequentially
        if not strategies:
            print("No strategies discovered to run.")
            return
        print("\n=== Running All Experiments ===")
        successful_experiments = []
        failed_experiments = []
        
        for idx, s in enumerate(strategies, start=1):
            sampling_method = str(s['sampling_method'])
            experiment_name = s.get('experiment_name') or create_experiment_name(sampling_method)
            print(f"\n[{idx}/{len(strategies)}] Training: {experiment_name} ({sampling_method})")
            
            try:
                result = train_experiment(experiment_name, sampling_method, database_filepath, model_filepath)
                if result is not None:
                    successful_experiments.append(experiment_name)
                    print(f"[SUCCESS] Completed: {experiment_name}")
                else:
                    failed_experiments.append(experiment_name)
                    print(f"[FAILED] Skipped: {experiment_name}")
            except Exception as e:
                logging.error("Experiment failed with exception: %s", e)
                failed_experiments.append(experiment_name)
                print(f"[ERROR] Failed: {experiment_name} - {e}")
        
        # Print batch summary
        print(f"\n{'='*60}")
        print("BATCH EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Total experiments: {len(strategies)}")
        print(f"Successful: {len(successful_experiments)}")
        print(f"Failed: {len(failed_experiments)}")
        
        if successful_experiments:
            print(f"\nSuccessful experiments:")
            for exp in successful_experiments:
                print(f"  ✓ {exp}")
        
        if failed_experiments:
            print(f"\nFailed experiments:")
            for exp in failed_experiments:
                print(f"  ✗ {exp}")
        # Generate comparison only if we have successful experiments
        if successful_experiments:
            print("\n" + "="*50)
            print("GENERATING COMPARISON FOR SUCCESSFUL EXPERIMENTS")
            print("="*50)
            comparison_path = create_experiment_comparison()
            if comparison_path:
                print(f"\nComparison complete! Results saved to: {comparison_path}")
            else:
                print("\nNo metrics found to compare.")
        else:
            print("\nNo successful experiments to compare.")
        return
    elif choice_num == base_offset + 2:
        # Compare all experiments
        print("\n=== Comparing All Experiments ===")
        comparison_path = create_experiment_comparison()
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
        print(f"  - Model: experiments/results/{date_str}_{experiment_name}.pkl")
        print(f"  - Config: experiments/results/{date_str}_{experiment_name}_config.json")
        print(f"  - Summary: experiments/results/{date_str}_{experiment_name}_summary.json")
        print(f"  - Metrics: experiments/results/{date_str}_{experiment_name}_metrics.csv")

        # Show quick comparison if other experiments exist
        print("\n" + "="*50)
        print("QUICK COMPARISON")
        print("="*50)
        comparison_path = create_experiment_comparison()
        if comparison_path:
            print(f"\nFull comparison saved to: {comparison_path}")
        print("="*50)
    else:
        print("\n[ERROR] Experiment failed. Check logs for details.")


if __name__ == "__main__":
    main()
