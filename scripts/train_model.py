#!/usr/bin/env python3
"""
Clean training script for disaster response classification.

This script demonstrates professional ML engineering practices with:
- Modular, single-responsibility components
- Clear experiment tracking
- Organized results and model storage
- Easy comparison between different approaches
"""

import sys
import os
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from disaster_classifier.utils.config import setup_logging, TARGET_COLUMNS
from disaster_classifier.data.loader import load_data
from disaster_classifier.models.samplers import apply_multi_label_aware_sampling
from disaster_classifier.models.pipeline import create_pipeline, build_model, run_grid_search
from disaster_classifier.evaluation.metrics import evaluate_model, save_model, save_gs_results, save_best_parameters
from disaster_classifier.utils.io import load_model_parameters, load_grid_search_parameters
from disaster_classifier.utils.interaction import get_user_input
from disaster_classifier.utils.experiment_tracker import ExperimentTracker, create_experiment_name, build_slug
from sklearn.model_selection import train_test_split


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
    logging.info(f"Loading data for experiment: {experiment_name}")
    X, Y = load_data(database_filepath)
    if X is None or Y is None:
        logging.error("Error loading data from database")
        return None
    
    # Split data
    logging.info("Splitting data into training and test sets...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Apply sampling if not baseline
    if sampling_method != 'baseline':
        logging.info(f"Applying {sampling_method} sampling...")
        X_train, Y_train = apply_multi_label_aware_sampling(X_train, Y_train, method=sampling_method)
    
    # Create pipeline
    logging.info("Creating ML pipeline...")
    pipeline = create_pipeline()
    
    # Save experiment configuration
    config = {
        'sampling_method': sampling_method,
        'test_size': 0.2,
        'random_state': 42,
        'target_columns': TARGET_COLUMNS,
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
    except Exception:
        pass
    tracker.save_experiment_config_flat(slug, experiment_name, config)
    
    # Train model
    logging.info("Training model...")
    model = build_model(pipeline, None)  # Use default parameters
    model.fit(X_train, Y_train)
    
    # Evaluate model (use slug to make fct file unique per run)
    logging.info("Evaluating model...")
    evaluate_model(model, slug, X_test, Y_test, TARGET_COLUMNS)
    
    # Save model
    saved_model_path = None
    if model_filepath is None:
        # Save to flat experiments bucket
        saved_model_path = tracker.save_model_flat(slug, model)
    else:
        # Respect provided path (backward compatible)
        save_model(model, model_filepath)
        saved_model_path = model_filepath
    
    # Save results summary (flat; legacy results kept optional)
    results = {
        'experiment_name': experiment_name,
        'sampling_method': sampling_method,
        'slug': slug,
        'model_saved_to': saved_model_path,
        'evaluation_completed': True
    }
    tracker.save_results_flat(slug, results)
    
    logging.info(f"Experiment {experiment_name} completed successfully!")
    print(f"SLUG: {slug}")
    print(f"MODEL_PATH: {saved_model_path}")
    return model


def main():
    """
    Main function with interactive training workflow.
    """
    # Set up logging
    setup_logging()
    
    # Accept either: database only, or database + explicit model path (backward compatible)
    if len(sys.argv) not in (2, 3):
        logging.info(
            "Usage:\n"
            "  python scripts/train_model.py data/02_stg/stg_disaster_response.db [models/output.pkl]"
        )
        return

    database_filepath = sys.argv[1]
    model_filepath = sys.argv[2] if len(sys.argv) == 3 else None
    
    # Interactive experiment selection
    print("\n=== Disaster Response Classification Training ===")
    print("Available experiments:")
    print("1. baseline_no_sampling - No sampling applied")
    print("2. smote_conservative - SMOTE with conservative parameters")
    print("3. adasyn_moderate - ADASYN with moderate parameters")
    print("4. conservative_sampling - Very conservative SMOTE")
    print("5. Custom experiment")
    
    choice = input("\nSelect experiment (1-5): ").strip()
    
    if choice == "1":
        experiment_name = create_experiment_name("baseline")
        sampling_method = "baseline"
    elif choice == "2":
        experiment_name = create_experiment_name("smote")
        sampling_method = "smote"
    elif choice == "3":
        experiment_name = create_experiment_name("adasyn")
        sampling_method = "adasyn"
    elif choice == "4":
        experiment_name = create_experiment_name("conservative")
        sampling_method = "conservative"
    elif choice == "5":
        experiment_name = input("Enter experiment name: ").strip()
        sampling_method = input("Enter sampling method (baseline/smote/adasyn/conservative): ").strip()
    else:
        print("Invalid choice. Exiting.")
        return
    
    # Train the experiment
    model = train_experiment(experiment_name, sampling_method, database_filepath, model_filepath)
    
    if model is not None:
        print(f"\n[SUCCESS] Experiment '{experiment_name}' completed successfully!")
        if model_filepath:
            print(f"Model saved to: {model_filepath}")
        print("Results saved to flat experiments buckets (configs/models/results)")
    else:
        print("\n[ERROR] Experiment failed. Check logs for details.")


if __name__ == "__main__":
    main()
