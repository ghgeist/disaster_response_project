#!/usr/bin/env python3
"""
Multi-label sampling validation and testing script.

This script validates the proper multi-label sampling approaches and class weighting
implementation for the disaster response classification system.
"""

import sys
import os
import logging
from time import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from disaster_classifier.data.loader import load_data
from disaster_classifier.models.samplers import (
    apply_proper_multilabel_sampling,
    get_multilabel_class_weights
)
from disaster_classifier.models.pipeline import (
    create_pipeline,
    create_pipeline_with_custom_weights,
    build_model
)
from disaster_classifier.evaluation.metrics import evaluate_model
from disaster_classifier.utils.config import setup_logging, TARGET_COLUMNS
from sklearn.model_selection import train_test_split
import numpy as np


def analyze_class_distribution(y_original, y_resampled, method_name):
    """Analyze and compare class distributions before and after sampling."""
    logging.info(f"\n=== Class Distribution Analysis: {method_name} ===")
    
    n_labels = y_original.shape[1]
    improvements = []
    
    for i in range(min(10, n_labels)):  # Show top 10 most imbalanced
        # Original distribution
        unique_orig, counts_orig = np.unique(y_original[:, i], return_counts=True)
        if len(unique_orig) == 2:
            ratio_orig = max(counts_orig) / min(counts_orig)
            
            # Resampled distribution
            unique_res, counts_res = np.unique(y_resampled[:, i], return_counts=True)
            if len(unique_res) == 2:
                ratio_res = max(counts_res) / min(counts_res)
                improvement = (ratio_orig - ratio_res) / ratio_orig * 100
                improvements.append(improvement)
                
                label_name = TARGET_COLUMNS[i] if i < len(TARGET_COLUMNS) else f"Label_{i}"
                logging.info(f"  {label_name}:")
                logging.info(f"    Before: {dict(zip(unique_orig, counts_orig))} (ratio: {ratio_orig:.1f}:1)")
                logging.info(f"    After:  {dict(zip(unique_res, counts_res))} (ratio: {ratio_res:.1f}:1)")
                logging.info(f"    Improvement: {improvement:.1f}%")
    
    if improvements:
        avg_improvement = np.mean(improvements)
        logging.info(f"  Average imbalance improvement: {avg_improvement:.1f}%")
    
    logging.info(f"  Sample count: {len(y_original)} -> {len(y_resampled)}")


def test_sampling_method(X_train, y_train, method, **kwargs):
    """Test a specific sampling method and return results."""
    logging.info(f"\n{'='*60}")
    logging.info(f"TESTING: {method.upper()} Sampling Method")
    logging.info(f"{'='*60}")
    
    start_time = time()
    
    try:
        # Apply sampling
        X_resampled, y_resampled = apply_proper_multilabel_sampling(
            X_train, y_train, method=method, **kwargs
        )
        
        sampling_time = time() - start_time
        
        # Analyze results
        analyze_class_distribution(y_train, y_resampled, method)
        
        logging.info(f"Sampling completed in {sampling_time:.2f} seconds")
        
        return {
            'method': method,
            'original_samples': len(X_train),
            'resampled_samples': len(X_resampled),
            'sampling_time': sampling_time,
            'success': True
        }
        
    except Exception as e:
        logging.error(f"Error in {method} sampling: {e}")
        return {
            'method': method,
            'success': False,
            'error': str(e)
        }


def test_class_weighting(X_train, y_train):
    """Test class weighting approach."""
    logging.info(f"\n{'='*60}")
    logging.info("TESTING: Class Weighting Approach")
    logging.info(f"{'='*60}")
    
    start_time = time()
    
    try:
        # Calculate class weights
        class_weights = get_multilabel_class_weights(y_train, strategy='balanced')
        
        if class_weights is None:
            logging.error("Failed to calculate class weights")
            return {'success': False}
        
        # Create pipeline with class weights
        pipeline = create_pipeline_with_custom_weights()
        
        if pipeline is None:
            logging.error("Failed to create pipeline with class weights")
            return {'success': False}
        
        # Build model with default parameters
        model = build_model(pipeline, None)
        
        if model is None:
            logging.error("Failed to build model")
            return {'success': False}
        
        weight_time = time() - start_time
        
        logging.info(f"Class weights calculated for {len(class_weights)} labels")
        logging.info(f"Pipeline created with balanced class weights")
        logging.info(f"Class weighting setup completed in {weight_time:.2f} seconds")
        
        return {
            'method': 'class_weighting',
            'n_labels_weighted': len(class_weights),
            'setup_time': weight_time,
            'success': True,
            'model': model
        }
        
    except Exception as e:
        logging.error(f"Error in class weighting: {e}")
        return {
            'method': 'class_weighting',
            'success': False,
            'error': str(e)
        }


def validate_multilabel_sampling(database_filepath):
    """Main validation function for multi-label sampling approaches."""
    
    logging.info("Starting multi-label sampling validation...")
    logging.info(f"Database: {database_filepath}")
    
    # Load data
    logging.info("Loading data...")
    X, Y = load_data(database_filepath)
    
    if X is None or Y is None:
        logging.error("Failed to load data")
        return False
    
    logging.info(f"Loaded {len(X)} samples with {Y.shape[1]} labels")
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    
    logging.info(f"Training samples: {len(X_train)}")
    logging.info(f"Test samples: {len(X_test)}")
    
    # Test results storage
    results = []
    
    # Test sampling methods
    sampling_methods = [
        ('none', {}),
        ('mlsmote', {'k_neighbors': 5, 'sampling_strategy': 0.5}),
        ('random_oversample', {'sampling_strategy': 0.5}),
        ('label_powerset', {'sampling_ratio': 0.5})
    ]
    
    for method, kwargs in sampling_methods:
        result = test_sampling_method(X_train, Y_train, method, **kwargs)
        results.append(result)
    
    # Test class weighting
    class_weight_result = test_class_weighting(X_train, Y_train)
    results.append(class_weight_result)
    
    # Summary
    logging.info(f"\n{'='*60}")
    logging.info("VALIDATION SUMMARY")
    logging.info(f"{'='*60}")
    
    successful_methods = [r for r in results if r.get('success', False)]
    failed_methods = [r for r in results if not r.get('success', False)]
    
    logging.info(f"Successful methods: {len(successful_methods)}")
    logging.info(f"Failed methods: {len(failed_methods)}")
    
    if successful_methods:
        logging.info("\nSuccessful Methods:")
        for result in successful_methods:
            method = result['method']
            if method == 'class_weighting':
                logging.info(f"  [SUCCESS] {method}: {result.get('n_labels_weighted', 0)} labels weighted")
            else:
                orig = result.get('original_samples', 0)
                resampled = result.get('resampled_samples', 0)
                logging.info(f"  [SUCCESS] {method}: {orig} -> {resampled} samples")
    
    if failed_methods:
        logging.info("\nFailed Methods:")
        for result in failed_methods:
            method = result['method']
            error = result.get('error', 'Unknown error')
            logging.info(f"  [FAILED] {method}: {error}")
    
    # Test model training with class weighting (if successful)
    class_weight_model = None
    for result in results:
        if result.get('method') == 'class_weighting' and result.get('success'):
            class_weight_model = result.get('model')
            break
    
    if class_weight_model is not None:
        logging.info(f"\n{'='*60}")
        logging.info("TESTING MODEL TRAINING WITH CLASS WEIGHTS")
        logging.info(f"{'='*60}")
        
        try:
            # Train model
            logging.info("Training model with class weights...")
            train_start = time()
            class_weight_model.fit(X_train, Y_train)
            train_time = time() - train_start
            
            logging.info(f"Model training completed in {train_time:.2f} seconds")
            
            # Quick evaluation
            logging.info("Performing quick evaluation...")
            evaluate_model(class_weight_model, 'class_weighted_validation', 
                         X_test, Y_test, TARGET_COLUMNS[:10])  # Test first 10 labels
            
            logging.info("Class weighting model training and evaluation successful!")
            
        except Exception as e:
            logging.error(f"Error in model training with class weights: {e}")
    
    return len(failed_methods) == 0


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate multi-label sampling methods.')
    parser.add_argument('database_filepath', 
                       help='Path to SQLite database file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run validation
    success = validate_multilabel_sampling(args.database_filepath)
    
    if success:
        print("\n[SUCCESS] All multi-label sampling methods validated successfully!")
        sys.exit(0)
    else:
        print("\n[WARNING] Some validation tests failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()