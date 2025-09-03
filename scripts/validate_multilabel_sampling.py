#!/usr/bin/env python3
"""
Multi-label sampling validation and testing script.

This script validates the multi-label sampling implementations and provides
comprehensive testing of all sampling methods against baseline performance.
"""

import sys
import os
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from disaster_classifier.data.loader import load_data
from disaster_classifier.models.samplers import (
    apply_proper_multilabel_sampling,
    get_multilabel_class_weights
)
from disaster_classifier.models.pipeline import create_pipeline, build_model
from disaster_classifier.utils.config import setup_logging, TARGET_COLUMNS


def analyze_class_distribution(y_before, y_after, method_name):
    """Analyze and log class distribution changes."""
    logging.info(f"\n=== Class Distribution Analysis for {method_name} ===")
    
    n_labels = y_before.shape[1]
    improvements = []
    
    for i in range(n_labels):
        # Before distribution
        unique_before, counts_before = np.unique(y_before[:, i], return_counts=True)
        if len(unique_before) == 2:
            ratio_before = max(counts_before) / min(counts_before)
        else:
            ratio_before = 0
            
        # After distribution
        unique_after, counts_after = np.unique(y_after[:, i], return_counts=True)
        if len(unique_after) == 2:
            ratio_after = max(counts_after) / min(counts_after)
        else:
            ratio_after = 0
            
        # Calculate improvement
        if ratio_before > 0 and ratio_after > 0:
            improvement = (ratio_before - ratio_after) / ratio_before * 100
            improvements.append(improvement)
            
            if improvement > 10:  # Significant improvement
                logging.info(f"  {TARGET_COLUMNS[i]}: {ratio_before:.1f}:1 -> {ratio_after:.1f}:1 ({improvement:.1f}% improvement)")
    
    if improvements:
        avg_improvement = np.mean(improvements)
        logging.info(f"Average imbalance reduction: {avg_improvement:.1f}%")


def validate_multilabel_sampling(database_filepath):
    """Validate multi-label sampling approaches."""
    
    logging.info("=== Multi-Label Sampling Validation ===")
    
    # Load data
    logging.info("Loading data...")
    X, Y = load_data(database_filepath)
    if X is None or Y is None:
        logging.error("Error loading data from database")
        return False
        
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    
    logging.info(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Test each sampling method
    methods = ['none', 'mlsmote', 'random_oversample', 'label_powerset']
    results = {}
    
    for method in methods:
        logging.info(f"\n{'='*50}")
        logging.info(f"Testing {method.upper()} sampling...")
        logging.info(f"{'='*50}")
        
        try:
            # Apply sampling
            X_resampled, Y_resampled = apply_proper_multilabel_sampling(
                X_train, Y_train, method=method,
                k_neighbors=3,  # Conservative for sparse classes
                sampling_strategy=0.5  # Moderate oversampling
            )
            
            # Analyze class distribution changes
            analyze_class_distribution(Y_train, Y_resampled, method)
            
            # Quick model training and evaluation
            logging.info("Training quick model for performance comparison...")
            pipeline = create_pipeline()
            model = build_model(pipeline, None)
            model.fit(X_resampled, Y_resampled)
            
            # Predictions
            Y_pred = model.predict(X_test)
            
            # Calculate F1 scores
            f1_micro = f1_score(Y_test, Y_pred, average='micro')
            f1_macro = f1_score(Y_test, Y_pred, average='macro')
            
            results[method] = {
                'f1_micro': f1_micro,
                'f1_macro': f1_macro,
                'samples_before': len(X_train),
                'samples_after': len(X_resampled),
                'success': True
            }
            
            logging.info(f"Performance - F1 Micro: {f1_micro:.3f}, F1 Macro: {f1_macro:.3f}")
            logging.info(f"Sample count: {len(X_train)} -> {len(X_resampled)}")
            
        except Exception as e:
            logging.error(f"Error testing {method}: {e}")
            results[method] = {'success': False, 'error': str(e)}
    
    # Test class weighting approach
    logging.info(f"\n{'='*50}")
    logging.info("Testing CLASS WEIGHTING approach...")
    logging.info(f"{'='*50}")
    
    try:
        class_weights = get_multilabel_class_weights(Y_train, strategy='balanced')
        logging.info(f"Generated class weights for {len(class_weights)} labels")
        
        # Note: Class weighting requires pipeline modification to be fully tested
        # This is a placeholder for the next phase
        results['class_weighting'] = {
            'weights_generated': True,
            'n_labels': len(class_weights),
            'note': 'Requires pipeline modification for full testing'
        }
        
    except Exception as e:
        logging.error(f"Error generating class weights: {e}")
        results['class_weighting'] = {'success': False, 'error': str(e)}
    
    # Summary
    logging.info(f"\n{'='*60}")
    logging.info("VALIDATION SUMMARY")
    logging.info(f"{'='*60}")
    
    for method, result in results.items():
        if result.get('success', False):
            if 'f1_micro' in result:
                logging.info(f"{method.upper()}: F1 Micro={result['f1_micro']:.3f}, F1 Macro={result['f1_macro']:.3f}")
            else:
                logging.info(f"{method.upper()}: {result}")
        else:
            logging.error(f"{method.upper()}: FAILED - {result.get('error', 'Unknown error')}")
    
    return True


def main():
    """Main function."""
    setup_logging()
    
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_multilabel_sampling.py <database_filepath>")
        print("Example: python scripts/validate_multilabel_sampling.py data/02_stg/stg_disaster_response.db")
        return
    
    database_filepath = sys.argv[1]
    
    if not os.path.exists(database_filepath):
        logging.error(f"Database file not found: {database_filepath}")
        return
    
    success = validate_multilabel_sampling(database_filepath)
    
    if success:
        logging.info("\n✅ Multi-label sampling validation completed successfully!")
        print("\n🎯 Next steps:")
        print("1. Review the validation results above")
        print("2. Run: python scripts/create_model.py --out models/validation_test.pkl")
        print("3. Test model loading and inference")
        print("4. Modify pipeline for class weighting support")
    else:
        logging.error("\n❌ Validation failed. Check the logs above for details.")


if __name__ == "__main__":
    main()
