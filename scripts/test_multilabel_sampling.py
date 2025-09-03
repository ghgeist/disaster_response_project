#!/usr/bin/env python3
"""
Test script for multi-label aware sampling strategies.

This script demonstrates how to use the proper multi-label sampling approaches
instead of the failing SMOTE/ADASYN methods.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from disaster_classifier.data.loader import load_data
from disaster_classifier.models.samplers import (
    apply_proper_multilabel_sampling,
    get_multilabel_class_weights
)
from disaster_classifier.models.pipeline import create_pipeline
from disaster_classifier.utils.config import setup_logging, TARGET_COLUMNS
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import logging


def test_sampling_methods(database_filepath):
    """Test different multi-label sampling methods."""
    
    # Load data
    logging.info("Loading data...")
    X, Y = load_data(database_filepath)
    if X is None or Y is None:
        logging.error("Failed to load data")
        return
        
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    
    logging.info(f"Original training set size: {len(X_train)}")
    
    # Test different sampling methods
    sampling_methods = ['none', 'mlsmote', 'random_oversample', 'label_powerset']
    
    for method in sampling_methods:
        logging.info(f"\n{'='*50}")
        logging.info(f"Testing {method} sampling...")
        logging.info(f"{'='*50}")
        
        try:
            # Apply sampling
            X_resampled, Y_resampled = apply_proper_multilabel_sampling(
                X_train, Y_train,
                method=method,
                k_neighbors=5,
                sampling_strategy=0.5
            )
            
            logging.info(f"Resampled size: {len(X_resampled)}")
            
            # Quick test with simple classifier
            if method == 'none':
                # Also test class weighting approach
                logging.info("\nTesting class weighting approach...")
                class_weights = get_multilabel_class_weights(Y_train)
                logging.info(f"Generated class weights for {len(class_weights)} labels")
                
        except Exception as e:
            logging.error(f"Error with {method}: {e}")


def test_with_classifier(database_filepath):
    """Test the sampling with an actual classifier."""
    
    logging.info("\nTesting with actual classifier...")
    
    # Load and split data
    X, Y = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    
    # Create pipeline
    pipeline = create_pipeline()
    
    # Option 1: Use class weighting (recommended)
    logging.info("\nOption 1: Testing with class weights...")
    class_weights = get_multilabel_class_weights(Y_train)
    
    # Note: To use class weights, you would need to modify your pipeline
    # to support sample_weight parameter in fit() method
    
    # Option 2: Use ML-SMOTE
    logging.info("\nOption 2: Testing with ML-SMOTE...")
    X_train_balanced, Y_train_balanced = apply_proper_multilabel_sampling(
        X_train, Y_train,
        method='mlsmote',
        k_neighbors=3,  # Lower for sparse minority classes
        sampling_strategy=0.3  # Conservative ratio
    )
    
    # Train model on balanced data
    logging.info("Training model on balanced data...")
    try:
        pipeline.fit(X_train_balanced, Y_train_balanced)
        logging.info("Model training completed successfully!")
        
        # Make predictions
        Y_pred = pipeline.predict(X_test[:100])  # Test on subset
        logging.info(f"Predictions shape: {Y_pred.shape}")
        
    except Exception as e:
        logging.error(f"Error during model training: {e}")


def main():
    """Main function."""
    setup_logging()
    
    if len(sys.argv) < 2:
        print("Usage: python test_multilabel_sampling.py database.db")
        sys.exit(1)
        
    database_filepath = sys.argv[1]
    
    # Test sampling methods
    test_sampling_methods(database_filepath)
    
    # Test with classifier
    test_with_classifier(database_filepath)
    
    logging.info("\nTesting completed!")


if __name__ == "__main__":
    main()
