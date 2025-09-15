#!/usr/bin/env python3
"""
System validation script for disaster response classification.

This script validates the end-to-end system functionality including:
- Model training
- Model loading and inference
- Application integration
"""

import sys
import os
import logging
import pickle
import tempfile
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from disasterproject.utils.config import setup_logging, TARGET_COLUMNS
from disasterproject.data.loader import load_data
from disasterproject.models.pipeline import create_pipeline, build_model
from disasterproject.data.preprocessor import tokenize
from sklearn.model_selection import train_test_split


def test_model_training(database_filepath):
    """Test model training functionality."""
    logging.info("=== Testing Model Training ===")
    
    try:
        # Load data
        X, Y = load_data(database_filepath)
        if X is None or Y is None:
            logging.error("Failed to load data")
            return False
            
        # Split data
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
        
        # Create and train model
        pipeline = create_pipeline()
        model = build_model(pipeline, None)
        model.fit(X_train, Y_train)
        
        logging.info(f"✅ Model training successful")
        logging.info(f"   Training samples: {len(X_train)}")
        logging.info(f"   Test samples: {len(X_test)}")
        logging.info(f"   Target columns: {len(TARGET_COLUMNS)}")
        
        return model, X_test, Y_test
        
    except Exception as e:
        logging.error(f"❌ Model training failed: {e}")
        return False


def test_model_inference(model, X_test, Y_test):
    """Test model inference functionality."""
    logging.info("\n=== Testing Model Inference ===")
    
    try:
        # Test predictions
        predictions = model.predict(X_test)
        
        # Test with sample disaster message
        test_message = 'We need urgent medical help and clean water for 50 people'
        sample_prediction = model.predict([test_message])
        
        logging.info(f"✅ Model inference successful")
        logging.info(f"   Predictions shape: {predictions.shape}")
        logging.info(f"   Sample prediction shape: {sample_prediction.shape}")
        logging.info(f"   Sample prediction (first 5 labels): {sample_prediction[0][:5]}")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Model inference failed: {e}")
        return False


def test_model_serialization(model):
    """Test model saving and loading."""
    logging.info("\n=== Testing Model Serialization ===")
    
    try:
        # Save model to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            model_path = tmp_file.name
            
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Load model back
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Test loaded model
        test_message = 'Emergency: Building collapse, need rescue team immediately'
        original_pred = model.predict([test_message])
        loaded_pred = loaded_model.predict([test_message])
        
        # Compare predictions
        predictions_match = np.array_equal(original_pred, loaded_pred)
        
        # Clean up
        os.unlink(model_path)
        
        if predictions_match:
            logging.info("✅ Model serialization successful")
            logging.info("   Saved and loaded model predictions match")
            return True
        else:
            logging.error("❌ Model serialization failed - predictions don't match")
            return False
            
    except Exception as e:
        logging.error(f"❌ Model serialization failed: {e}")
        return False


def test_application_integration():
    """Test Flask application integration."""
    logging.info("\n=== Testing Application Integration ===")
    
    try:
        # Test app import
        from app.app import create_app
        
        # Create app instance
        app = create_app()
        
        # Test app configuration
        with app.app_context():
            logging.info("✅ Application integration successful")
            logging.info("   Flask app created successfully")
            logging.info("   App context working")
            
        return True
        
    except Exception as e:
        logging.error(f"❌ Application integration failed: {e}")
        return False


def main():
    """Main validation function."""
    setup_logging()
    
    if len(sys.argv) != 2:
        print("Usage: python scripts/system_validation.py <database_filepath>")
        print("Example: python scripts/system_validation.py data/02_stg/stg_disaster_response.db")
        return
    
    database_filepath = sys.argv[1]
    
    if not os.path.exists(database_filepath):
        logging.error(f"Database file not found: {database_filepath}")
        return
    
    logging.info("🚀 Starting System Validation")
    logging.info("=" * 50)
    
    # Test 1: Model Training
    training_result = test_model_training(database_filepath)
    if not training_result:
        logging.error("❌ System validation failed at model training")
        return
    
    model, X_test, Y_test = training_result
    
    # Test 2: Model Inference
    inference_success = test_model_inference(model, X_test, Y_test)
    if not inference_success:
        logging.error("❌ System validation failed at model inference")
        return
    
    # Test 3: Model Serialization
    serialization_success = test_model_serialization(model)
    if not serialization_success:
        logging.error("❌ System validation failed at model serialization")
        return
    
    # Test 4: Application Integration
    app_success = test_application_integration()
    if not app_success:
        logging.error("❌ System validation failed at application integration")
        return
    
    # Success summary
    logging.info("\n" + "=" * 50)
    logging.info("🎉 SYSTEM VALIDATION COMPLETED SUCCESSFULLY!")
    logging.info("=" * 50)
    logging.info("✅ All core systems are working correctly:")
    logging.info("   • Model training pipeline")
    logging.info("   • Model inference and predictions")
    logging.info("   • Model serialization (save/load)")
    logging.info("   • Flask application integration")
    
    print("\n🎯 Next Steps:")
    print("1. Run multi-label sampling validation:")
    print(f"   python scripts/validate_multilabel_sampling.py {database_filepath}")
    print("2. Create a validation model:")
    print("   python scripts/create_model.py --out model/validation_test.pkl")
    print("3. Test the web application:")
    print("   cd app && python app.py")


if __name__ == "__main__":
    main()
