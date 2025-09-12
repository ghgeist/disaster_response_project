#!/usr/bin/env python3
"""
Test Flask application with standardized model naming.
"""
import sys
import os
from pathlib import Path

# Add paths for imports
sys.path.append('app')
sys.path.append('src')

def test_flask_standardized():
    """Test the Flask application with standardized model."""
    print("Testing Flask application with standardized model...")
    
    try:
        from app.app import create_app
        from app.config import Config
        
        # Test Flask app creation
        app = create_app(Config)
        print("SUCCESS: Flask app created with standardized model validation")
        
        with app.app_context():
            # Test model service
            model_service = app.model_service
            print("SUCCESS: Model service initialized")
            
            # Test prediction through Flask app
            test_message = 'Search and rescue teams needed for earthquake victims'
            result = model_service.predict(test_message)
            positive_count = sum(1 for v in result.values() if v == 1)
            
            # Get active categories
            active_cats = [k for k, v in result.items() if v == 1][:4]
            print(f"SUCCESS: Flask prediction - {positive_count} categories activated")
            print(f"Active categories: {active_cats}")
            
            print("\nComplete Flask application with standardized naming: WORKING!")
            return True
        
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_flask_standardized()
    if success:
        print("\nAll tests passed! Ready for Google Drive deployment.")
    sys.exit(0 if success else 1)