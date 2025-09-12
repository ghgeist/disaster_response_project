#!/usr/bin/env python3
"""
Test script for Google Drive model deployment.
Run this after uploading model to Google Drive and setting GDRIVE_MODEL_ID.

Usage:
    export GDRIVE_MODEL_ID="your_file_id_here"
    python test_gdrive_deployment.py
"""
import os
import sys
from pathlib import Path

# Add paths for imports
sys.path.append('app')
sys.path.append('src')

def test_gdrive_deployment():
    """Test complete Google Drive deployment flow."""
    print("🧪 Testing Google Drive Model Deployment")
    print("=" * 50)
    
    # Check environment variable
    gdrive_id = os.environ.get('GDRIVE_MODEL_ID')
    if not gdrive_id or gdrive_id.strip() in {'', 'YOUR_FILE_ID', 'YOUR_GOOGLE_DRIVE_FILE_ID'}:
        print("❌ GDRIVE_MODEL_ID not set properly")
        print("Please run: export GDRIVE_MODEL_ID='your_actual_file_id'")
        return False
    
    print(f"✅ GDRIVE_MODEL_ID set: {gdrive_id[:10]}...")
    
    try:
        from app.services import ModelService
        
        # Test with a temporary path (will trigger download)
        temp_model_path = Path('temp_test_model.pkl')
        if temp_model_path.exists():
            temp_model_path.unlink()  # Remove if exists
        
        print("\n📥 Testing Google Drive download...")
        model_service = ModelService(temp_model_path, gdrive_id)
        
        # This should trigger download from Google Drive
        model = model_service.load_model()
        print("✅ SUCCESS: Model downloaded and loaded from Google Drive")
        
        # Test prediction
        print("\n🔮 Testing prediction...")
        test_messages = [
            "We need urgent medical supplies for earthquake victims",
            "Food and water running low in shelter area",
            "Search and rescue teams needed immediately"
        ]
        
        for msg in test_messages:
            result = model_service.predict(msg)
            positive_count = sum(1 for v in result.values() if v == 1)
            print(f"✅ '{msg[:40]}...' -> {positive_count} categories activated")
        
        # Cleanup
        if temp_model_path.exists():
            temp_model_path.unlink()
            print("\n🧹 Cleaned up temporary files")
        
        print("\n🎉 Google Drive deployment test PASSED!")
        print("Your model is ready for production deployment!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        temp_model_path = Path('temp_test_model.pkl')
        if temp_model_path.exists():
            temp_model_path.unlink()
        
        return False

if __name__ == "__main__":
    success = test_gdrive_deployment()
    sys.exit(0 if success else 1)
