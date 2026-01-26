#!/usr/bin/env python3
"""
Production model validation script.
Comprehensive testing before deployment.
"""
# Standard library imports
import json
import os
import sys
import time
from pathlib import Path

# Third-party imports
import joblib
import psutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

# Local imports
from disasterproject.utils.config import TARGET_COLUMNS
from services import ModelService

def test_model_loading():
    """Test model loading performance and memory usage."""
    print("🔍 Testing Model Loading...")
    
    model_path = Path('model/disaster_rf_v1-2-0_prod_2025-09-11.pkl')
    if not model_path.exists():
        print("❌ Model file not found!")
        return False
    
    # Test load time
    start_time = time.time()
    try:
        model = joblib.load(model_path)
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.3f}s")
        
        # Check if within threshold
        if load_time > 0.2:
            print(f"⚠️ Load time {load_time:.3f}s exceeds 0.2s threshold")
        
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_model_size():
    """Test model file size."""
    print("\n📏 Testing Model Size...")
    
    model_path = Path('model/disaster_rf_v1-2-0_prod_2025-09-11.pkl')
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"✅ Model size: {size_mb:.1f} MB")
    
    if size_mb > 50:
        print(f"⚠️ Model size {size_mb:.1f}MB exceeds 50MB threshold")
        return False
    
    return True

def test_prediction_functionality():
    """Test basic prediction functionality."""
    print("\n🧪 Testing Prediction Functionality...")
    
    try:
        model_service = ModelService(Path('model/disaster_rf_v1-2-0_prod_2025-09-11.pkl'))
        
        # Test cases
        test_cases = [
            "People trapped on roof. Send search and rescue.",
            "Storm destroyed houses",
            "Hospital is closed",
            "We need medical help",
            "No water available"
        ]
        
        all_passed = True
        for i, test_case in enumerate(test_cases, 1):
            try:
                result = model_service.predict(test_case)
                labels = result.get('labels', {})
                print(f"✅ Test {i}: Prediction successful ({len(labels)} labels)")

                # Check if all expected labels are present
                if len(labels) != len(TARGET_COLUMNS):
                    print(f"⚠️ Expected {len(TARGET_COLUMNS)} labels, got {len(labels)}")
                    all_passed = False
                    
            except Exception as e:
                print(f"❌ Test {i} failed: {e}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ ModelService initialization failed: {e}")
        return False

def test_critical_labels():
    """Test that all critical labels can be predicted."""
    print("\n🎯 Testing Critical Labels...")
    
    critical_labels = [
        'medical_help', 'search_and_rescue', 'water', 'food',
        'shelter', 'hospitals', 'security', 'weather_related'
    ]
    
    try:
        model_service = ModelService(Path('model/disaster_rf_v1-2-0_prod_2025-09-11.pkl'))
        
        # Test messages designed to trigger each critical label
        test_messages = {
            'medical_help': "We need medical assistance urgently",
            'search_and_rescue': "People trapped in building",
            'water': "No clean water available",
            'food': "Food supplies running low",
            'shelter': "Need emergency shelter",
            'hospitals': "Hospital damaged in earthquake",
            'security': "Security situation deteriorating",
            'weather_related': "Hurricane approaching coast"
        }
        
        all_passed = True
        for label, message in test_messages.items():
            result = model_service.predict(message)
            labels = result.get('labels', {})
            if label in labels:
                status = "✅" if labels[label] else "⚠️"
                print(f"{status} {label}: {labels[label]}")
            else:
                print(f"❌ {label}: Not found in predictions")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Critical labels test failed: {e}")
        return False

def test_required_files():
    """Test that all required model files exist."""
    print("\n📁 Testing Required Files...")
    
    required_files = [
        'model/disaster_rf_v1-2-0_prod_2025-09-11.pkl',
        'model/disaster_rf_v1-2-0_prod_2025-09-11_thresholds.json',
        'model/disaster_rf_v1-2-0_prod_2025-09-11_labels.json',
        'model/disaster_rf_v1-2-0_prod_2025-09-11_training.json'
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            all_exist = False
    
    return all_exist

def test_memory_usage():
    """Test memory usage during model loading."""
    print("\n💾 Testing Memory Usage...")
    
    # Get initial memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    try:
        # Load model and measure memory
        model = joblib.load('model/disaster_rf_v1-2-0_prod_2025-09-11.pkl')
        peak_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = peak_memory - initial_memory
        
        print(f"✅ Memory usage: {memory_increase:.1f} MB increase")
        
        if memory_increase > 100:
            print(f"⚠️ Memory increase {memory_increase:.1f}MB exceeds 100MB threshold")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🚀 Production Model Validation")
    print("=" * 50)
    
    tests = [
        ("Required Files", test_required_files),
        ("Model Size", test_model_size),
        ("Model Loading", test_model_loading),
        ("Memory Usage", test_memory_usage),
        ("Prediction Functionality", test_prediction_functionality),
        ("Critical Labels", test_critical_labels)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - READY FOR PRODUCTION DEPLOYMENT!")
        return 0
    else:
        print("⚠️ SOME TESTS FAILED - REVIEW BEFORE DEPLOYMENT")
        return 1

if __name__ == '__main__':
    sys.exit(main())
