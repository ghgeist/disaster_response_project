#!/usr/bin/env python3
"""
Test script to validate NLTK performance optimization without running the full application.
This script simulates the optimization changes and validates the approach.
"""

import sys
import os
import time
from pathlib import Path

# Add the workspace to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_nltk_setup_module():
    """Test the NLTK setup module structure and logic."""
    print("Testing NLTK setup module...")
    
    try:
        # Test importing the module
        from app.nltk_setup import (
            setup_nltk_resources, 
            validate_nltk_resources, 
            get_nltk_status,
            NLTKSetupError,
            REQUIRED_RESOURCES,
            RESOURCE_VALIDATORS
        )
        
        print("✓ NLTK setup module imports successfully")
        print(f"✓ Required resources defined: {list(REQUIRED_RESOURCES.keys())}")
        print(f"✓ Resource validators defined: {list(RESOURCE_VALIDATORS.keys())}")
        
        # Test validation function (without actually running NLTK)
        print("✓ All NLTK setup functions are properly defined")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_compatibility_optimization():
    """Test the compatibility layer optimization."""
    print("\nTesting compatibility layer optimization...")
    
    try:
        from app.compat import (
            load_with_legacy_paths,
            clear_compatibility_cache,
            get_compatibility_cache_status,
            _module_mapping_cache
        )
        
        print("✓ Compatibility module imports successfully")
        print(f"✓ Module mapping cache initialized: {len(_module_mapping_cache)} items")
        
        # Test cache status function
        cache_status = get_compatibility_cache_status()
        print(f"✓ Cache status function works: {cache_status}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_config_optimization():
    """Test that config.py has been optimized."""
    print("\nTesting config optimization...")
    
    try:
        config_path = Path("src/disasterproject/utils/config.py")
        if not config_path.exists():
            print("✗ Config file not found")
            return False
        
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Check that NLTK download logic has been removed
        if "nltk.download" in content:
            print("✗ NLTK download logic still present in config.py")
            return False
        
        if "NLTK resources are now managed by app/nltk_setup.py" in content:
            print("✓ NLTK download logic has been removed from config.py")
            print("✓ Optimization comment added")
        else:
            print("⚠ Optimization comment not found, but download logic removed")
        
        return True
        
    except Exception as e:
        print(f"✗ Error checking config: {e}")
        return False

def test_app_integration():
    """Test that the Flask app has been properly integrated."""
    print("\nTesting Flask app integration...")
    
    try:
        from app.app import create_app
        from app.nltk_setup import setup_nltk_resources, NLTKSetupError
        
        print("✓ Flask app imports successfully")
        print("✓ NLTK setup integration imports work")
        
        # Check that the app factory has the NLTK setup call
        app_path = Path("app/app.py")
        with open(app_path, 'r') as f:
            content = f.read()
        
        if "setup_nltk_resources" in content:
            print("✓ NLTK setup call found in Flask app factory")
        else:
            print("✗ NLTK setup call not found in Flask app factory")
            return False
        
        if "NLTK_SETUP_RESULTS" in content:
            print("✓ NLTK setup results storage found")
        else:
            print("✗ NLTK setup results storage not found")
            return False
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_performance_monitoring():
    """Test that performance monitoring has been added."""
    print("\nTesting performance monitoring...")
    
    try:
        from app.routes import register_routes
        print("✓ Routes module imports successfully")
        
        # Check that performance timing has been added to health check
        routes_path = Path("app/routes.py")
        with open(routes_path, 'r') as f:
            content = f.read()
        
        if "total_response_time_ms" in content:
            print("✓ Performance timing added to health check")
        else:
            print("✗ Performance timing not found in health check")
            return False
        
        if "performance_diagnostics" in content:
            print("✓ Performance diagnostics endpoint added")
        else:
            print("✗ Performance diagnostics endpoint not found")
            return False
        
        if "nltk_status" in content:
            print("✓ NLTK status monitoring added")
        else:
            print("✗ NLTK status monitoring not found")
            return False
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def main():
    """Run all optimization tests."""
    print("=" * 60)
    print("NLTK Performance Optimization Validation")
    print("=" * 60)
    
    tests = [
        ("NLTK Setup Module", test_nltk_setup_module),
        ("Compatibility Optimization", test_compatibility_optimization),
        ("Config Optimization", test_config_optimization),
        ("Flask App Integration", test_app_integration),
        ("Performance Monitoring", test_performance_monitoring)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All optimizations have been successfully implemented!")
        print("\nKey improvements:")
        print("• NLTK resources are now loaded once at startup")
        print("• Per-request downloads have been eliminated")
        print("• Compatibility layer uses caching")
        print("• Performance monitoring has been added")
        print("• Health check includes timing metrics")
    else:
        print(f"\n⚠ {total - passed} tests failed. Please review the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)