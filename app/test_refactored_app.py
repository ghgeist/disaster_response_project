"""
Simple test to verify the refactored application structure.
"""
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_app_creation():
    """Test that the application can be created successfully."""
    try:
        from app import create_app
        
        # Create app with testing configuration
        app = create_app('testing')
        
        # Basic checks
        assert app is not None
        assert app.config['TESTING'] is True
        assert app.config['DEBUG'] is True
        
        print("✓ Application creation test passed")
        return True
        
    except Exception as e:
        print(f"✗ Application creation test failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    try:
        from app.config import config, DevelopmentConfig, ProductionConfig, TestingConfig
        
        # Test configuration classes exist
        assert DevelopmentConfig is not None
        assert ProductionConfig is not None
        assert TestingConfig is not None
        
        # Test configuration mapping
        assert 'development' in config
        assert 'production' in config
        assert 'testing' in config
        
        print("✓ Configuration loading test passed")
        return True
        
    except Exception as e:
        print(f"✗ Configuration loading test failed: {e}")
        return False

def test_blueprint_registration():
    """Test that blueprints are registered."""
    try:
        from app import create_app
        
        app = create_app('testing')
        
        # Check that main blueprint is registered
        assert 'main' in [bp.name for bp in app.blueprints.values()]
        
        print("✓ Blueprint registration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Blueprint registration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing refactored application structure...")
    print("=" * 50)
    
    tests = [
        test_config_loading,
        test_app_creation,
        test_blueprint_registration,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The refactored structure is working correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
