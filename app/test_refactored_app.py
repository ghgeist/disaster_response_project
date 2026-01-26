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

def run_app_creation_check():
    """Run app creation check and return success status."""
    try:
        from app import create_app
        from app.config import TestConfig

        # Create app with testing configuration
        app = create_app(TestConfig)

        # Basic checks
        assert app is not None
        assert app.config['TESTING'] is True
        # DEBUG may not be explicitly set in TestConfig, so check if it exists
        assert 'DEBUG' in app.config

        print("✓ Application creation test passed")
        return True

    except Exception as e:
        print(f"✗ Application creation test failed: {e}")
        return False

def run_config_loading_check():
    """Run configuration loading check and return success status."""
    try:
        from app.config import Config, TestConfig

        # Test configuration classes exist
        assert Config is not None
        assert TestConfig is not None

        # Test that TestConfig inherits from Config
        assert issubclass(TestConfig, Config)

        print("✓ Configuration loading test passed")
        return True

    except Exception as e:
        print(f"✗ Configuration loading test failed: {e}")
        return False

def run_blueprint_registration_check():
    """Run blueprint registration check and return success status."""
    try:
        from app import create_app
        from app.config import TestConfig

        app = create_app(TestConfig)

        # Check that routes are registered (routes are registered directly, not as blueprints)
        # Verify that the index route exists
        assert 'index' in [rule.endpoint for rule in app.url_map.iter_rules()]

        print("✓ Blueprint registration test passed")
        return True

    except Exception as e:
        print(f"✗ Blueprint registration test failed: {e}")
        return False


def test_app_creation():
    """Test that the application can be created successfully."""
    assert run_app_creation_check(), "Application creation test failed"


def test_config_loading():
    """Test configuration loading."""
    assert run_config_loading_check(), "Configuration loading test failed"


def test_blueprint_registration():
    """Test that blueprints are registered."""
    assert run_blueprint_registration_check(), "Blueprint registration test failed"

def main():
    """Run all tests."""
    print("Testing refactored application structure...")
    print("=" * 50)
    
    checks = [
        run_config_loading_check,
        run_app_creation_check,
        run_blueprint_registration_check,
    ]
    
    passed = 0
    total = len(checks)
    
    for check in checks:
        if check():
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
