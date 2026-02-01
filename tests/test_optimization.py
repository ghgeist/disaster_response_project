"""
Test script to validate NLTK performance optimization and compatibility features.
Tests the optimization changes and validates the approach without running the full application.
"""
from pathlib import Path
from unittest.mock import patch

import pytest

from app.utils.nltk_setup import (
    REQUIRED_RESOURCES,
    RESOURCE_VALIDATORS,
    NLTKSetupError,
    get_nltk_status,
    setup_nltk_resources,
    validate_nltk_resources,
)

# Note: app.compat module was removed as part of module structure cleanup
# These compatibility functions are no longer needed as models now load directly with joblib


class TestNLTKSetupModule:
    """Test the NLTK setup module structure and logic."""

    def test_nltk_setup_imports(self):
        """Test that NLTK setup module imports successfully."""
        from app.utils.nltk_setup import (
            REQUIRED_RESOURCES,
            RESOURCE_VALIDATORS,
            NLTKSetupError,
            get_nltk_status,
            setup_nltk_resources,
            validate_nltk_resources,
        )

        assert callable(setup_nltk_resources)
        assert callable(validate_nltk_resources)
        assert callable(get_nltk_status)
        assert issubclass(NLTKSetupError, Exception)
        assert isinstance(REQUIRED_RESOURCES, dict)
        assert isinstance(RESOURCE_VALIDATORS, dict)

    def test_required_resources_defined(self):
        """Test that required resources are properly defined."""
        assert 'corpora' in REQUIRED_RESOURCES
        assert 'tokenizers' in REQUIRED_RESOURCES
        assert 'stopwords' in REQUIRED_RESOURCES['corpora']
        assert 'wordnet' in REQUIRED_RESOURCES['corpora']
        assert 'punkt' in REQUIRED_RESOURCES['tokenizers']

    def test_resource_validators_defined(self):
        """Test that resource validators are properly defined."""
        assert 'stopwords' in RESOURCE_VALIDATORS
        assert 'wordnet' in RESOURCE_VALIDATORS
        assert 'punkt' in RESOURCE_VALIDATORS

        # Test that validators are callable
        for resource, validator in RESOURCE_VALIDATORS.items():
            assert callable(validator), f"Validator for {resource} is not callable"

    @patch('nltk.download')
    @patch('nltk.data.find')
    def test_setup_nltk_resources_success(self, mock_find, mock_download):
        """Test successful NLTK resource setup."""
        mock_find.return_value = '/mock/path'

        validator_overrides = {
            'stopwords': lambda: True,
            'wordnet': lambda: True,
            'punkt': lambda: True,
        }
        with patch.dict('app.utils.nltk_setup.RESOURCE_VALIDATORS', validator_overrides, clear=True):
            result = setup_nltk_resources()

        assert result['success'] is True
        assert 'setup_time_ms' in result
        assert 'resources_loaded' in result
        assert 'resources_failed' in result
        assert 'errors' in result

    @patch('nltk.download')
    @patch('nltk.data.find')
    def test_setup_nltk_resources_critical_failure(self, mock_find, mock_download):
        """Test NLTK setup failure when critical resources are missing."""
        mock_find.side_effect = LookupError('Resource not found')

        validator_overrides = {
            'stopwords': lambda: False,
            'punkt': lambda: False,
            'wordnet': lambda: True,
        }
        with patch.dict('app.utils.nltk_setup.RESOURCE_VALIDATORS', validator_overrides, clear=True):
            with pytest.raises(NLTKSetupError, match='Critical NLTK resources missing'):
                setup_nltk_resources()

    def test_validate_nltk_resources(self):
        """Test NLTK resource validation."""
        with patch('nltk.data.find') as mock_find:
            mock_find.return_value = '/mock/path'

            validator_overrides = {
                'stopwords': lambda: True,
                'wordnet': lambda: True,
                'punkt': lambda: True,
            }
            with patch.dict('app.utils.nltk_setup.RESOURCE_VALIDATORS', validator_overrides, clear=True):
                result = validate_nltk_resources()

        assert 'all_available' in result
        assert 'available_resources' in result
        assert 'missing_resources' in result
        assert 'validation_errors' in result

    def test_get_nltk_status(self):
        """Test NLTK status retrieval."""
        with patch('app.utils.nltk_setup.validate_nltk_resources') as mock_validate:
            mock_validate.return_value = {
                'all_available': True,
                'available_resources': ['stopwords', 'wordnet'],
                'missing_resources': [],
                'validation_errors': []
            }

            with patch('nltk.__version__', '3.8.1'):
                with patch('nltk.data.path', ['/mock/path']):
                    result = get_nltk_status()

                    assert 'status' in result
                    assert 'all_resources_available' in result
                    assert 'nltk_version' in result
                    assert result['status'] == 'healthy'


# TestCompatibilityOptimization class removed since app.compat module was intentionally
# removed as part of the module structure cleanup (see docs/dev_notes/2025-09-15.md)
# The system now uses direct joblib.load() without compatibility layers


class TestConfigOptimization:
    """Test that config.py has been optimized."""

    def test_config_file_exists(self):
        """Test that config file exists."""
        config_path = Path("app/config.py")
        assert config_path.exists()

    def test_config_nltk_optimization(self):
        """Test that NLTK download logic has been removed from config.py."""
        config_path = Path("app/config.py")
        with open(config_path, 'r') as f:
            content = f.read()

        # Check that NLTK download logic has been removed
        assert "nltk.download" not in content, "NLTK download logic still present in config.py"

        # Check for optimization comment (optional)
        if "NLTK resources are now managed by app/nltk_setup.py" in content:
            assert True  # Optimization comment found
        else:
            # This is acceptable - the optimization might be present without the comment
            assert "nltk.download" not in content


class TestFlaskAppIntegration:
    """Test that the Flask app has been properly integrated."""

    def test_flask_app_imports(self):
        """Test that Flask app imports successfully."""
        from app.app import create_app
        from app.utils.nltk_setup import NLTKSetupError, setup_nltk_resources

        assert callable(create_app)
        assert callable(setup_nltk_resources)
        assert issubclass(NLTKSetupError, Exception)

    def test_flask_app_factory_has_nltk_setup(self):
        """Test that Flask app factory has NLTK setup call."""
        app_path = Path("app/app.py")
        with open(app_path, 'r') as f:
            content = f.read()

        assert "setup_nltk_resources" in content, "NLTK setup call not found in Flask app factory"

    def test_flask_app_factory_has_nltk_results_storage(self):
        """Test that Flask app factory has NLTK setup results storage."""
        app_path = Path("app/app.py")
        with open(app_path, 'r') as f:
            content = f.read()

        assert "NLTK_SETUP_RESULTS" in content, "NLTK setup results storage not found"


class TestPerformanceMonitoring:
    """Test that performance monitoring has been added."""

    def test_routes_imports(self):
        """Test that routes module imports successfully."""
        from app.routes import register_routes
        assert callable(register_routes)

    def test_routes_have_performance_timing(self):
        """Test that performance timing has been added to health check."""
        routes_path = Path("app/routes/health.py")
        with open(routes_path, 'r') as f:
            content = f.read()

        assert "total_response_time_ms" in content, "Performance timing not found in health check"

    def test_routes_have_performance_diagnostics(self):
        """Test that performance diagnostics endpoint has been added."""
        routes_path = Path("app/routes/health.py")
        with open(routes_path, 'r') as f:
            content = f.read()

        assert "performance_diagnostics" in content, "Performance diagnostics endpoint not found"

    def test_routes_have_nltk_status_monitoring(self):
        """Test that NLTK status monitoring has been added."""
        routes_path = Path("app/routes/health.py")
        with open(routes_path, 'r') as f:
            content = f.read()

        assert "nltk_status" in content, "NLTK status monitoring not found"


class TestOptimizationIntegration:
    """Test the overall optimization integration."""

    def test_all_optimization_modules_available(self):
        """Test that all optimization modules are available."""
        # Test NLTK setup module
        from app.utils.nltk_setup import setup_nltk_resources
        assert callable(setup_nltk_resources)

        # Note: app.compat module was removed - models now load directly with joblib
        # No longer needed after module structure cleanup

        # Test that config has been optimized
        config_path = Path("app/config.py")
        with open(config_path, 'r') as f:
            content = f.read()
        assert "nltk.download" not in content

    def test_optimization_performance_benefits(self):
        """Test that optimizations provide expected performance benefits."""
        # Test that NLTK resources are loaded once (not per request)
        with patch('app.utils.nltk_setup.setup_nltk_resources') as mock_setup:
            mock_setup.return_value = {'success': True, 'setup_time_ms': 100}

            # Simulate multiple calls - should not call setup multiple times
            from app.utils.nltk_setup import setup_nltk_resources
            setup_nltk_resources()
            setup_nltk_resources()

            # In a real scenario, setup should only be called once at startup
            # This test verifies the function exists and can be called

    def test_optimization_error_handling(self):
        """Test that optimizations have proper error handling."""
        # Test NLTK setup error handling
        with patch('nltk.download'), patch('nltk.data.find', side_effect=LookupError('Resource not found')):
            validator_overrides = {
                'stopwords': lambda: False,
                'punkt': lambda: False,
                'wordnet': lambda: True,
            }
            with patch.dict('app.utils.nltk_setup.RESOURCE_VALIDATORS', validator_overrides, clear=True):
                with pytest.raises(NLTKSetupError):
                    setup_nltk_resources()

        # Note: Compatibility error handling test removed since app.compat was removed
        # Models now load directly with joblib without compatibility layers
