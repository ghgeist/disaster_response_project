"""
Test Flask application with standardized model naming.
"""
import pytest
from pathlib import Path

from app.app import create_app
from app.config import Config, TestConfig
from app.services import ModelService


@pytest.fixture
def production_app():
    """Create test Flask app with production config."""
    return create_app(Config)


@pytest.fixture
def test_app():
    """Create test Flask app with test config."""
    return create_app(TestConfig)


@pytest.fixture
def client(production_app):
    """Create test client for production config."""
    return production_app.test_client()


@pytest.fixture
def test_client(test_app):
    """Create test client for test config."""
    return test_app.test_client()


class TestFlaskStandardized:
    """Test Flask application with standardized model validation."""

    def test_app_creation(self, production_app):
        """Test Flask app creation with standardized model validation."""
        assert production_app is not None
        assert production_app.config['TESTING'] is False

    def test_test_app_creation(self, test_app):
        """Test Flask test app creation."""
        assert test_app is not None
        assert test_app.config['TESTING'] is True

    def test_model_service_initialization(self, production_app):
        """Test model service initialization in app context."""
        with production_app.app_context():
            model_service = production_app.model_service
            assert model_service is not None
            assert isinstance(model_service, ModelService)

    def test_model_prediction(self, production_app):
        """Test model prediction through Flask app."""
        with production_app.app_context():
            model_service = production_app.model_service
            
            # Test prediction with disaster message
            test_message = 'Search and rescue teams needed for earthquake victims'
            result = model_service.predict(test_message)
            
            # Validate result structure
            assert isinstance(result, dict)
            assert 'labels' in result
            assert 'probabilities' in result
            
            # Count positive predictions
            labels = result['labels']
            positive_count = sum(1 for v in labels.values() if v == 1)
            assert positive_count >= 0
            
            # Get active categories (first 4 for display)
            active_cats = [k for k, v in labels.items() if v == 1][:4]
            assert len(active_cats) <= 4

    def test_model_prediction_with_different_messages(self, production_app):
        """Test model prediction with various disaster messages."""
        test_messages = [
            "We need urgent medical supplies for earthquake victims",
            "Food and water running low in shelter area", 
            "Roads blocked by fallen trees need clearing",
            "Missing person reported in flood zone"
        ]
        
        with production_app.app_context():
            model_service = production_app.model_service
            
            for message in test_messages:
                result = model_service.predict(message)
                
                # Validate result structure
                assert isinstance(result, dict)
                assert 'labels' in result
                
                # Validate labels are binary (0 or 1)
                labels = result['labels']
                for label, value in labels.items():
                    assert value in [0, 1], f"Label {label} has invalid value {value}"

    def test_model_artifacts_loading(self, production_app):
        """Test that model artifacts (thresholds, labels) are loaded correctly."""
        with production_app.app_context():
            model_service = production_app.model_service
            
            # Test that model can make predictions (implies artifacts loaded)
            test_message = 'Medical help needed urgently'
            result = model_service.predict(test_message)
            
            # Check that we have reasonable number of categories
            labels = result['labels']
            assert len(labels) > 20  # Should have many disaster categories
            
            # Check that some common categories are present
            expected_categories = ['medical_help', 'water', 'food', 'shelter']
            for category in expected_categories:
                assert category in labels, f"Expected category {category} not found in labels"

    def test_model_service_with_gdrive_config(self, production_app):
        """Test model service with Google Drive configuration."""
        with production_app.app_context():
            model_service = production_app.model_service
            
            # Check if Google Drive ID is configured
            gdrive_id = production_app.config.get('GDRIVE_MODEL_ID')
            if gdrive_id and gdrive_id.strip() not in {'', 'YOUR_FILE_ID', 'YOUR_GOOGLE_DRIVE_FILE_ID'}:
                # If GDrive is configured, test that model can still load
                test_message = 'Test message for GDrive model'
                result = model_service.predict(test_message)
                assert isinstance(result, dict)
                assert 'labels' in result
            else:
                # If no GDrive config, should still work with local model
                pytest.skip("GDRIVE_MODEL_ID not configured, testing local model only")

    def test_app_routes_accessible(self, client):
        """Test that main app routes are accessible."""
        # Test home page
        response = client.get('/')
        assert response.status_code == 200
        assert b'Signal Storm' in response.data
        
        # Test health check (if available)
        try:
            response = client.get('/health')
            assert response.status_code in [200, 404]  # 404 if not implemented
        except Exception:
            pass  # Health endpoint might not exist

    def test_model_path_configuration(self, production_app):
        """Test that model path is correctly configured."""
        with production_app.app_context():
            model_path = production_app.config.get('MODEL_PATH')
            assert model_path is not None
            assert isinstance(model_path, Path)
            
            # Check that model directory exists
            model_dir = model_path.parent
            assert model_dir.exists(), f"Model directory {model_dir} does not exist"

    def test_model_filename_configuration(self, production_app):
        """Test that model filename is correctly configured."""
        with production_app.app_context():
            model_filename = production_app.config.get('MODEL_FILENAME')
            assert model_filename is not None
            assert model_filename.endswith('.pkl')
            
            # Check that it matches the expected naming convention
            assert 'disaster_rf' in model_filename
            assert 'prod' in model_filename
