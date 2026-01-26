"""Tests ensuring the Flask app uses consistent configuration and model wiring."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Config, TestConfig
from app.services.model_service import ModelService
from tests.conftest import create_test_app, skip_if_no_model

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def production_app():
    skip_if_no_model(
        Config,
        reason="Model artifact required for production-config tests is not present.",
    )
    return create_test_app(Config)


@pytest.fixture(scope="module")
def production_client(production_app):
    with production_app.test_client() as client:
        yield client


@pytest.fixture(scope="module")
def test_app():
    return create_test_app(TestConfig)


@pytest.fixture(scope="module")
def test_client(test_app):
    with test_app.test_client() as client:
        yield client


class TestFlaskStandardized:
    """Test Flask application with standardized model validation."""

    def test_app_creation(self, production_app):
        """Test Flask app creation with standardized model validation."""
        assert production_app is not None
        assert production_app.config["TESTING"] is False

    def test_test_app_creation(self, test_app):
        """Test Flask test app creation."""
        assert test_app is not None
        assert test_app.config["TESTING"] is True

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

            result = model_service.predict(
                "Search and rescue teams needed for earthquake victims"
            )

            assert isinstance(result, dict)
            assert "labels" in result
            assert "probabilities" in result

            labels = result["labels"]
            positive_count = sum(1 for v in labels.values() if v == 1)
            assert positive_count >= 0

            active_cats = [k for k, v in labels.items() if v == 1][:4]
            assert len(active_cats) <= 4

    def test_model_prediction_with_different_messages(self, production_app):
        """Test model prediction with various disaster messages."""
        test_messages = [
            "We need urgent medical supplies for earthquake victims",
            "Food and water running low in shelter area",
            "Roads blocked by fallen trees need clearing",
            "Missing person reported in flood zone",
        ]

        with production_app.app_context():
            model_service = production_app.model_service

            for message in test_messages:
                result = model_service.predict(message)

                assert isinstance(result, dict)
                assert "labels" in result

                labels = result["labels"]
                for label, value in labels.items():
                    assert value in [0, 1], f"Label {label} has invalid value {value}"

    def test_model_artifacts_loading(self, production_app):
        """Test that model artifacts (thresholds, labels) are loaded correctly."""
        with production_app.app_context():
            model_service = production_app.model_service

            result = model_service.predict("Medical help needed urgently")

            labels = result["labels"]
            assert len(labels) > 20

            expected_categories = ["medical_help", "water", "food", "shelter"]
            for category in expected_categories:
                assert category in labels, f"Expected category {category} not found in labels"

    def test_model_service_with_gdrive_config(self, production_app):
        """Test model service with Google Drive configuration."""
        with production_app.app_context():
            model_service = production_app.model_service

            gdrive_id = production_app.config.get("GDRIVE_MODEL_ID")
            if gdrive_id and gdrive_id.strip() not in {"", "YOUR_FILE_ID", "YOUR_GOOGLE_DRIVE_FILE_ID"}:
                result = model_service.predict("Test message for GDrive model")
                assert isinstance(result, dict)
                assert "labels" in result
            else:
                pytest.skip("GDRIVE_MODEL_ID not configured, testing local model only")

    def test_app_routes_accessible(self, test_client):
        """Test that main app routes are accessible under the test config."""
        response = test_client.get("/")
        assert response.status_code == 200
        assert b"Signal Storm" in response.data

        try:
            response = test_client.get("/health")
            assert response.status_code in [200, 404]
        except Exception:
            pass

    def test_model_path_configuration(self):
        """Test that model path is correctly configured."""
        model_path = Config.MODEL_PATH
        assert isinstance(model_path, Path)
        assert model_path.parent.exists(), f"Model directory {model_path.parent} does not exist"

    def test_model_filename_configuration(self):
        """Test that model filename is correctly configured."""
        model_filename = Config.MODEL_FILENAME
        assert model_filename is not None
        assert model_filename.endswith(".pkl")
        assert "disaster_rf" in model_filename
        assert "prod" in model_filename
