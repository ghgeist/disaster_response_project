"""Test script for Google Drive model deployment."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pytest
from unittest.mock import MagicMock, patch

from app.services import ModelService

pytestmark = pytest.mark.gdrive


@pytest.fixture
def temp_model_path(tmp_path: Path) -> Path:
    """Create a temporary model path for testing."""
    return tmp_path / "temp_test_model.pkl"


@pytest.fixture
def mock_gdrive_id() -> str:
    """Mock Google Drive file ID for testing."""
    return "test_file_id_12345"


@pytest.fixture
def mock_model_data() -> bytes:
    """Mock model data for testing."""
    return b"mock_model_data_for_testing"


def _configure_mock_response(mock_get: MagicMock, payload: Iterable[bytes], *, content_type: str = "application/octet-stream") -> None:
    mock_response = MagicMock()
    mock_response.headers = {"content-type": content_type}
    mock_response.iter_content.return_value = payload
    mock_response.raise_for_status.return_value = None
    mock_get.return_value.__enter__.return_value = mock_response


class TestGoogleDriveDeployment:
    """Test Google Drive model deployment functionality."""

    def test_gdrive_model_id_validation(self, temp_model_path: Path) -> None:
        """Test that GDRIVE_MODEL_ID validation works correctly."""
        with pytest.raises(RuntimeError, match="GDRIVE_MODEL_ID is not set or is using a placeholder"):
            ModelService(temp_model_path, "YOUR_FILE_ID")

        with pytest.raises(RuntimeError, match="GDRIVE_MODEL_ID is not set or is using a placeholder"):
            ModelService(temp_model_path, "")

        with pytest.raises(RuntimeError, match="GDRIVE_MODEL_ID is not set or is using a placeholder"):
            ModelService(temp_model_path, None)

    def test_gdrive_model_id_acceptance(self, temp_model_path: Path) -> None:
        """Test that valid GDRIVE_MODEL_ID is accepted."""
        service = ModelService(temp_model_path, "valid_file_id_12345")
        assert service.gdrive_model_id == "valid_file_id_12345"

    @patch("requests.get")
    def test_gdrive_download_success(
        self,
        mock_get: MagicMock,
        temp_model_path: Path,
        mock_gdrive_id: str,
        mock_model_data: bytes,
    ) -> None:
        """Test successful Google Drive download."""
        _configure_mock_response(mock_get, [mock_model_data])

        with patch("joblib.load") as mock_joblib_load:
            mock_model = MagicMock()
            mock_joblib_load.return_value = mock_model

            service = ModelService(temp_model_path, mock_gdrive_id)
            model = service.load_model()

            assert model is not None
            assert temp_model_path.exists(), "Model file should be persisted after download"
            mock_get.assert_called_once()
            assert not list(temp_model_path.parent.glob("*.tmp")), "Temporary files were not cleaned up"

    @patch("requests.get")
    def test_gdrive_download_html_response_error(
        self, mock_get: MagicMock, temp_model_path: Path, mock_gdrive_id: str
    ) -> None:
        """Test error handling when Google Drive returns HTML instead of file."""
        _configure_mock_response(mock_get, [b"<html>Authentication required</html>"], content_type="text/html")

        service = ModelService(temp_model_path, mock_gdrive_id)

        with pytest.raises(RuntimeError, match="Google Drive returned HTML instead of the model file"):
            service.load_model()

    @patch("requests.get")
    def test_gdrive_download_network_error(self, mock_get: MagicMock, temp_model_path: Path, mock_gdrive_id: str) -> None:
        """Test error handling for network errors during download."""
        mock_get.side_effect = ConnectionError("Network connection failed")

        service = ModelService(temp_model_path, mock_gdrive_id)

        with pytest.raises(RuntimeError, match="Network error downloading model"):
            service.load_model()

    @patch("requests.get")
    def test_gdrive_download_timeout_error(self, mock_get: MagicMock, temp_model_path: Path, mock_gdrive_id: str) -> None:
        """Test error handling for timeout errors during download."""
        mock_get.side_effect = TimeoutError("Request timed out")

        service = ModelService(temp_model_path, mock_gdrive_id)

        with pytest.raises(RuntimeError, match="Download timed out"):
            service.load_model()

    @patch("requests.get")
    def test_gdrive_download_corrupted_file(self, mock_get: MagicMock, temp_model_path: Path, mock_gdrive_id: str) -> None:
        """Test error handling for corrupted downloaded file."""
        _configure_mock_response(mock_get, [b"corrupted_data"])

        with patch("joblib.load") as mock_joblib_load:
            mock_joblib_load.side_effect = Exception("Corrupted file")

            service = ModelService(temp_model_path, mock_gdrive_id)

            with pytest.raises(RuntimeError, match="Downloaded model file is corrupted"):
                service.load_model()

    @patch("requests.get")
    def test_gdrive_download_file_too_small(self, mock_get: MagicMock, temp_model_path: Path, mock_gdrive_id: str) -> None:
        """Test error handling for file that's too small."""
        _configure_mock_response(mock_get, [b"x"])

        service = ModelService(temp_model_path, mock_gdrive_id)

        with pytest.raises(RuntimeError, match="Downloaded file is too small"):
            service.load_model()

    def test_gdrive_download_url_construction(self, temp_model_path: Path, mock_gdrive_id: str) -> None:
        """Test that Google Drive download URL is constructed correctly."""
        with patch("requests.get") as mock_get:
            _configure_mock_response(mock_get, [b"test_data"])

            with patch("joblib.load") as mock_joblib_load:
                mock_joblib_load.return_value = MagicMock()

                service = ModelService(temp_model_path, mock_gdrive_id)
                service.load_model()

                expected_url = f"https://drive.google.com/uc?export=download&id={mock_gdrive_id}"
                mock_get.assert_called_once()
                call_args = mock_get.call_args
                assert call_args[0][0] == expected_url

    @patch("requests.get")
    def test_gdrive_prediction_after_download(
        self, mock_get: MagicMock, temp_model_path: Path, mock_gdrive_id: str
    ) -> None:
        """Test that predictions work after successful Google Drive download."""
        _configure_mock_response(mock_get, [b"mock_model_data"])

        mock_model = MagicMock()
        mock_model.predict.return_value = [[1, 0, 1, 0]]
        mock_model.predict_proba.return_value = [[[0.2, 0.8], [0.9, 0.1], [0.3, 0.7], [0.6, 0.4]]]

        with patch("joblib.load", return_value=mock_model):
            service = ModelService(temp_model_path, mock_gdrive_id)

            for msg in [
                "We need urgent medical supplies for earthquake victims",
                "Food and water running low in shelter area",
                "Search and rescue teams needed immediately",
            ]:
                result = service.predict(msg)

                assert isinstance(result, dict)
                assert "labels" in result
                assert "probabilities" in result

                labels = result["labels"]
                assert isinstance(labels, dict)
                assert len(labels) > 0

    def test_gdrive_environment_variable_handling(self, temp_model_path: Path) -> None:
        """Test that GDRIVE_MODEL_ID environment variable is handled correctly."""
        with patch.dict(os.environ, {"GDRIVE_MODEL_ID": "env_file_id_12345"}):
            service = ModelService(temp_model_path, "env_file_id_12345")
            assert service.gdrive_model_id == "env_file_id_12345"

    @patch("requests.get")
    def test_gdrive_cleanup_on_error(self, mock_get: MagicMock, temp_model_path: Path, mock_gdrive_id: str) -> None:
        """Test that temporary files are cleaned up on error."""
        mock_get.side_effect = ConnectionError("Network error")

        service = ModelService(temp_model_path, mock_gdrive_id)

        with pytest.raises(RuntimeError):
            service.load_model()

        temp_files = list(temp_model_path.parent.glob("*.tmp"))
        assert len(temp_files) == 0

    def test_gdrive_model_service_initialization(self, temp_model_path: Path, mock_gdrive_id: str) -> None:
        """Test ModelService initialization with Google Drive configuration."""
        service = ModelService(temp_model_path, mock_gdrive_id)

        assert service.model_path == temp_model_path
        assert service.gdrive_model_id == mock_gdrive_id
        assert service._model is None
        assert service._thresholds is None
        assert service._label_order is None

    @pytest.mark.skipif(
        not os.environ.get("GDRIVE_MODEL_ID")
        or os.environ.get("GDRIVE_MODEL_ID") in {"", "YOUR_FILE_ID", "YOUR_GOOGLE_DRIVE_FILE_ID"},
        reason="GDRIVE_MODEL_ID not set or is placeholder",
    )
    def test_gdrive_integration_with_real_id(self, temp_model_path: Path) -> None:
        """Test Google Drive integration with real file ID (requires valid GDRIVE_MODEL_ID)."""
        gdrive_id = os.environ.get("GDRIVE_MODEL_ID")
        service = ModelService(temp_model_path, gdrive_id)

        model = service.load_model()
        assert model is not None

        result = service.predict("Test message for real Google Drive model")
        assert isinstance(result, dict)
        assert "labels" in result

        if temp_model_path.exists():
            temp_model_path.unlink()
