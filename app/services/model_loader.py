"""
Model loading utilities with download and diagnostic support.
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Optional

import joblib
import requests
from sklearn.exceptions import InconsistentVersionWarning

from .errors import ModelDownloadSkipped, ModelServiceError

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handle model loading, downloads, version checks, and diagnostics."""

    def __init__(self, model_path: Path, gdrive_model_id: Optional[str] = None) -> None:
        self.model_path = model_path
        self.gdrive_model_id = gdrive_model_id

    def load_model(self) -> Any:
        """Load the ML model, downloading if necessary."""
        try:
            should_download = (
                os.getenv("REPLIT_DB_URL") is not None or not self.model_path.exists()
            )

            if should_download:
                try:
                    self._download_model()
                except ModelDownloadSkipped:
                    logger.debug("Model download skipped, using existing local model")
                except RuntimeError as error:
                    if self.model_path.exists():
                        logger.warning("Download failed but using existing local model: %s", error)
                    else:
                        raise

            return self._load_model_with_version_check(self.model_path)

        except (FileNotFoundError, OSError) as error:
            logger.error("Model file not found or inaccessible: %s", error)
            raise ModelServiceError("Model file not found.") from error
        except (joblib.externals.loky.process_executor.TerminatedWorkerError, pickle.PickleError) as error:
            logger.error("Model file corrupted or incompatible: %s", error)
            raise ModelServiceError("Model file is corrupted.") from error
        except ModelServiceError:
            raise
        except Exception as error:
            logger.exception("Unexpected error loading model from %s", self.model_path)
            raise ModelServiceError("Failed to load model.") from error

    def load_local_model(self) -> Any:
        """Load a local model without attempting downloads."""
        return self._load_model_with_version_check(self.model_path)

    def log_model_diagnostics(self, model: Any) -> None:
        """Log model metadata and fitted-vectorizer checks for debugging."""
        try:
            model_size_bytes = self.model_path.stat().st_size
        except OSError:
            model_size_bytes = None

        model_sha256 = self._compute_model_hash()
        tfidf_fitted = self._inspect_tfidf_fitted(model)

        logger.info(
            "Model diagnostics: path=%s size_bytes=%s sha256=%s tfidf_fitted=%s",
            self.model_path,
            model_size_bytes,
            model_sha256,
            tfidf_fitted,
        )

    def _compute_model_hash(self) -> Optional[str]:
        try:
            with open(self.model_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except OSError as error:
            logger.warning("Failed to read model file for hash: %s", error)
            return None

    def _inspect_tfidf_fitted(self, model: Any) -> Optional[bool]:
        try:
            tfidf = None
            if hasattr(model, "named_steps"):
                tfidf = model.named_steps.get("tfidf")
            if tfidf is not None:
                return hasattr(tfidf, "idf_")
        except Exception as error:
            logger.warning("Failed to inspect tfidf fitted state: %s", error)
        return None

    def _download_model(self) -> None:
        """Download model from Google Drive if not available locally."""
        self._validate_gdrive_config()
        logger.info("Model not found locally, downloading from Google Drive...")

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = f"{self.model_path}.tmp"

        try:
            self._perform_download(temp_path)
            self._validate_downloaded_file(temp_path)
            self._finalize_download(temp_path)
            logger.info("Model downloaded and validated successfully!")

        except requests.exceptions.RequestException as error:
            self._cleanup_temp_file(temp_path)
            logger.error("Network error downloading model: %s", error)
            raise ModelServiceError("Network error downloading model.") from error
        except Exception as error:
            self._cleanup_temp_file(temp_path)
            self._handle_download_error(error)

    def _validate_gdrive_config(self) -> None:
        if not self.gdrive_model_id or self.gdrive_model_id.strip() in {
            "",
            "YOUR_FILE_ID",
            "YOUR_GOOGLE_DRIVE_FILE_ID",
        }:
            if self.model_path.exists():
                logger.info("Model found at %s, skipping download", self.model_path)
                raise ModelDownloadSkipped("Local model exists, skipping download")
            raise ModelServiceError("GDRIVE_MODEL_ID is not configured for model downloads.")

    def _perform_download(self, temp_path: str) -> None:
        url = f"https://drive.google.com/uc?export=download&id={self.gdrive_model_id}"
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            self._validate_response_content_type(response)
            self._write_download_to_file(response, temp_path)

    def _validate_response_content_type(self, response: requests.Response) -> None:
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type.lower():
            raise ModelServiceError("Google Drive returned HTML instead of the model file.")

    def _write_download_to_file(self, response: requests.Response, temp_path: str) -> None:
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    def _validate_downloaded_file(self, temp_path: str) -> None:
        if os.path.getsize(temp_path) < 1000:
            raise ModelServiceError("Downloaded model file is too small.")

        try:
            test_model = self._load_model_with_version_check(Path(temp_path))
            del test_model
        except ModelServiceError as error:
            raise ModelServiceError(
                "Downloaded model was trained with an incompatible scikit-learn version."
            ) from error
        except Exception as error:
            raise ModelServiceError("Downloaded model file is corrupted.") from error

    def _finalize_download(self, temp_path: str) -> None:
        os.replace(temp_path, self.model_path)

    def _cleanup_temp_file(self, temp_path: str) -> None:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    def _handle_download_error(self, error: Exception) -> None:
        error_str = str(error).lower()
        if "timeout" in error_str:
            raise ModelServiceError("Download timed out.") from error
        if "corrupted" in error_str:
            raise ModelServiceError("Downloaded model file is corrupted.") from error
        raise ModelServiceError("Failed to download model from Google Drive.") from error

    def _load_model_with_version_check(self, file_path: Path) -> Any:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", InconsistentVersionWarning)
            model = joblib.load(file_path)

        for warning in caught:
            if issubclass(warning.category, InconsistentVersionWarning):
                logger.error("scikit-learn version mismatch detected: %s", warning.message)
                raise ModelServiceError(
                    "Model was trained with a different scikit-learn version. "
                    "Retrain the model or install a matching scikit-learn version."
                )

        return model
