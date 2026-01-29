"""
Model loading utilities with version checks and diagnostics.
"""
from __future__ import annotations

import hashlib
import logging
import pickle
import warnings
from pathlib import Path
from typing import Any, Optional

import joblib
from sklearn.exceptions import InconsistentVersionWarning

from .errors import ModelServiceError

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handle model loading, version checks, and diagnostics."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path

    def load_model(self) -> Any:
        """Load the ML model from local storage."""
        try:
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
