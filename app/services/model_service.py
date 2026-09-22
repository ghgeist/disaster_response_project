"""
Facade service for model loading and prediction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .artifact_loader import ModelArtifactLoader
from .category_mapper import CategoryMapper
from .errors import ModelServiceError
from .model_loader import ModelLoader
from .model_predictor import ModelPredictor
from .production_artifacts import ProductionArtifactError, ProductionArtifacts
from .threshold_manager import ThresholdManager

logger = logging.getLogger(__name__)


class ModelService:
    """Service for managing ML model loading and prediction."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self._model = None
        self._thresholds: Optional[Dict[str, float]] = None
        self._label_order: Optional[List[str]] = None
        self._artifacts: Optional[ProductionArtifacts] = None
        self._artifact_loader = ModelArtifactLoader(model_path)
        self._threshold_manager = ThresholdManager()
        self._category_mapper = CategoryMapper()
        self._predictor = ModelPredictor(self._category_mapper, self._threshold_manager)
        self._loader = ModelLoader(model_path)

    def load_model(self) -> Any:
        """Load the ML model and fail-closed production artifacts."""
        if self._model is not None:
            return self._model

        try:
            self._load_artifacts()
            self._model = self._loader.load_model()
        except ProductionArtifactError as error:
            self._model = None
            self._thresholds = None
            self._label_order = None
            self._artifacts = None
            logger.error("Production model provenance unavailable: %s", error)
            raise ModelServiceError(
                "Model unavailable: production artifact provenance failed."
            ) from error

        self._loader.log_model_diagnostics(self._model)
        logger.info("Model loaded successfully from %s", self.model_path)
        return self._model

    def predict(self, text: str) -> dict:
        """Make a prediction using the deployed thresholds operating point."""
        if self._model is None:
            self.load_model()

        try:
            return self._predictor.predict(
                self._model,
                text,
                self._label_order,
                self._thresholds,
                allow_predict_fallback=False,
            )
        except (ValueError, TypeError, AttributeError) as error:
            logger.error("Model prediction unavailable: %s", error)
            raise ModelServiceError(
                "Model unavailable: probability/threshold inference failed."
            ) from error
        except (OSError, FileNotFoundError) as error:
            logger.error("Model file access error during prediction: %s", error)
            raise ModelServiceError("Model file access failed.") from error
        except ModelServiceError:
            raise
        except Exception as error:
            logger.exception("Unexpected error during prediction for model %s", self.model_path)
            raise ModelServiceError("Prediction failed.") from error

    def _load_artifacts(self) -> None:
        """Load and validate stem-bound thresholds/labels with hash checks."""
        artifacts = self._artifact_loader.resolve()
        self._artifacts = artifacts
        self._thresholds = artifacts.thresholds
        self._label_order = artifacts.label_order

    def get_production_artifacts(self) -> ProductionArtifacts:
        """
        Return validated production artifacts for shared consumers (e.g. dashboard).

        Ensures the model/artifacts are loaded first.
        """
        if self._artifacts is None:
            self.load_model()
        if self._artifacts is None:
            raise ModelServiceError("Model unavailable: production artifacts not loaded.")
        return self._artifacts

    def get_thresholds_map(self) -> Dict[str, float]:
        """
        Return the deployed production thresholds map.

        Fails closed when production thresholds were not validated/loaded.
        Does not substitute smart defaults or 0.5.
        """
        if self._thresholds is None:
            if self._model is None:
                self.load_model()
            if self._thresholds is None:
                raise ModelServiceError(
                    "Model unavailable: production thresholds are required."
                )
        return dict(self._thresholds)
