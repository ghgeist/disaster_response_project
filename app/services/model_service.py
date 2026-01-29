"""
Facade service for model loading and prediction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from .artifact_loader import ModelArtifactLoader
from .category_mapper import CategoryMapper
from .errors import ModelServiceError
from .model_loader import ModelLoader
from .model_predictor import ModelPredictor
from .threshold_manager import ThresholdManager

logger = logging.getLogger(__name__)


class ModelService:
    """Service for managing ML model loading and prediction."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self._model = None
        self._thresholds = None
        self._label_order = None
        self._artifact_loader = ModelArtifactLoader(model_path)
        self._threshold_manager = ThresholdManager()
        self._category_mapper = CategoryMapper()
        self._predictor = ModelPredictor(self._category_mapper, self._threshold_manager)
        self._loader = ModelLoader(model_path)

    def load_model(self) -> Any:
        """Load the ML model from local storage."""
        if self._model is not None:
            return self._model

        self._model = self._loader.load_model()
        self._load_artifacts()
        self._loader.log_model_diagnostics(self._model)
        logger.info("Model loaded successfully from %s", self.model_path)
        return self._model

    def predict(self, text: str) -> dict:
        """Make a prediction on the given text using per-label thresholds when available."""
        if self._model is None:
            self.load_model()

        try:
            return self._predictor.predict(self._model, text, self._label_order, self._thresholds)
        except (ValueError, AttributeError) as error:
            logger.error("Model prediction input error: %s", error)
            raise ModelServiceError("Invalid input for prediction.") from error
        except (OSError, FileNotFoundError) as error:
            logger.error("Model file access error during prediction: %s", error)
            raise ModelServiceError("Model file access failed.") from error
        except Exception as error:
            logger.exception("Unexpected error during prediction for model %s", self.model_path)
            raise ModelServiceError("Prediction failed.") from error

    def _load_artifacts(self) -> None:
        """Load thresholds and label_order artifacts from the model directory if present."""
        thresholds, label_order = self._artifact_loader.load_artifacts()
        self._thresholds = thresholds
        self._label_order = label_order

    def _get_label_order(self) -> list:
        """Return label order from artifact if present, else fallback to defaults."""
        return self._category_mapper.get_label_order(self._label_order)

    def get_thresholds_map(self) -> Dict[str, float]:
        """
        Return thresholds map; if missing, return smart defaults based on category type.
        """
        label_order = self._get_label_order()
        return self._threshold_manager.get_thresholds_map(label_order, self._thresholds)
