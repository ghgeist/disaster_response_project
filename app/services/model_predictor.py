"""
Prediction orchestration for loaded models.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from .category_mapper import CategoryMapper
from .threshold_manager import ThresholdManager

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Handle prediction logic and probability processing."""

    def __init__(self, category_mapper: CategoryMapper, threshold_manager: ThresholdManager) -> None:
        self._category_mapper = category_mapper
        self._threshold_manager = threshold_manager

    def predict(
        self,
        model: Any,
        text: str,
        label_order: List[str] | None,
        thresholds: Dict[str, float] | None,
    ) -> dict:
        """Predict labels and probabilities for an input message."""
        category_names = self._category_mapper.get_label_order(label_order)

        try:
            return self._predict_with_probabilities(model, text, category_names, thresholds)
        except Exception as prob_exc:
            logger.warning(
                "Probability path failed (%s); falling back to default predict",
                prob_exc,
            )
            return self._predict_fallback(model, text, category_names)

    def _predict_with_probabilities(
        self,
        model: Any,
        text: str,
        category_names: List[str],
        thresholds: Dict[str, float] | None,
    ) -> dict:
        """Use predict_proba and per-label thresholds when available."""
        proba = model.predict_proba([text])
        if not isinstance(proba, list):
            raise TypeError("Unexpected predict_proba output; using predict fallback")

        model_probs, category_mapping = self._extract_probabilities(model, proba, category_names)
        normalized_probs = self._normalize_outputs(
            category_names,
            model_probs=model_probs,
            category_mapping=category_mapping,
        )

        thresholds_map = self._threshold_manager.get_thresholds_map(category_names, thresholds)
        labels = self._apply_thresholds(category_names, normalized_probs, thresholds_map)
        return self._build_prediction(category_names, labels, normalized_probs)

    def _predict_fallback(self, model: Any, text: str, category_names: List[str]) -> dict:
        """Fallback to simple predict when probabilities are unavailable."""
        raw_predictions = model.predict([text])[0]
        normalized_labels = self._normalize_outputs(
            category_names,
            raw_predictions=list(raw_predictions),
        )
        return self._build_prediction(category_names, normalized_labels, {})

    def _extract_probabilities(
        self,
        model: Any,
        proba: List[Any],
        category_names: List[str],
    ) -> Tuple[List[float], Dict[int, int]]:
        model_output_count = len(proba)
        expected_count = len(category_names)

        if model_output_count != expected_count:
            logger.warning(
                "Model output count (%d) != expected count (%d). "
                "Model may have been trained on a subset of categories.",
                model_output_count,
                expected_count,
            )
            active_categories, category_mapping = self._category_mapper.create_category_mapping(
                category_names,
                model_output_count,
            )
        else:
            active_categories = category_names
            category_mapping = {i: i for i in range(len(category_names))}

        multi_output_clf = self._extract_multi_output_classifier(model)
        probs = [
            self._resolve_probability(
                idx,
                p,
                multi_output_clf,
                active_categories,
            )
            for idx, p in enumerate(proba)
        ]
        return probs, category_mapping

    def _extract_multi_output_classifier(self, model: Any) -> Any | None:
        if hasattr(model, "named_steps") and "clf" in model.named_steps:
            return model.named_steps["clf"]
        return None

    def _resolve_probability(
        self,
        idx: int,
        probs_array: Any,
        multi_output_clf: Any | None,
        active_categories: List[str],
    ) -> float:
        if probs_array.shape[1] == 1:
            prob_val = self._resolve_degenerate_probability(idx, multi_output_clf)
            category_name = self._safe_category_name(active_categories, idx)
            logger.debug(
                "Label %d (%s): degenerate classifier detected, positive prob set to %.1f",
                idx,
                category_name,
                prob_val,
            )
            return prob_val

        if probs_array.shape[1] == 2:
            prob_val = probs_array[:, 1][0]
            category_name = self._safe_category_name(active_categories, idx)
            logger.debug(
                "Label %d (%s): two columns prob=%.4f (class 1)",
                idx,
                category_name,
                prob_val,
            )
            return prob_val

        logger.warning(
            "Unexpected predict_proba shape %s for label %d, falling back to predict",
            probs_array.shape,
            idx,
        )
        raise TypeError(f"Unexpected predict_proba shape {probs_array.shape}")

    def _resolve_degenerate_probability(self, idx: int, multi_output_clf: Any | None) -> float:
        if multi_output_clf is not None and hasattr(multi_output_clf, "classes_"):
            if idx < len(multi_output_clf.classes_):
                classes = multi_output_clf.classes_[idx]
                if len(classes) == 1 and classes[0] == 0:
                    return 0.0
                if len(classes) == 1 and classes[0] == 1:
                    return 1.0
        return 0.0

    def _safe_category_name(self, active_categories: List[str], idx: int) -> str:
        if idx < len(active_categories):
            return active_categories[idx]
        return f"unknown_{idx}"

    def _apply_thresholds(
        self,
        category_names: List[str],
        probabilities: Dict[str, float],
        thresholds_map: Dict[str, float],
    ) -> List[int]:
        return [
            1 if probabilities.get(category_name, 0.0) >= thresholds_map.get(category_name, 0.5) else 0
            for category_name in category_names
        ]

    def _normalize_outputs(
        self,
        category_names: List[str],
        *,
        model_probs: List[float] | None = None,
        category_mapping: Dict[int, int] | None = None,
        raw_predictions: List[int] | None = None,
    ) -> Dict[str, float] | List[int]:
        if model_probs is not None:
            if category_mapping is None:
                raise ValueError("category_mapping required when normalizing probabilities")
            return self._category_mapper.map_probabilities_to_expected(
                category_names,
                model_probs,
                category_mapping,
            )

        if raw_predictions is not None:
            return self._category_mapper.pad_or_truncate_labels(
                category_names,
                raw_predictions,
            )

        raise ValueError("Either model_probs or raw_predictions is required")

    def _build_prediction(
        self,
        category_names: List[str],
        labels: List[int],
        probabilities: Dict[str, float],
    ) -> dict:
        results = dict(zip(category_names, labels))
        prob_dict = probabilities or {}
        return {"labels": results, "probabilities": prob_dict}
