"""
Category mapping utilities for model predictions.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

DEFAULT_LABEL_ORDER = [
    "related",
    "request",
    "offer",
    "aid_related",
    "medical_help",
    "medical_products",
    "search_and_rescue",
    "security",
    "military",
    "child_alone",
    "water",
    "food",
    "shelter",
    "clothing",
    "money",
    "missing_people",
    "refugees",
    "death",
    "other_aid",
    "infrastructure_related",
    "transport",
    "buildings",
    "electricity",
    "tools",
    "hospitals",
    "shops",
    "aid_centers",
    "other_infrastructure",
    "weather_related",
    "floods",
    "storm",
    "fire",
    "earthquake",
    "cold",
    "other_weather",
    "direct_report",
]


class CategoryMapper:
    """Handle category ordering, padding, and mapping."""

    def get_label_order(self, label_order: List[str] | None) -> List[str]:
        """Return label order from artifact if present, else fallback to defaults."""
        if isinstance(label_order, list) and label_order:
            return label_order
        return list(DEFAULT_LABEL_ORDER)

    def create_category_mapping(
        self, expected_categories: List[str], model_output_count: int
    ) -> Tuple[List[str], Dict[int, int]]:
        """
        Create a mapping between model outputs and expected categories.
        """
        if model_output_count >= len(expected_categories):
            active_categories = expected_categories[:model_output_count]
            category_mapping = {i: i for i in range(model_output_count)}
            logger.info(
                "Model has %d outputs, using first %d expected categories",
                model_output_count,
                len(active_categories),
            )
        else:
            active_categories = expected_categories[:model_output_count]
            category_mapping = {i: i for i in range(model_output_count)}
            logger.warning(
                "Model has %d outputs but %d expected categories. Using first %d expected categories."
                " Consider updating the model or expected categories to match.",
                model_output_count,
                len(expected_categories),
                model_output_count,
            )

        return active_categories, category_mapping

    def map_probabilities_to_expected(
        self,
        expected_categories: List[str],
        model_probs: List[float],
        category_mapping: Dict[int, int],
    ) -> Dict[str, float]:
        """
        Map model probabilities to all expected categories.
        """
        final_probs: Dict[str, float] = {}

        for expected_idx, category_name in enumerate(expected_categories):
            model_idx = None
            for model_output_idx, mapped_expected_idx in category_mapping.items():
                if mapped_expected_idx == expected_idx and model_output_idx < len(model_probs):
                    model_idx = model_output_idx
                    break

            if model_idx is not None:
                prob_val = model_probs[model_idx]
            else:
                prob_val = 0.0
                logger.debug("Category %s not in model output, setting to 0", category_name)

            final_probs[category_name] = prob_val

        return final_probs

    def pad_or_truncate_labels(
        self, expected_categories: List[str], raw_predictions: List[int]
    ) -> List[int]:
        """
        Pad or truncate labels to match expected category count.
        """
        model_output_count = len(raw_predictions)
        expected_count = len(expected_categories)

        if model_output_count != expected_count:
            logger.warning(
                "Predict fallback: Model output count (%d) != expected count (%d)",
                model_output_count,
                expected_count,
            )

        if model_output_count < expected_count:
            logger.info("Padded %d missing categories with 0", expected_count - model_output_count)
            return list(raw_predictions) + [0] * (expected_count - model_output_count)

        if model_output_count > expected_count:
            logger.info("Truncated %d extra categories", model_output_count - expected_count)
            return list(raw_predictions[:expected_count])

        return list(raw_predictions)
