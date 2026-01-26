"""
Threshold management for model predictions.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional

from disasterproject.utils.config import CRITICAL_LABELS
from disasterproject.utils.model_defaults import (
    CRITICAL_CATEGORY_THRESHOLDS,
    DEFAULT_THRESHOLD,
)

logger = logging.getLogger(__name__)


class ThresholdManager:
    """Handle threshold defaults, overrides, and application."""

    def __init__(
        self,
        critical_labels: Optional[Iterable[str]] = None,
        default_threshold: float = DEFAULT_THRESHOLD,
        critical_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        self._critical_labels = set(critical_labels) if critical_labels else set(CRITICAL_LABELS)
        self._default_threshold = default_threshold
        self._critical_thresholds = (
            dict(critical_thresholds) if critical_thresholds else dict(CRITICAL_CATEGORY_THRESHOLDS)
        )

    def get_thresholds_map(
        self,
        label_order: Iterable[str],
        loaded_thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Build a thresholds map with defaults merged by loaded overrides.

        Args:
            label_order: Ordered iterable of category names.
            loaded_thresholds: Threshold overrides loaded from artifacts.

        Returns:
            Threshold map keyed by category name.
        """
        default_map: Dict[str, float] = {}
        for name in label_order:
            if name in self._critical_labels and name in self._critical_thresholds:
                default_map[name] = self._critical_thresholds[name]
            else:
                default_map[name] = self._default_threshold

        if loaded_thresholds:
            return {**default_map, **loaded_thresholds}
        return default_map
