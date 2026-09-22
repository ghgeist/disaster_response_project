"""
Threshold management for model predictions.
"""
from __future__ import annotations

import logging
import math
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

    def require_loaded_thresholds(
        self,
        label_order: Iterable[str],
        loaded_thresholds: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Return a complete thresholds map without smart defaults.

        Production inference must supply every label with a finite ``[0, 1]`` value.
        """
        labels = list(label_order)
        if not isinstance(loaded_thresholds, dict) or not loaded_thresholds:
            raise ValueError("Production thresholds map is required")

        missing = [name for name in labels if name not in loaded_thresholds]
        if missing:
            preview = ", ".join(missing[:8])
            suffix = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
            raise ValueError(f"Thresholds map incomplete; missing: {preview}{suffix}")

        validated: Dict[str, float] = {}
        for name in labels:
            raw_value = loaded_thresholds[name]
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Threshold for {name!r} is not numeric: {raw_value!r}") from exc
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(
                    f"Threshold for {name!r} out of range: {raw_value!r} (need [0, 1])"
                )
            validated[name] = value
        return validated

    def get_thresholds_map(
        self,
        label_order: Iterable[str],
        loaded_thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Build a thresholds map with defaults merged by loaded overrides.

        Prefer :meth:`require_loaded_thresholds` for production inference.
        """
        default_map: Dict[str, float] = {}
        for name in label_order:
            if name in self._critical_labels and name in self._critical_thresholds:
                default_map[name] = self._critical_thresholds[name]
            else:
                default_map[name] = self._default_threshold

        if loaded_thresholds:
            if not isinstance(loaded_thresholds, dict):
                logger.warning(
                    "Loaded thresholds is not a dict (type: %s), using defaults only",
                    type(loaded_thresholds).__name__,
                )
                return default_map
            return {**default_map, **loaded_thresholds}
        return default_map