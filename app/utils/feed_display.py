"""
Display helpers for Storm Signal feed-shaped payloads.

Shared by the dashboard API routes and the offline demo-feed builder so the
script never imports Flask routes.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

CATEGORY_DISPLAY_NAMES = {
    "search_and_rescue": "Search & Rescue",
    "infrastructure_related": "Infrastructure",
    "aid_centers": "Aid Centers",
    "other_infrastructure": "Other Infrastructure",
    "weather_related": "Weather Related",
    "direct_report": "Direct Report",
    "child_alone": "Child Alone",
    "medical_products": "Medical Products",
    "other_aid": "Other Aid",
    "other_weather": "Other Weather",
}

CRITICAL_INTERNAL_CATEGORIES = {
    "medical_help",
    "medical_products",
    "search_and_rescue",
    "water",
    "food",
    "shelter",
    "security",
    "hospitals",
    "missing_people",
    "refugees",
    "death",
}

GENRE_TO_SOURCE = {
    "direct": "Direct Report",
    "news": "News",
    "social": "Social",
}
DEFAULT_SOURCE = "X"


def to_display_name(internal: str) -> str:
    """Convert internal category names to display names."""
    return CATEGORY_DISPLAY_NAMES.get(internal, internal.replace("_", " ").title())


def _safe_float_prob(value: Any) -> float:
    """Return probability as float; treat NaN/None/inf/non-numeric as 0.0."""
    if value is None:
        return 0.0
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return 0.0
    try:
        result = float(value)
        return 0.0 if math.isnan(result) or math.isinf(result) else result
    except (TypeError, ValueError):
        return 0.0


def _safe_text_value(value: Any) -> str:
    """Return safe string value, treating NaN/None as empty."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        if math.isnan(value):
            return ""
    except TypeError:
        pass
    return str(value)


def calculate_severity(probabilities: Mapping[str, Any]) -> str:
    """Determine severity based on critical category probabilities.

    Display heuristic only (fixed 0.5 / 0.70 / 0.85 rules). Not a calibrated
    production operating-point decision.
    """
    critical_count = sum(
        1
        for category, probability in probabilities.items()
        if category in CRITICAL_INTERNAL_CATEGORIES and _safe_float_prob(probability) > 0.5
    )
    critical_probabilities = [
        _safe_float_prob(probability)
        for category, probability in probabilities.items()
        if category in CRITICAL_INTERNAL_CATEGORIES
    ]
    max_confidence = max(critical_probabilities) if critical_probabilities else 0.0
    if critical_count >= 2 or max_confidence > 0.85:
        return "HIGH"
    if critical_count >= 1 or max_confidence > 0.70:
        return "MEDIUM"
    return "LOW"


def genre_to_source(genre: str) -> str:
    """Map database genre to display source. Unknown genres map to X."""
    normalized = _safe_text_value(genre).strip().lower() or "direct"
    return GENRE_TO_SOURCE.get(normalized, DEFAULT_SOURCE)
