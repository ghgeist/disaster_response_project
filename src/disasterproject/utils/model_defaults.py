"""
Default threshold configuration for model predictions.

Source: critical-category optimization session 2025-11-04 (historical defaults).
"""
from __future__ import annotations

from typing import Dict

DEFAULT_THRESHOLD: float = 0.5

# Critical categories need lower thresholds to improve recall.
CRITICAL_CATEGORY_THRESHOLDS: Dict[str, float] = {
    "hospitals": 0.014,  # 1.4% - extremely low for life-safety
    "security": 0.020,  # 2.0% - very low for emergencies
    "search_and_rescue": 0.033,  # 3.3% - low for urgent needs
    "medical_products": 0.095,  # 9.5% - low for medical supplies
    "medical_help": 0.124,  # 12.4% - low for medical emergencies
    "shelter": 0.240,  # 24.0% - moderate for shelter needs
    "water": 0.362,  # 36.2% - moderate for water needs
    "food": 0.431,  # 43.1% - moderate-high for food needs
}
