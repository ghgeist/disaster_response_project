"""Regression/invariant tests for dashboard API (severity, probability bands, safe helpers).

Option A: Severity and simulated probability invariants — lock in business rules.
Option B: Safe helpers / NaN contract (Data Reality Gate) — no NaN/Infinity in JSON.
"""

import json
import math
import random
from unittest.mock import patch

import pytest

from app.routes.api import (
    CATEGORY_GROUPS,
    CRITICAL_INTERNAL_CATEGORIES,
    _safe_category_display,
    _safe_float_prob,
    _safe_label_value,
    _safe_text_value,
    _simulated_probabilities,
    calculate_severity,
    to_display_name,
)


# ---- Option A: Severity invariants ----


def test_calculate_severity_high_when_two_critical_above_half():
    """Two critical categories above 0.5 → HIGH."""
    probs = {
        "water": 0.6,
        "medical_help": 0.7,
        "other_aid": 0.9,
    }
    assert calculate_severity(probs) == "HIGH"


def test_calculate_severity_high_when_max_critical_above_85():
    """Single critical category with prob > 0.85 → HIGH."""
    probs = {
        "water": 0.9,
        "other_aid": 0.3,
    }
    assert calculate_severity(probs) == "HIGH"


def test_calculate_severity_medium_when_one_critical_above_half():
    """Exactly one critical above 0.5, max critical <= 0.85 → MEDIUM."""
    probs = {
        "water": 0.6,
        "medical_help": 0.3,
        "other_aid": 0.2,
    }
    assert calculate_severity(probs) == "MEDIUM"


def test_calculate_severity_medium_when_max_critical_between_70_and_85():
    """No critical above 0.5 but max critical in (0.70, 0.85] → MEDIUM."""
    probs = {
        "water": 0.75,
        "medical_help": 0.4,
    }
    assert calculate_severity(probs) == "MEDIUM"


def test_calculate_severity_low_when_no_critical_above_half():
    """No critical category above 0.5 and max critical <= 0.70 → LOW."""
    probs = {
        "water": 0.4,
        "medical_help": 0.3,
        "other_aid": 0.8,
    }
    assert calculate_severity(probs) == "LOW"


def test_calculate_severity_low_when_empty_probabilities():
    """Empty probabilities → LOW."""
    assert calculate_severity({}) == "LOW"


def test_calculate_severity_uses_only_critical_categories_for_max():
    """Severity uses max over critical categories only; non-critical high prob does not raise severity."""
    probs = {
        "water": 0.5,
        "medical_help": 0.5,
        "other_aid": 0.99,
    }
    assert calculate_severity(probs) == "LOW"


def test_calculate_severity_treats_new_critical_categories_as_critical():
    """Missing People, Refugees, and Death are treated as critical categories."""
    # Test missing_people
    probs_missing = {
        "missing_people": 0.6,
        "water": 0.7,
    }
    assert calculate_severity(probs_missing) == "HIGH"
    
    # Test refugees
    probs_refugees = {
        "refugees": 0.75,
    }
    assert calculate_severity(probs_refugees) == "MEDIUM"
    
    # Test death
    probs_death = {
        "death": 0.9,
    }
    assert calculate_severity(probs_death) == "HIGH"
    
    # Test combination of new critical categories
    probs_combined = {
        "missing_people": 0.6,
        "refugees": 0.7,
    }
    assert calculate_severity(probs_combined) == "HIGH"


# ---- Category Group Consistency Tests ----


def _display_to_internal(display_name: str) -> str:
    """Convert display name to internal name (reverse of to_display_name)."""
    # Handle special cases
    if display_name == "Search & Rescue":
        return "search_and_rescue"
    if display_name == "Medical Products":
        return "medical_products"
    if display_name == "Medical Help":
        return "medical_help"
    if display_name == "Missing People":
        return "missing_people"
    if display_name == "Other Infrastructure":
        return "other_infrastructure"
    if display_name == "Other Weather":
        return "other_weather"
    if display_name == "Other Aid":
        return "other_aid"
    if display_name == "Aid Centers":
        return "aid_centers"
    if display_name == "Child Alone":
        return "child_alone"
    if display_name == "Direct Report":
        return "direct_report"
    # Default: lowercase and replace spaces with underscores
    return display_name.lower().replace(" ", "_")


def test_critical_internal_categories_matches_critical_needs_group():
    """CRITICAL_INTERNAL_CATEGORIES must match Critical Needs group in CATEGORY_GROUPS."""
    critical_needs_display = CATEGORY_GROUPS["Critical Needs"]
    critical_needs_internal = {_display_to_internal(name) for name in critical_needs_display}
    
    assert critical_needs_internal == CRITICAL_INTERNAL_CATEGORIES, (
        f"CRITICAL_INTERNAL_CATEGORIES mismatch:\n"
        f"  Expected (from Critical Needs): {sorted(critical_needs_internal)}\n"
        f"  Actual: {sorted(CRITICAL_INTERNAL_CATEGORIES)}\n"
        f"  Missing: {critical_needs_internal - CRITICAL_INTERNAL_CATEGORIES}\n"
        f"  Extra: {CRITICAL_INTERNAL_CATEGORIES - critical_needs_internal}"
    )


def test_all_critical_needs_categories_have_display_name_mapping():
    """All critical internal categories must have valid display name mappings."""
    critical_needs_display = CATEGORY_GROUPS["Critical Needs"]
    
    for internal_cat in CRITICAL_INTERNAL_CATEGORIES:
        display_name = to_display_name(internal_cat)
        assert display_name in critical_needs_display, (
            f"Critical category '{internal_cat}' maps to '{display_name}' "
            f"which is not in Critical Needs group: {critical_needs_display}"
        )


def test_category_groups_contains_expected_critical_categories():
    """Critical Needs group must contain Missing People, Refugees, and Death."""
    critical_needs = CATEGORY_GROUPS["Critical Needs"]
    assert "Missing People" in critical_needs, "Missing People must be in Critical Needs"
    assert "Refugees" in critical_needs, "Refugees must be in Critical Needs"
    assert "Death" in critical_needs, "Death must be in Critical Needs"


def test_category_groups_other_does_not_contain_critical_categories():
    """Other group must not contain categories that are in Critical Needs."""
    other_group = CATEGORY_GROUPS["Other"]
    critical_needs = CATEGORY_GROUPS["Critical Needs"]
    
    overlap = set(other_group) & set(critical_needs)
    assert not overlap, (
        f"Categories in both Other and Critical Needs (should not happen): {overlap}"
    )


# ---- Option A: Simulated probability bands (Tripwire #6) ----


def test_simulated_probability_bands_label_0_below_half_label_1_at_or_above_half():
    """Simulated probs for label=0 must be < 0.5; for label=1 must be >= 0.5."""
    category_columns = ["water", "food", "shelter"]
    n_samples = 80
    for _ in range(n_samples):
        row = {
            "water": 1,
            "food": 0,
            "shelter": 0,
        }
        result = _simulated_probabilities(row, category_columns)
        assert result["water"] >= 0.5, "label=1 must yield prob >= 0.5"
        assert result["food"] < 0.5, "label=0 must yield prob < 0.5"
        assert result["shelter"] < 0.5, "label=0 must yield prob < 0.5"


def test_simulated_probability_bands_deterministic_with_mocked_random():
    """With fixed random, label=0 band [0.1, 0.2), label=1 band [0.85, 1.0]."""
    row = {"water": 1, "food": 0}
    category_columns = ["water", "food"]
    with patch.object(random, "uniform", side_effect=[0.0, 0.0]):
        result = _simulated_probabilities(row, category_columns)
    assert result["water"] == 0.85
    assert result["food"] == 0.1


# ---- Option B: _safe_float_prob ----


def test_safe_float_prob_none_nan_inf_returns_zero():
    """None, NaN, Infinity → 0.0 (no NaN/Inf in JSON)."""
    assert _safe_float_prob(None) == 0.0
    assert _safe_float_prob(float("nan")) == 0.0
    assert _safe_float_prob(float("inf")) == 0.0
    assert _safe_float_prob(float("-inf")) == 0.0


def test_safe_float_prob_non_numeric_returns_zero():
    """Non-numeric (string, etc.) → 0.0."""
    assert _safe_float_prob("x") == 0.0
    assert _safe_float_prob([]) == 0.0


def test_safe_float_prob_valid_returns_float_json_serializable():
    """Valid numeric input → float; output is JSON-serializable (no NaN/Inf)."""
    assert _safe_float_prob(0.5) == 0.5
    assert _safe_float_prob("0.7") == 0.7
    val = _safe_float_prob(0.5)
    assert not math.isnan(val) and not math.isinf(val)
    json.dumps({"p": val})


# ---- Option B: _safe_category_display ----


def test_safe_category_display_none_nan_returns_unknown():
    """None and NaN keys → 'Unknown' (no crash, no NaN in JSON)."""
    assert _safe_category_display(None) == "Unknown"
    assert _safe_category_display(float("nan")) == "Unknown"


def test_safe_category_display_valid_returns_display_name():
    """Valid internal name → display name."""
    assert _safe_category_display("water") == "Water"
    assert _safe_category_display("search_and_rescue") == "Search & Rescue"


def test_safe_category_display_output_never_nan_or_none():
    """Output is always a string (JSON-safe)."""
    for key in (None, float("nan"), "water", "", 123):
        out = _safe_category_display(key)
        assert isinstance(out, str)
        assert out is not None
        json.dumps({"name": out})


# ---- Option B: _safe_label_value (extend coverage) ----


def test_safe_label_value_inf_and_invalid_returns_zero():
    """Inf and non-coercible values → 0."""
    assert _safe_label_value(float("inf")) == 0
    assert _safe_label_value("x") == 0


def test_safe_label_value_valid_returns_int_json_serializable():
    """Valid 0/1 → int; output JSON-serializable."""
    assert _safe_label_value(0) == 0
    assert _safe_label_value(1) == 1
    assert _safe_label_value(1.0) == 1
    json.dumps({"count": _safe_label_value(1)})


# ---- Option B: _safe_text_value (extend coverage) ----


def test_safe_text_value_inf_returns_string_no_nan():
    """Inf as input → str representation; output usable in JSON."""
    out = _safe_text_value(float("inf"))
    assert isinstance(out, str)
    json.dumps({"text": out})
