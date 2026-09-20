"""Regression tests: POST /api/classify applies hierarchy correction."""
from __future__ import annotations

from typing import Any, Dict


class _FakeModelService:
    """Model service stub returning crafted probabilities and thresholds."""

    def __init__(
        self,
        probabilities: Dict[str, float],
        thresholds: Dict[str, float] | None = None,
    ) -> None:
        self._probabilities = probabilities
        self._thresholds = thresholds or {}

    def predict(self, text: str) -> dict:
        labels = {
            name: 1 if prob >= self._thresholds.get(name, 0.5) else 0
            for name, prob in self._probabilities.items()
        }
        return {"labels": labels, "probabilities": dict(self._probabilities)}

    def get_thresholds_map(self) -> Dict[str, float]:
        return dict(self._thresholds)


def _category_by_name(payload: dict, name: str) -> dict | None:
    for item in payload.get("categories", []):
        if item.get("name") == name:
            return item
    return None


def _category_names(payload: dict) -> set[str]:
    return {item["name"] for item in payload.get("categories", [])}


def test_api_classify_activates_parent_when_child_clears_threshold(app, client):
    """Child above threshold with parent below must still surface the parent."""
    # water clears 0.30; aid_related stays below 0.50 until hierarchy forces parent=1
    probabilities = {
        "related": 0.9,
        "aid_related": 0.40,
        "water": 0.40,
        "food": 0.01,
        "medical_help": 0.01,
        "child_alone": 0.01,
    }
    thresholds = {
        "related": 0.5,
        "aid_related": 0.5,
        "water": 0.30,
        "food": 0.5,
        "medical_help": 0.5,
        "child_alone": 0.5,
    }
    original = getattr(app, "model_service", None)
    app.model_service = _FakeModelService(probabilities, thresholds)
    try:
        response = client.post(
            "/api/classify",
            json={"message": "Need water urgently"},
        )
        assert response.status_code == 200
        payload = response.get_json()
        names = _category_names(payload)
        assert "Water" in names
        assert "Aid Related" in names
        aid = _category_by_name(payload, "Aid Related")
        assert aid is not None
        assert aid["meetsThreshold"] is True
        assert aid["confidence"] < aid["threshold"]
    finally:
        app.model_service = original


def test_api_classify_isolated_child_alone_does_not_activate_aid_related(app, client):
    """child_alone is not an aid_related taxonomy child; high score must not create Aid Related.

    Exclusion-set behavior for labels that *are* taxonomy children is covered in
    tests/test_hierarchy.py; this API test guards the live taxonomy + response path.
    """
    probabilities = {
        "related": 0.9,
        "aid_related": 0.10,
        "water": 0.01,
        "food": 0.01,
        "child_alone": 0.95,
    }
    thresholds = {
        "related": 0.5,
        "aid_related": 0.5,
        "water": 0.5,
        "food": 0.5,
        "child_alone": 0.5,
    }
    original = getattr(app, "model_service", None)
    app.model_service = _FakeModelService(probabilities, thresholds)
    try:
        response = client.post(
            "/api/classify",
            json={"message": "Child alone situation"},
        )
        assert response.status_code == 200
        payload = response.get_json()
        names = _category_names(payload)
        assert "Child Alone" in names
        assert "Aid Related" not in names
    finally:
        app.model_service = original


def test_api_classify_debug_exposes_raw_and_fixed(app, client):
    """Debug mode uses nested raw/fixed maps (replaces flat debug probabilities/labels)."""
    probabilities = {
        "related": 0.9,
        "aid_related": 0.40,
        "water": 0.40,
    }
    thresholds = {
        "related": 0.5,
        "aid_related": 0.5,
        "water": 0.30,
    }
    original = getattr(app, "model_service", None)
    app.model_service = _FakeModelService(probabilities, thresholds)
    try:
        response = client.post(
            "/api/classify?debug=1",
            json={"message": "Need water urgently"},
        )
        assert response.status_code == 200
        payload = response.get_json()
        debug: Dict[str, Any] = payload.get("debug") or {}
        assert "thresholds" in debug
        assert "raw" in debug and "fixed" in debug
        assert "probabilities" in debug["raw"] and "labels" in debug["raw"]
        assert "probabilities" in debug["fixed"] and "labels" in debug["fixed"]
        assert debug["raw"]["labels"].get("aid_related") == 0
        assert debug["fixed"]["labels"].get("aid_related") == 1
        assert debug["raw"]["labels"].get("water") == 1
        assert debug["fixed"]["labels"].get("water") == 1
        assert "probabilities" not in debug  # nested under raw/fixed only
        assert "labels" not in debug
    finally:
        app.model_service = original
