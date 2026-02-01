"""Contract smoke tests for stubbed dashboard API endpoints."""

import random
from unittest.mock import patch

import pandas as pd
import pytest

from app.routes.api import (
    _row_to_feed_item,
    _safe_label_value,
    _safe_text_value,
    _simulated_probabilities,
    genre_to_source,
)


class StubDataService:
    """Minimal data service stub for feed pagination tests."""

    def __init__(self, df: pd.DataFrame, category_columns: list):
        self._df = df
        self._category_columns = category_columns

    def get_data(self) -> pd.DataFrame:
        return self._df.copy()

    def get_category_columns(self) -> list:
        return list(self._category_columns)


def _make_feed_df(n_rows: int, category_columns: list | None = None) -> pd.DataFrame:
    """Build a minimal DataFrame for feed tests (id, message, original, genre + categories)."""
    if category_columns is None:
        category_columns = ["water", "food"]
    rows = []
    for i in range(1, n_rows + 1):
        row = {
            "id": i,
            "message": f"message {i}",
            "original": None,
            "genre": "direct",
        }
        for col in category_columns:
            row[col] = 1 if (i % 2 == 0) else 0
        rows.append(row)
    return pd.DataFrame(rows)


def test_api_feed_contract(client):
    response = client.get("/api/feed")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    assert "items" in payload
    assert "pagination" in payload

    items = payload["items"]
    assert isinstance(items, list)
    assert items
    item = items[0]
    for key in (
        "id",
        "timestamp",
        "source",
        "content",
        "language",
        "riskLevel",
        "categories",
        "classifications",
        "isTranslated",
    ):
        assert key in item

    pagination = payload["pagination"]
    for key in ("page", "limit", "total", "totalPages"):
        assert key in pagination


def test_api_metrics_contract(client):
    response = client.get("/api/metrics")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    for key in ("volToday", "flaggedRate", "topCategories", "trendData"):
        assert key in payload

    assert isinstance(payload["topCategories"], list)
    assert isinstance(payload["trendData"], list)


def test_api_categories_contract(client):
    response = client.get("/api/categories")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    assert "categories" in payload
    assert "groups" in payload

    categories = payload["categories"]
    assert isinstance(categories, list)
    assert categories
    category = categories[0]
    for key in ("internal", "display", "count"):
        assert key in category


def test_api_classify_contract(client):
    response = client.post("/api/classify", json={"message": "Need water and medical aid"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    for key in ("categories", "severity", "maxConfidence", "avgConfidence"):
        assert key in payload

    categories = payload["categories"]
    assert isinstance(categories, list)
    assert categories
    category = categories[0]
    for key in ("name", "confidence", "volume"):
        assert key in category


def test_safe_label_value_handles_nan():
    assert _safe_label_value(None) == 0
    assert _safe_label_value(float("nan")) == 0
    assert _safe_label_value("0") == 0
    assert _safe_label_value("1") == 1
    assert _safe_label_value("invalid") == 0


def test_simulated_probabilities_accept_nan():
    row = {"medical_help": float("nan"), "water": 1, "food": 0}
    result = _simulated_probabilities(row, ["medical_help", "water", "food"])
    assert set(result.keys()) == {"medical_help", "water", "food"}
    assert 0.1 <= result["medical_help"] <= 0.2
    assert 0.85 <= result["water"] <= 1.0
    assert 0.1 <= result["food"] <= 0.2


def test_row_to_feed_item_handles_nan_message_genre():
    row = {"id": 7, "message": float("nan"), "genre": float("nan")}
    item = _row_to_feed_item(row, [])
    assert item["content"] == ""
    assert item["source"] == "Direct Report"


# ---- Pagination: use stub data_service so total/offset/limit are deterministic ----


def test_feed_pagination_offset_in_range(app, client):
    """Offset in range: limit=10, offset=5, total=25 → items are indices 5–14 (ids 6–15)."""
    df = _make_feed_df(25, ["water", "food"])
    stub = StubDataService(df, ["water", "food"])
    original = app.data_service
    app.data_service = stub
    try:
        response = client.get("/api/feed?limit=10&offset=5")
        assert response.status_code == 200
        data = response.get_json()
        items = data["items"]
        pagination = data["pagination"]
        assert len(items) == 10
        assert pagination["total"] == 25
        assert pagination["limit"] == 10
        assert pagination["page"] == 1
        assert pagination["totalPages"] == 3
        assert items[0]["id"] == "SIG-6"
        assert items[-1]["id"] == "SIG-15"
    finally:
        app.data_service = original


def test_feed_pagination_clamp_out_of_range_offset(app, client):
    """When offset >= total: page == totalPages, items are last page, effective_offset clamped."""
    df = _make_feed_df(25, ["water", "food"])
    stub = StubDataService(df, ["water", "food"])
    original = app.data_service
    app.data_service = stub
    try:
        response = client.get("/api/feed?limit=10&offset=30")
        assert response.status_code == 200
        data = response.get_json()
        items = data["items"]
        pagination = data["pagination"]
        assert pagination["page"] == pagination["totalPages"]
        assert pagination["totalPages"] == 3
        assert pagination["total"] == 25
        assert len(items) == 5
        assert items[0]["id"] == "SIG-21"
        assert items[-1]["id"] == "SIG-25"
    finally:
        app.data_service = original


def test_feed_pagination_empty_dataset(app, client):
    """Empty dataset: items=[], total=0, totalPages=0, page=1."""
    df = _make_feed_df(0, ["water", "food"])
    stub = StubDataService(df, ["water", "food"])
    original = app.data_service
    app.data_service = stub
    try:
        response = client.get("/api/feed")
        assert response.status_code == 200
        data = response.get_json()
        assert data["items"] == []
        assert data["pagination"]["total"] == 0
        assert data["pagination"]["totalPages"] == 0
        assert data["pagination"]["page"] == 1
    finally:
        app.data_service = original


def test_feed_limit_bounds(app, client):
    """limit=0 becomes 1; limit=999 is capped at 100."""
    df = _make_feed_df(150, ["water", "food"])
    stub = StubDataService(df, ["water", "food"])
    original = app.data_service
    app.data_service = stub
    try:
        r0 = client.get("/api/feed?limit=0")
        assert r0.status_code == 200
        assert r0.get_json()["pagination"]["limit"] == 1
        assert len(r0.get_json()["items"]) == 1

        r999 = client.get("/api/feed?limit=999")
        assert r999.status_code == 200
        assert r999.get_json()["pagination"]["limit"] == 100
        assert len(r999.get_json()["items"]) == 100
    finally:
        app.data_service = original


def test_feed_filter_categories_offset_clamp(app, client):
    """When filters reduce results and offset is out of range, return last page (valid items)."""
    df = _make_feed_df(25, ["water", "food"])
    stub = StubDataService(df, ["water", "food"])
    original = app.data_service
    app.data_service = stub
    try:
        response = client.get("/api/feed?limit=10&offset=20&categories[]=water")
        assert response.status_code == 200
        data = response.get_json()
        pagination = data["pagination"]
        items = data["items"]
        total = pagination["total"]
        assert total <= 13
        assert pagination["page"] == pagination["totalPages"]
        if total > 0:
            assert len(items) >= 1
    finally:
        app.data_service = original


# ---- Genre / _safe_text_value ----


def test_safe_text_value_none_nan():
    assert _safe_text_value(None) == ""
    assert _safe_text_value(float("nan")) == ""


def test_safe_text_value_numpy_nan():
    try:
        import numpy as np
    except ImportError:
        pytest.skip("numpy not available")
    assert _safe_text_value(np.nan) == ""


def test_genre_unknown_maps_to_x():
    assert genre_to_source("unknown") == "X"


# ---- Classification inclusion threshold ----


def test_classification_inclusion_threshold_label_0_excluded_label_1_included():
    """With random.uniform fixed to 0.0, label=0 categories not in classifications, label=1 in."""
    row = {"id": 1, "water": 1, "food": 0, "shelter": 0}
    category_columns = ["water", "food", "shelter"]
    with patch.object(random, "uniform", return_value=0.0):
        item = _row_to_feed_item(row, category_columns)
    classifications = {c["category"] for c in item["classifications"]}
    assert "Water" in classifications
    assert "Food" not in classifications
    assert "Shelter" not in classifications
