"""Contract smoke tests for stubbed dashboard API endpoints."""

import hashlib
import json
import math
import random
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from app.routes.api import (
    _row_to_feed_item,
    _safe_label_value,
    _safe_optional_prob,
    _safe_text_value,
    _simulated_probabilities,
    genre_to_source,
)
from disasterproject.utils.config import TARGET_COLUMNS


class StubDataService:
    """Minimal data service stub for feed pagination tests."""

    def __init__(self, df: pd.DataFrame, category_columns: list):
        self._df = df
        self._category_columns = category_columns

    def get_data(self) -> pd.DataFrame:
        return self._df.copy()

    def get_category_columns(self) -> list:
        return list(self._category_columns)


def _json_contains_no_nan_or_infinity(obj) -> bool:
    """Return True if obj (JSON-serializable structure) contains no float('nan') or float('inf')."""
    if isinstance(obj, float):
        return not (math.isnan(obj) or math.isinf(obj))
    if isinstance(obj, dict):
        return all(_json_contains_no_nan_or_infinity(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_json_contains_no_nan_or_infinity(v) for v in obj)
    return True


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


def test_metrics_empty_dataset(app, client):
    """Empty dataset: volToday=0, flaggedRate=0, topCategories=[], trendData has 7 entries count=0."""
    df = _make_feed_df(0, ["water", "food"])
    stub = StubDataService(df, ["water", "food"])
    original = app.data_service
    app.data_service = stub
    try:
        response = client.get("/api/metrics")
        assert response.status_code == 200
        data = response.get_json()
        assert data["volToday"] == 0
        assert data["flaggedRate"] == 0.0
        assert data["topCategories"] == []
        assert len(data["trendData"]) == 7
        for entry in data["trendData"]:
            assert "time" in entry and "count" in entry
            assert entry["count"] == 0
    finally:
        app.data_service = original


def test_metrics_nan_in_categories_does_not_crash(app, client):
    """DataFrame with NaN in category columns does not crash metrics endpoint."""
    df = pd.DataFrame(
        [
            {"id": 1, "message": "a", "genre": "direct", "water": 1, "food": float("nan")},
            {"id": 2, "message": "b", "genre": "news", "water": float("nan"), "food": 0},
        ]
    )
    stub = StubDataService(df, ["water", "food"])
    original = app.data_service
    app.data_service = stub
    try:
        response = client.get("/api/metrics")
        assert response.status_code == 200
        data = response.get_json()
        assert "topCategories" in data
        assert "volToday" in data
        for cat in data["topCategories"]:
            assert isinstance(cat["count"], int)
    finally:
        app.data_service = original


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


def test_api_classify_empty_message_returns_400(client):
    """Empty or missing message returns 400 with error."""
    r1 = client.post("/api/classify", json={})
    assert r1.status_code == 400
    assert r1.get_json().get("error")

    r2 = client.post("/api/classify", json={"message": ""})
    assert r2.status_code == 400

    r3 = client.post("/api/classify", json={"message": "   "})
    assert r3.status_code == 400


def test_api_classify_rejects_overlong_message(client):
    """Messages exceeding the max length are rejected with 400."""
    too_long_message = "a" * 1001
    response = client.post("/api/classify", json={"message": too_long_message})
    assert response.status_code == 400
    assert response.get_json().get("error")


def test_api_classify_no_model_service_returns_503(app, client):
    """When model_service is not configured, classify returns 503."""
    original = getattr(app, "model_service", None)
    app.model_service = None
    try:
        response = client.post("/api/classify", json={"message": "Need water"})
        assert response.status_code == 503
        assert response.get_json().get("error")
    finally:
        app.model_service = original


def test_safe_label_value_handles_nan():
    assert _safe_label_value(None) == 0
    assert _safe_label_value(float("nan")) == 0
    assert _safe_label_value("0") == 0
    assert _safe_label_value("1") == 1
    assert _safe_label_value("invalid") == 0


def test_simulated_probabilities_accept_nan():
    """Test that NaN values are handled correctly (treated as 0, get low probabilities)."""
    row = {"medical_help": float("nan"), "water": 1, "food": 0}
    result = _simulated_probabilities(row, ["medical_help", "water", "food"])
    assert set(result.keys()) == {"medical_help", "water", "food"}
    # NaN treated as 0, so gets low probability (new range: 0.05-0.30 depending on context)
    assert 0.05 <= result["medical_help"] <= 0.35
    # water=1 gets high probability (new range: 0.70-0.98 for non-critical, but water is critical so 0.80-0.98)
    assert 0.70 <= result["water"] <= 0.98
    # food=0 gets low probability
    assert 0.05 <= result["food"] <= 0.35


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


def test_feed_categories_only_from_actual_labels():
    """
    Regression test: categories shown must only come from actual label=1 in training data.
    
    This prevents the bug where messages with no labels (only related=1) were showing
    random categories due to simulated probabilities being assigned to label=0 categories.
    """
    category_columns = ["electricity", "infrastructure_related", "medical_help", "water", "food"]
    
    # Test case 1: Message with no labels (only related=1) should show empty categories
    row_no_labels = {
        "id": 2,
        "message": "Weather update - a cold front from Cuba that could pass over Haiti",
        "original": None,
        "genre": "direct",
        "related": 1,
        "electricity": 0,
        "infrastructure_related": 0,
        "medical_help": 0,
        "water": 0,
        "food": 0,
    }
    item_no_labels = _row_to_feed_item(row_no_labels, category_columns)
    assert item_no_labels["categories"] == [], (
        "Messages with no category labels should show empty categories list, "
        "not random categories from simulated probabilities"
    )
    assert item_no_labels["classifications"] == []
    
    # Test case 2: Message with actual labels should only show those labels
    row_with_labels = {
        "id": 9,
        "message": "UN reports Leogane 80-90 destroyed. Only Hospital St. Croix functioning.",
        "original": None,
        "genre": "direct",
        "related": 1,
        "electricity": 0,
        "infrastructure_related": 1,
        "medical_help": 0,
        "water": 0,
        "food": 0,
    }
    item_with_labels = _row_to_feed_item(row_with_labels, category_columns)
    categories_set = set(item_with_labels["categories"])
    assert "Infrastructure" in categories_set, "Should show Infrastructure (label=1)"
    assert "Electricity" not in categories_set, "Should NOT show Electricity (label=0)"
    assert "Medical Help" not in categories_set, "Should NOT show Medical Help (label=0)"
    
    # Verify classifications also only include label=1 categories
    classification_categories = {c["category"] for c in item_with_labels["classifications"]}
    assert "Infrastructure" in classification_categories
    assert "Electricity" not in classification_categories
    assert "Medical Help" not in classification_categories


def test_risk_level_consistency_with_labeled_categories():
    """
    Regression test: risk level should only consider categories with label=1.
    
    This ensures that a message with no labeled critical categories cannot get
    HIGH/MEDIUM risk level based on simulated probabilities for label=0 categories.
    """
    category_columns = ["medical_help", "water", "food", "search_and_rescue", "infrastructure_related"]
    
    # Test case: Message with no critical categories labeled (all critical have label=0)
    # Even if simulated probabilities for critical categories are high, risk should be LOW
    # Use a non-critical category like "infrastructure_related" or "buildings"
    row_no_critical_labels = {
        "id": 100,
        "message": "General weather update - no immediate emergency",
        "original": None,
        "genre": "news",
        "related": 1,
        "medical_help": 0,  # Critical category with label=0
        "water": 0,  # Critical category with label=0
        "food": 0,  # Critical category with label=0
        "search_and_rescue": 0,  # Critical category with label=0
        "infrastructure_related": 1,  # Non-critical category with label=1
    }
    
    # Run multiple times to account for randomness in probability simulation
    risk_levels = []
    for _ in range(10):
        item = _row_to_feed_item(row_no_critical_labels, category_columns)
        risk_levels.append(item["riskLevel"])
    
    # All risk levels should be LOW since no critical categories have label=1
    # (Even if simulated probabilities for label=0 critical categories are high)
    assert all(level == "LOW" for level in risk_levels), (
        "Messages with no labeled critical categories should always get LOW risk level, "
        "regardless of simulated probabilities for label=0 categories"
    )
    
    # Test case: Message with labeled critical categories should get appropriate risk level
    row_with_critical_labels = {
        "id": 101,
        "message": "Urgent: Medical assistance needed, water supplies running low",
        "original": None,
        "genre": "direct",
        "related": 1,
        "medical_help": 1,  # Critical category with label=1
        "water": 1,  # Critical category with label=1
        "food": 0,
        "search_and_rescue": 0,
        "shelter": 0,
    }
    
    item_with_critical = _row_to_feed_item(row_with_critical_labels, category_columns)
    # Should get HIGH or MEDIUM since we have 2 critical categories with label=1
    assert item_with_critical["riskLevel"] in ["HIGH", "MEDIUM"], (
        "Messages with labeled critical categories should get HIGH or MEDIUM risk level"
    )
    
    # Verify classifications include the critical categories
    classification_categories = {c["category"] for c in item_with_critical["classifications"]}
    assert "Medical Help" in classification_categories
    assert "Water" in classification_categories


# ---- Data Reality Gate: no NaN/Infinity in JSON ----


def test_api_responses_contain_no_nan_or_infinity(client):
    """Invariant: feed, metrics, categories, and classify responses contain no NaN or Infinity in JSON."""
    feed_resp = client.get("/api/feed")
    assert feed_resp.status_code == 200
    assert _json_contains_no_nan_or_infinity(feed_resp.get_json()), "GET /api/feed must not emit NaN/Infinity"

    metrics_resp = client.get("/api/metrics")
    assert metrics_resp.status_code == 200
    assert _json_contains_no_nan_or_infinity(metrics_resp.get_json()), "GET /api/metrics must not emit NaN/Infinity"

    categories_resp = client.get("/api/categories")
    assert categories_resp.status_code == 200
    assert _json_contains_no_nan_or_infinity(categories_resp.get_json()), "GET /api/categories must not emit NaN/Infinity"

    classify_resp = client.post("/api/classify", json={"message": "Need water and medical aid"})
    assert classify_resp.status_code == 200
    assert _json_contains_no_nan_or_infinity(classify_resp.get_json()), "POST /api/classify must not emit NaN/Infinity"

    dashboard_resp = client.get("/api/model-info/dashboard")
    assert dashboard_resp.status_code == 200
    assert _json_contains_no_nan_or_infinity(dashboard_resp.get_json()), "GET /api/model-info/dashboard must not emit NaN/Infinity"


def test_api_model_info_dashboard_contract(client):
    """GET /api/model-info/dashboard returns contract shape: model, metrics, categories, criticalThresholds, registry."""
    response = client.get("/api/model-info/dashboard")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    for key in ("model", "metrics", "categories", "criticalThresholds", "registry"):
        assert key in payload, f"Missing key: {key}"

    model = payload["model"]
    for key in ("id", "version", "lastUpdated", "status", "generatedAt"):
        assert key in model, f"model missing key: {key}"

    metrics = payload["metrics"]
    for key in ("f1", "precision", "recall", "evalCriticalRecall"):
        assert key in metrics, f"metrics missing key: {key}"
    assert isinstance(metrics["f1"], (int, float))
    assert isinstance(metrics["precision"], (int, float))
    assert isinstance(metrics["recall"], (int, float))
    eval_critical = metrics["evalCriticalRecall"]
    assert eval_critical is None or isinstance(eval_critical, (int, float))
    if isinstance(eval_critical, float):
        assert not (math.isnan(eval_critical) or math.isinf(eval_critical))
        assert 0.0 <= eval_critical <= 1.0

    categories = payload["categories"]
    assert isinstance(categories, list)
    if categories:
        cat = categories[0]
        for key in ("key", "label", "f1", "precision", "recall", "support"):
            assert key in cat, f"categories[0] missing key: {key}"

    critical = payload["criticalThresholds"]
    assert isinstance(critical, list)
    if critical:
        ct = critical[0]
        for key in ("key", "label", "threshold"):
            assert key in ct, f"criticalThresholds[0] missing key: {key}"

    registry = payload["registry"]
    assert isinstance(registry, list)
    if registry:
        reg = registry[0]
        for key in ("name", "size", "type"):
            assert key in reg, f"registry[0] missing key: {key}"

    assert _json_contains_no_nan_or_infinity(payload), "Dashboard payload must not contain NaN/Infinity"


def test_model_info_dashboard_null_realism(client, tmp_path):
    """Empty model dir fails closed: unavailable status, no orphan MODEL_INFO metadata."""
    from unittest.mock import patch

    with patch("app.routes.api._get_model_dir", return_value=tmp_path):
        response = client.get("/api/model-info/dashboard")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    assert "model" in payload
    assert payload["model"]["status"] == "unavailable"
    assert payload["model"]["id"] in ("unknown", "UNKNOWN")
    assert payload["model"]["version"] == "unknown"
    assert payload["model"]["provenanceError"] == "Production model provenance unavailable"
    assert payload["model"]["provenanceCode"] == "active_model_missing"
    assert "metrics" in payload
    assert payload["metrics"]["f1"] == 0.0
    assert payload["metrics"]["precision"] == 0.0
    assert payload["metrics"]["recall"] == 0.0
    assert payload["metrics"]["evalCriticalRecall"] is None
    assert payload["categories"] == []
    assert payload["criticalThresholds"] == []
    assert isinstance(payload["registry"], list)
    assert _json_contains_no_nan_or_infinity(payload), "Dashboard with empty model dir must not emit NaN/Infinity"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_resolver_valid_dashboard_bundle(
    model_dir: Path,
    *,
    performance: dict | None = None,
    validation_results: dict | None = None,
) -> Path:
    """Write a stem-bound production bundle the dashboard resolver accepts."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "disaster_lr_v_test_prod_2026-09-21.pkl"
    model_path.write_bytes(b"dashboard-contract-model")
    thresholds_path = model_dir / f"{model_path.stem}_thresholds.json"
    labels_path = model_dir / f"{model_path.stem}_labels.json"
    thresholds_map = {label: 0.5 for label in TARGET_COLUMNS}
    thresholds_path.write_text(
        json.dumps({"thresholds": thresholds_map, "critical_only": {"water": 0.5}}),
        encoding="utf-8",
    )
    labels_path.write_text(json.dumps(list(TARGET_COLUMNS)), encoding="utf-8")
    info = {
        "version": "test",
        "status": "production",
        "algorithm": "lr",
        "algorithm_name": "LogisticRegression",
        "performance": performance or {},
        "validation_results": validation_results or {},
        "sha256": _sha256(model_path),
        "thresholds_sha256": _sha256(thresholds_path),
        "labels_sha256": _sha256(labels_path),
    }
    (model_dir / "MODEL_INFO.json").write_text(json.dumps(info), encoding="utf-8")
    return model_path


def test_safe_optional_prob_rejects_invalid_values():
    """_safe_optional_prob keeps valid [0,1] probs and maps everything else to None."""
    assert _safe_optional_prob(0.0) == 0.0
    assert _safe_optional_prob(1.0) == 1.0
    assert _safe_optional_prob(0.6149) == pytest.approx(0.6149)
    assert _safe_optional_prob(None) is None
    assert _safe_optional_prob("not-a-number") is None
    assert _safe_optional_prob(True) is None
    assert _safe_optional_prob(False) is None
    assert _safe_optional_prob(float("nan")) is None
    assert _safe_optional_prob(float("inf")) is None
    assert _safe_optional_prob(-0.01) is None
    assert _safe_optional_prob(1.01) is None


def test_eval_critical_recall_prefers_performance_block(client, tmp_path):
    """When performance.eval_critical_recall is present, the API returns that value."""
    _write_resolver_valid_dashboard_bundle(
        tmp_path,
        performance={"eval_critical_recall": 0.61},
        validation_results={"eval_critical_recall": 0.42},
    )
    with patch("app.routes.api._get_model_dir", return_value=tmp_path):
        with patch(
            "app.routes.api._resolve_active_production_model_path",
            return_value=tmp_path / "disaster_lr_v_test_prod_2026-09-21.pkl",
        ):
            response = client.get("/api/model-info/dashboard")
    assert response.status_code == 200
    metrics = response.get_json()["metrics"]
    assert metrics["evalCriticalRecall"] == pytest.approx(0.61)


def test_eval_critical_recall_falls_back_to_validation_results(client, tmp_path):
    """When performance is missing the field, validation_results is used."""
    _write_resolver_valid_dashboard_bundle(
        tmp_path,
        performance={},
        validation_results={"eval_critical_recall": 0.42},
    )
    with patch("app.routes.api._get_model_dir", return_value=tmp_path):
        with patch(
            "app.routes.api._resolve_active_production_model_path",
            return_value=tmp_path / "disaster_lr_v_test_prod_2026-09-21.pkl",
        ):
            response = client.get("/api/model-info/dashboard")
    assert response.status_code == 200
    metrics = response.get_json()["metrics"]
    assert metrics["evalCriticalRecall"] == pytest.approx(0.42)


@pytest.mark.parametrize(
    "bad_value",
    [None, "bad", -0.1, 1.5],
)
def test_eval_critical_recall_null_when_missing_or_malformed(client, tmp_path, bad_value):
    """Missing or invalid eval_critical_recall values surface as null, not 0.0."""
    _write_resolver_valid_dashboard_bundle(
        tmp_path,
        performance={"eval_critical_recall": bad_value},
    )
    with patch("app.routes.api._get_model_dir", return_value=tmp_path):
        with patch(
            "app.routes.api._resolve_active_production_model_path",
            return_value=tmp_path / "disaster_lr_v_test_prod_2026-09-21.pkl",
        ):
            response = client.get("/api/model-info/dashboard")
    assert response.status_code == 200
    assert response.get_json()["metrics"]["evalCriticalRecall"] is None


def test_eval_critical_recall_null_when_key_absent(client, tmp_path):
    """Absent eval_critical_recall keys yield null."""
    _write_resolver_valid_dashboard_bundle(
        tmp_path,
        performance={},
        validation_results={},
    )
    with patch("app.routes.api._get_model_dir", return_value=tmp_path):
        with patch(
            "app.routes.api._resolve_active_production_model_path",
            return_value=tmp_path / "disaster_lr_v_test_prod_2026-09-21.pkl",
        ):
            response = client.get("/api/model-info/dashboard")
    assert response.status_code == 200
    assert response.get_json()["metrics"]["evalCriticalRecall"] is None


def test_dashboard_fails_closed_when_model_info_exists_without_pickle(client, tmp_path):
    """Orphan MODEL_INFO alone must not surface production metadata."""
    (tmp_path / "MODEL_INFO.json").write_text(
        json.dumps(
            {
                "version": "orphan-v1",
                "status": "production",
                "performance": {"eval_critical_recall": 0.88},
            }
        ),
        encoding="utf-8",
    )
    with patch("app.routes.api._get_model_dir", return_value=tmp_path):
        with patch(
            "app.routes.api._resolve_active_production_model_path",
            return_value=None,
        ):
            response = client.get("/api/model-info/dashboard")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["model"]["status"] == "unavailable"
    assert payload["model"]["version"] == "unknown"
    assert payload["model"]["provenanceCode"] == "active_model_missing"
    assert payload["metrics"]["evalCriticalRecall"] is None
    assert payload["metrics"]["f1"] == 0.0


def test_model_info_returns_validated_production_metadata(client, tmp_path):
    """GET /api/model-info uses resolver-valid bundle metadata."""
    _write_resolver_valid_dashboard_bundle(
        tmp_path,
        performance={"optimized_f1_weighted": 0.8975},
        validation_results={"optimized_f1_weighted": 0.1},
    )
    active = tmp_path / "disaster_lr_v_test_prod_2026-09-21.pkl"
    with patch("app.routes.api._get_model_dir", return_value=tmp_path):
        with patch(
            "app.routes.api._resolve_active_production_model_path",
            return_value=active,
        ):
            response = client.get("/api/model-info")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["version"] == "test"
    assert payload["status"] == "production"
    assert payload["f1_score"] == pytest.approx(0.8975)
    assert payload["hierarchy_violations"] == 0.0


def test_optimized_f1_prefers_performance_over_validation(client, tmp_path):
    """performance.optimized_f1_weighted wins over differing validation_results."""
    active = _write_resolver_valid_dashboard_bundle(
        tmp_path,
        performance={"optimized_f1_weighted": 0.8975},
        validation_results={"optimized_f1_weighted": 0.1},
    )
    with patch("app.routes.api._get_model_dir", return_value=tmp_path):
        with patch(
            "app.routes.api._resolve_active_production_model_path",
            return_value=active,
        ):
            info_response = client.get("/api/model-info")
            dash_response = client.get("/api/model-info/dashboard")
    assert info_response.status_code == 200
    assert info_response.get_json()["f1_score"] == pytest.approx(0.8975)
    assert dash_response.status_code == 200
    assert dash_response.get_json()["metrics"]["f1"] == pytest.approx(0.8975)


def test_optimized_f1_falls_back_to_validation_results(client, tmp_path):
    """validation_results.optimized_f1_weighted used when performance field absent."""
    active = _write_resolver_valid_dashboard_bundle(
        tmp_path,
        performance={"eval_critical_recall": 0.61},
        validation_results={"optimized_f1_weighted": 0.75},
    )
    with patch("app.routes.api._get_model_dir", return_value=tmp_path):
        with patch(
            "app.routes.api._resolve_active_production_model_path",
            return_value=active,
        ):
            info_response = client.get("/api/model-info")
            dash_response = client.get("/api/model-info/dashboard")
    assert info_response.status_code == 200
    assert info_response.get_json()["f1_score"] == pytest.approx(0.75)
    assert dash_response.status_code == 200
    assert dash_response.get_json()["metrics"]["f1"] == pytest.approx(0.75)


def test_legacy_f1_aliases_do_not_populate_f1_wire_keys(client, tmp_path):
    """Naked f1_weighted and baseline_f1_micro must not populate f1_score / metrics.f1."""
    active = _write_resolver_valid_dashboard_bundle(
        tmp_path,
        performance={"f1_weighted": 0.99, "baseline_f1_micro": 0.77},
        validation_results={"f1_weighted": 0.88, "baseline_f1_micro": 0.66},
    )
    with patch("app.routes.api._get_model_dir", return_value=tmp_path):
        with patch(
            "app.routes.api._resolve_active_production_model_path",
            return_value=active,
        ):
            info_response = client.get("/api/model-info")
            dash_response = client.get("/api/model-info/dashboard")
    assert info_response.status_code == 200
    assert info_response.get_json()["f1_score"] is None
    assert dash_response.status_code == 200
    assert dash_response.get_json()["metrics"]["f1"] == 0.0


def test_missing_optimized_f1_defaults_dashboard_zero_api_null(client, tmp_path):
    """Missing explicit OP F1: dashboard metrics.f1=0.0; /api/model-info f1_score=null."""
    active = _write_resolver_valid_dashboard_bundle(
        tmp_path,
        performance={"eval_critical_recall": 0.61},
        validation_results={},
    )
    with patch("app.routes.api._get_model_dir", return_value=tmp_path):
        with patch(
            "app.routes.api._resolve_active_production_model_path",
            return_value=active,
        ):
            info_response = client.get("/api/model-info")
            dash_response = client.get("/api/model-info/dashboard")
    assert info_response.status_code == 200
    assert info_response.get_json()["f1_score"] is None
    assert dash_response.status_code == 200
    assert dash_response.get_json()["metrics"]["f1"] == 0.0


def test_model_info_fails_closed_for_orphan_model_info(client, tmp_path):
    """Orphan MODEL_INFO without an active pickle must not look healthy."""
    (tmp_path / "MODEL_INFO.json").write_text(
        json.dumps(
            {
                "version": "orphan-v1",
                "status": "production",
                "performance": {"optimized_f1_weighted": 0.99},
            }
        ),
        encoding="utf-8",
    )
    with patch("app.routes.api._get_model_dir", return_value=tmp_path):
        with patch(
            "app.routes.api._resolve_active_production_model_path",
            return_value=None,
        ):
            response = client.get("/api/model-info")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "unavailable"
    assert payload["version"] == "unknown"
    assert payload["f1_score"] is None
    assert payload["provenanceCode"] == "active_model_missing"
    assert payload["provenanceError"] == "Production model provenance unavailable"


def test_model_info_fails_closed_on_hash_mismatch(client, tmp_path):
    """Active pickle with mismatched MODEL_INFO hashes reports unavailable."""
    active = _write_resolver_valid_dashboard_bundle(
        tmp_path,
        performance={"optimized_f1_weighted": 0.8975},
    )
    info_path = tmp_path / "MODEL_INFO.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["sha256"] = "0" * 64
    info["version"] = "stale-should-not-surface"
    info["status"] = "production"
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with patch("app.routes.api._get_model_dir", return_value=tmp_path):
        with patch(
            "app.routes.api._resolve_active_production_model_path",
            return_value=active,
        ):
            response = client.get("/api/model-info")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "unavailable"
    assert payload["version"] == "unknown"
    assert payload["f1_score"] is None
    assert payload["provenanceCode"] == "provenance_failed"
    assert "stale-should-not-surface" not in json.dumps(payload)


def test_eval_critical_recall_matches_checked_in_model_info(client):
    """Live endpoint equals the checked-in MODEL_INFO value when present and valid."""
    model_info_path = Path("model") / "MODEL_INFO.json"
    if not model_info_path.is_file():
        pytest.skip("checked-in model/MODEL_INFO.json not available")
    with open(model_info_path, encoding="utf-8") as handle:
        model_info = json.load(handle)
    expected = _safe_optional_prob(
        (model_info.get("performance") or {}).get("eval_critical_recall")
    )
    if expected is None:
        expected = _safe_optional_prob(
            (model_info.get("validation_results") or {}).get("eval_critical_recall")
        )
    if expected is None:
        pytest.skip("checked-in MODEL_INFO has no valid eval_critical_recall")

    response = client.get("/api/model-info/dashboard")
    assert response.status_code == 200
    actual = response.get_json()["metrics"]["evalCriticalRecall"]
    assert actual == pytest.approx(round(expected, 4))
