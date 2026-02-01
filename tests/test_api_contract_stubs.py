"""Contract smoke tests for stubbed dashboard API endpoints."""

from app.routes.api import _row_to_feed_item, _safe_label_value, _simulated_probabilities


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
    assert 0.7 <= result["water"] <= 0.8
    assert 0.1 <= result["food"] <= 0.2


def test_row_to_feed_item_handles_nan_message_genre():
    row = {"id": 7, "message": float("nan"), "genre": float("nan")}
    item = _row_to_feed_item(row, [])
    assert item["content"] == ""
    assert item["source"] == "Direct Report"
