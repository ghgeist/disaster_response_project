"""Contract smoke tests for stubbed dashboard API endpoints."""


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
