"""Additional smoke tests focused on application startup and happy-path predictions."""
from __future__ import annotations

import pytest

from app.config import TestConfig
from tests.conftest import create_test_app

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def smoke_app():
    """Provide a dedicated application instance for smoke assertions."""
    return create_test_app(TestConfig)


@pytest.fixture
def smoke_client(smoke_app):
    with smoke_app.test_client() as client:
        yield client


def test_app_startup_uses_testing_config(smoke_app) -> None:
    """The smoke app should be configured for deterministic testing."""
    assert smoke_app is not None
    assert smoke_app.testing is True, "Flask TESTING flag should be enabled for smoke tests"


def test_prediction_flow_returns_completion_banner(smoke_client) -> None:
    """Posting a message should eventually show the analysis complete banner."""
    response = smoke_client.post(
        "/go",
        data={"query": "Need water and shelter"},
        follow_redirects=True,
    )
    assert response.status_code == 200, "Prediction flow should render without redirect loops"
    assert b"Analysis Complete" in response.data, "Results page missing completion banner"


def test_dashboard_spa_returns_200_and_references_static_assets(smoke_client) -> None:
    """Dashboard SPA route returns HTTP 200 and HTML that references static assets."""
    response = smoke_client.get("/api/dashboard")
    assert response.status_code == 200, "GET /api/dashboard should return 200"
    assert response.content_type and "text/html" in response.content_type
    body = response.data.decode("utf-8")
    assert "/static/dashboard/" in body, "Dashboard HTML must reference static assets (SPA entry)"


def test_model_info_dashboard_spa_returns_200(smoke_client) -> None:
    """Model Information SPA route returns HTTP 200 and serves the same SPA shell."""
    response = smoke_client.get("/api/model-info-dashboard")
    assert response.status_code == 200, "GET /api/model-info-dashboard should return 200"
    assert response.content_type and "text/html" in response.content_type
    body = response.data.decode("utf-8")
    assert "/static/dashboard/" in body, "Model info dashboard must reference same static assets"


def test_model_info_dashboard_api_returns_200_and_valid_json(smoke_client) -> None:
    """GET /api/model-info/dashboard returns HTTP 200 and valid JSON with expected keys."""
    response = smoke_client.get("/api/model-info/dashboard")
    assert response.status_code == 200, "GET /api/model-info/dashboard (API) should return 200"
    payload = response.get_json()
    assert payload is not None, "Response must be valid JSON"
    for key in ("model", "metrics", "categories", "criticalThresholds", "registry"):
        assert key in payload, f"Dashboard API payload must include key: {key}"
