"""Additional smoke tests focused on application startup and happy-path predictions."""
from __future__ import annotations

import pytest

from tests.conftest import create_test_app
from app.config import TestConfig

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
