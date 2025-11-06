"""Core application smoke tests covering the homepage and prediction endpoint."""
from __future__ import annotations

import pytest

from tests.conftest import skip_if_no_model

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("method", "path", "payload", "allowed_statuses"),
    [
        ("get", "/", None, {200}),
        ("post", "/go", {"query": "Need clean water near the river"}, {200, 302, 400}),
    ],
    ids=["index", "go"],
)
def test_core_routes_do_not_error(client, method: str, path: str, payload: dict | None, allowed_statuses: set[int]) -> None:
    """Ensure the primary routes respond without server errors."""
    request = getattr(client, method)
    kwargs = {"follow_redirects": False}
    if payload:
        kwargs["data"] = payload
    response = request(path, **kwargs)
    assert (
        response.status_code in allowed_statuses
    ), f"{method.upper()} {path} returned {response.status_code}, expected one of {sorted(allowed_statuses)}"


def test_homepage_displays_branding(client) -> None:
    """The landing page should surface the Signal Storm brand copy."""
    response = client.get("/")
    assert response.status_code == 200, "Homepage should render successfully"
    assert b"Signal Storm" in response.data, "Homepage is missing the Signal Storm header"


def test_model_can_load_and_predict(client) -> None:
    """Verify the production model can be loaded and used for prediction."""
    from app.config import Config
    from pathlib import Path
    import joblib
    
    skip_if_no_model(Config, reason="Production model required for smoke test")
    
    model_path = Path(Config.MODEL_PATH)
    model = joblib.load(model_path)
    
    # Test that model can make predictions
    test_text = "Need clean water near the river"
    predictions = model.predict([test_text])
    assert predictions is not None, "Model should return predictions"
    assert predictions.shape[0] == 1, "Should return one prediction per input"
