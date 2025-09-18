"""CSRF smoke test ensuring the form workflow behaves when protection is enabled."""
from __future__ import annotations

import re

import pytest

from app.config import Config
from tests.conftest import create_test_app, skip_if_no_model


class CSRFTestConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = True
    SKIP_ENVIRONMENT_VALIDATION = True


pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def csrf_app():
    skip_if_no_model(
        CSRFTestConfig,
        reason="Model artifact required for CSRF workflow tests is not present.",
    )
    return create_test_app(CSRFTestConfig)


@pytest.fixture
def client(csrf_app):
    with csrf_app.test_client() as client:
        yield client


def extract_csrf_token(html_text: str) -> str:
    """Extract the CSRF token from the rendered HTML."""
    match = re.search(
        r'<input[^>]*name=["\']csrf_token["\'][^>]*value=["\']([^"\']+)',
        html_text,
        re.IGNORECASE,
    )
    assert match, "CSRF token input not found in HTML"
    return match.group(1)


def test_csrf_smoke_home_to_go(client):
    """Ensure we can capture a token from the homepage and submit a guarded form."""
    get_resp = client.get("/")
    assert get_resp.status_code == 200, "Homepage failed while preparing CSRF token"
    token = extract_csrf_token(get_resp.get_data(as_text=True))

    post_resp = client.post(
        "/go",
        data={"csrf_token": token, "query": "Need water and medical aid at 5th street"},
        follow_redirects=False,
        headers={"Referer": "http://localhost/"},
        base_url="http://localhost",
    )

    assert post_resp.status_code in (200, 302), "CSRF protected submission did not complete"
    if post_resp.status_code == 302:
        follow = client.get(post_resp.headers["Location"])
        assert follow.status_code == 200, "Redirect target after CSRF submission was not reachable"
