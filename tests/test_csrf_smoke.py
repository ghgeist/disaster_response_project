import re
import pytest

from app.app import create_app
from app.config import Config


class CSRFTestConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = True
    SKIP_ENVIRONMENT_VALIDATION = True


@pytest.fixture
def app():
    app = create_app(CSRFTestConfig)
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def extract_csrf_token(html_text: str) -> str:
    # Match csrf_token value regardless of attribute order
    match = re.search(r'<input[^>]*name=["\']csrf_token["\'][^>]*value=["\']([^"\']+)', html_text, re.IGNORECASE)
    assert match, "CSRF token input not found in HTML"
    return match.group(1)


def test_csrf_smoke_home_to_go(client):
    # GET home and ensure csrf_token is present
    get_resp = client.get('/')
    assert get_resp.status_code == 200
    html = get_resp.get_data(as_text=True)
    token = extract_csrf_token(html)

    # Submit POST to /go with token and minimal valid message
    form_data = {
        'csrf_token': token,
        'query': 'Need water and medical aid at 5th street'
    }
    post_resp = client.post(
        '/go',
        data=form_data,
        follow_redirects=False,
        headers={'Referer': 'http://localhost/'},
        base_url='http://localhost'
    )

    # Accept either 200 (render results) or 302 redirect back to index on errors
    assert post_resp.status_code in (200, 302)

    # If redirected, follow and expect 200
    if post_resp.status_code == 302:
        follow = client.get(post_resp.headers['Location'])
        assert follow.status_code == 200
