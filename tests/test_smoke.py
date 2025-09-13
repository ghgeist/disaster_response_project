import pytest
from app.app import create_app
from app.config import TestConfig


@pytest.fixture
def client():
    app = create_app(TestConfig)
    return app.test_client()


def test_home_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Signal Storm" in resp.data


def test_results_renders(client):
    resp = client.post("/go", data={"query": "water"})
    assert resp.status_code == 200
    assert b"Analysis Complete" in resp.data
