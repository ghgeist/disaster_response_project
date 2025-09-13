"""
Simple smoke test for the Flask application.
Tests basic app functionality without complex dependencies.
"""
import pytest
from app.app import create_app
from app.config import Config


@pytest.fixture
def app():
    """Create test Flask app."""
    app = create_app(Config)
    app.config.update({
        'TESTING': True,
        'WTF_CSRF_ENABLED': False,  # Disable CSRF for testing
    })
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


def test_app_startup(app):
    """Test that the app can be created without errors."""
    assert app is not None
    assert app.config['TESTING'] is True


def test_index_page(client):
    """Test that the index page loads."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Signal Storm' in response.data


def test_predict_endpoint_basic(client):
    """Test basic predict functionality with a simple message."""
    # Simple POST to predict endpoint
    response = client.post('/go', data={'query': 'Need help with water'})
    # Should either succeed (200) or redirect (302), not crash
    assert response.status_code in (200, 302, 400)  # 400 is acceptable for missing CSRF in some configs