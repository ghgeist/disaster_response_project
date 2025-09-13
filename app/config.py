"""
Configuration for the Disaster Response Flask application.
"""
import os
from pathlib import Path


class Config:
    """Application configuration."""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_ENV') == 'development'
    
    # Flask-WTF settings
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = 3600  # 1 hour
    # Optional: disable CSRF expiry for long demos (dev only)
    if os.environ.get('CSRF_TIME_LIMIT_NONE') == '1':
        WTF_CSRF_TIME_LIMIT = None
    
    # Application settings
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data' / '02_stg'
    MODELS_DIR = BASE_DIR / 'model'  # Fixed: use 'model' not 'models'
    IMAGES_DIR = BASE_DIR / 'images'
    
    # Database settings
    DATABASE_PATH = DATA_DIR / 'stg_disaster_response.db'
    DATABASE_URL = f'sqlite:///{DATABASE_PATH}'
    
    # Model settings
    MODEL_FILENAME = 'disaster_rf_v1-2-0_prod_2025-09-11.pkl'
    MODEL_PATH = MODELS_DIR / MODEL_FILENAME
    
    # Google Drive model configuration
    # Production: REQUIRED - Model downloaded from Google Drive
    # Development: OPTIONAL - Falls back to local model if not set
    GDRIVE_MODEL_ID = os.environ.get('GDRIVE_MODEL_ID')
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO' if not DEBUG else 'DEBUG')
    LOG_FILE = BASE_DIR / 'app.log'

    # Session / cookie settings
    # Allow third-party/embedded contexts (e.g., preview iframes) to receive cookies
    # Controlled via ALLOW_THIRD_PARTY_COOKIES env (default on for dev/demos)
    ALLOW_THIRD_PARTY_COOKIES = os.environ.get('ALLOW_THIRD_PARTY_COOKIES', '1') == '1'
    if ALLOW_THIRD_PARTY_COOKIES:
        SESSION_COOKIE_SAMESITE = 'None'
        SESSION_COOKIE_SECURE = True
    else:
        SESSION_COOKIE_SAMESITE = os.environ.get('SESSION_COOKIE_SAMESITE', 'Lax')
        SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'False').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = True
    PREFERRED_URL_SCHEME = 'https'


class TestConfig(Config):
    """Test-specific configuration that bypasses environment validation."""

    # Enable testing mode
    TESTING = True
    WTF_CSRF_ENABLED = False

    # Override environment validation to always pass
    SKIP_ENVIRONMENT_VALIDATION = True

    # Use in-memory database for testing
    DATABASE_URL = 'sqlite:///:memory:'

    # Mock model settings - these won't be validated
    MODEL_FILENAME = 'test_model.pkl'
    GDRIVE_MODEL_ID = None
