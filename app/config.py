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
    
    # Application settings
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 3000))
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data' / '02_stg'
    MODELS_DIR = BASE_DIR / 'model'  # Fixed: use 'model' not 'models'
    IMAGES_DIR = BASE_DIR / 'images'
    
    # Database settings
    DATABASE_PATH = DATA_DIR / 'stg_disaster_response.db'
    DATABASE_URL = f'sqlite:///{DATABASE_PATH}'
    
    # Model settings
    MODEL_FILENAME = 'classifier.pkl'
    MODEL_PATH = MODELS_DIR / MODEL_FILENAME
    GDRIVE_MODEL_ID = os.environ.get('GDRIVE_MODEL_ID')
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO' if not DEBUG else 'DEBUG')
    LOG_FILE = BASE_DIR / 'app.log'
