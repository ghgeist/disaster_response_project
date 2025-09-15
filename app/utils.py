"""
Utility functions for the Disaster Response application.
"""
import re
import logging
from typing import Optional, Tuple
from pathlib import Path
from flask import Flask

from .services import DataService, ModelService


class MockDataService:
    """Mock data service for testing."""

    def __init__(self, database_url: str = None):
        self.database_url = database_url
        self._df = None

    def load_data(self, table_name: str = 'stg_disaster_response'):
        """Mock data loading."""
        import pandas as pd

        # Create a comprehensive mock dataframe with all expected columns
        self._df = pd.DataFrame({
            'id': [1, 2, 3],
            'message': ['Need help with water', 'Offering food supplies', 'Road blocked'],
            'original': ['Need help with water', 'Offering food supplies', 'Road blocked'],
            'genre': ['direct', 'direct', 'news'],
            'related': [1, 1, 1],
            'request': [1, 0, 0],
            'offer': [0, 1, 0],
            'aid_related': [1, 1, 0],
            'medical_help': [0, 0, 0],
            'medical_products': [0, 0, 0],
            'search_and_rescue': [0, 0, 0],
            'security': [0, 0, 0],
            'military': [0, 0, 0],
            'child_alone': [0, 0, 0],
            'water': [1, 0, 0],
            'food': [0, 1, 0],
            'shelter': [0, 0, 0],
            'clothing': [0, 0, 0],
            'money': [0, 0, 0],
            'missing_people': [0, 0, 0],
            'refugees': [0, 0, 0],
            'death': [0, 0, 0],
            'other_aid': [0, 0, 0],
            'infrastructure_related': [0, 0, 1],
            'transport': [0, 0, 1],
            'buildings': [0, 0, 0],
            'electricity': [0, 0, 0],
            'tools': [0, 0, 0],
            'hospitals': [0, 0, 0],
            'shops': [0, 0, 0],
            'aid_centers': [0, 0, 0],
            'other_infrastructure': [0, 0, 0],
            'weather_related': [0, 0, 0],
            'floods': [0, 0, 0],
            'storm': [0, 0, 0],
            'fire': [0, 0, 0],
            'earthquake': [0, 0, 0],
            'cold': [0, 0, 0],
            'other_weather': [0, 0, 0],
            'direct_report': [1, 0, 1]
        })
        return self._df

    def get_data(self):
        """Get the mock data."""
        if self._df is None:
            self.load_data()
        return self._df

    def get_category_columns(self):
        """Get mock category columns."""
        df = self.get_data()
        return df.columns[4:].tolist()


class MockModelService:
    """Mock model service for testing."""

    def __init__(self):
        self._loaded = False

    def load_model(self):
        """Mock model loading."""
        self._loaded = True
        return self

    def predict(self, text: str) -> dict:
        """Mock prediction that returns sample data."""
        # Return a sample prediction with some categories marked as positive
        categories = [
            'related', 'request', 'offer', 'aid_related', 'medical_help',
            'medical_products', 'search_and_rescue', 'security', 'military',
            'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
            'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related',
            'transport', 'buildings', 'electricity', 'tools', 'hospitals',
            'shops', 'aid_centers', 'other_infrastructure', 'weather_related',
            'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather',
            'direct_report'
        ]

        # Mock some positive predictions based on keywords
        predictions = {}
        text_lower = text.lower()

        for category in categories:
            # Simple keyword-based mock predictions
            if category == 'related':
                # Only mark as related if message contains disaster-related keywords
                disaster_keywords = ['help', 'emergency', 'disaster', 'flood', 'fire', 'earthquake', 'storm', 'medical', 'water', 'food', 'shelter']
                predictions[category] = 1 if any(keyword in text_lower for keyword in disaster_keywords) else 0
            elif category == 'water' and 'water' in text_lower:
                predictions[category] = 1
            elif category == 'food' and 'food' in text_lower:
                predictions[category] = 1
            elif category == 'medical_help' and ('medical' in text_lower or 'help' in text_lower):
                predictions[category] = 1
            else:
                predictions[category] = 0

        return predictions


def setup_logging(app: Flask) -> None:
    """Setup application logging."""
    if not app.debug:
        # Create logs directory if it doesn't exist
        log_file = app.config['LOG_FILE']
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure file logging
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(getattr(logging, app.config['LOG_LEVEL']))
        app.logger.addHandler(file_handler)
        app.logger.setLevel(getattr(logging, app.config['LOG_LEVEL']))
        app.logger.info('Disaster Response application startup')


def init_services(app: Flask) -> None:
    """Initialize application services."""
    try:
        # Initialize services (use mocks for testing)
        if app.config.get('TESTING'):
            app.data_service = MockDataService(app.config['DATABASE_URL'])
            app.model_service = MockModelService()
        else:
            app.data_service = DataService(app.config['DATABASE_URL'])
            app.model_service = ModelService(
                model_path=app.config['MODEL_PATH'],
                gdrive_model_id=app.config['GDRIVE_MODEL_ID']
            )

        app.logger.info('Services initialized successfully')

    except Exception as e:
        app.logger.error(f'Failed to initialize services: {e}')
        raise


def validate_message_input(text: str) -> Tuple[bool, Optional[str]]:
    """
    Validate user input message.
    
    Args:
        text: Input text to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Message cannot be empty"
    
    if len(text.strip()) < 3:
        return False, "Message must be at least 3 characters long"
    
    if len(text) > 1000:
        return False, "Message cannot exceed 1000 characters"
    
    # Check for potentially harmful content (basic check)
    if re.search(r'<script|javascript:|data:', text.lower()):
        return False, "Message contains potentially harmful content"
    
    # Check for SQL injection patterns
    sql_patterns = [
        r'union\s+select', r'drop\s+table', r'delete\s+from',
        r'insert\s+into', r'update\s+set', r'exec\s*\('
    ]
    for pattern in sql_patterns:
        if re.search(pattern, text.lower()):
            logging.getLogger(__name__).warning(f"Potential SQL injection attempt detected: {text[:50]}...")
            return False, "Message contains potentially harmful content"
    
    return True, None


def sanitize_input(text: str) -> str:
    """
    Sanitize user input by removing potentially harmful characters.
    
    Args:
        text: Input text to sanitize
        
    Returns:
        Sanitized text
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove null bytes and control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    return text


def validate_environment(config_class=None) -> dict:
    """
    Validate environment configuration with comprehensive checks.

    Args:
        config_class: Configuration class to use for validation

    Returns:
        Dictionary with validation results including errors, warnings, and status
    """
    import os
    from .config import Config

    # Use provided config class or default to Config
    config = config_class or Config

    # Skip validation if configured to do so
    if hasattr(config, 'SKIP_ENVIRONMENT_VALIDATION') and config.SKIP_ENVIRONMENT_VALIDATION:
        return {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': ['Environment validation skipped for testing']
        }

    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': []
    }
    
    # Check required directories using config
    if not config.DATA_DIR.exists():
        validation_results['errors'].append(f"Data directory not found: {config.DATA_DIR}")
        validation_results['valid'] = False
    else:
        validation_results['info'].append(f"Data directory found: {config.DATA_DIR}")

    if not config.MODELS_DIR.exists():
        validation_results['warnings'].append(f"Models directory not found: {config.MODELS_DIR}")
    else:
        validation_results['info'].append(f"Models directory found: {config.MODELS_DIR}")

    if not config.IMAGES_DIR.exists():
        validation_results['warnings'].append(f"Images directory not found: {config.IMAGES_DIR}")
    else:
        validation_results['info'].append(f"Images directory found: {config.IMAGES_DIR}")

    # Check database file using config
    if hasattr(config, 'DATABASE_PATH') and not config.DATABASE_PATH.exists():
        validation_results['errors'].append(f"Database file not found: {config.DATABASE_PATH}")
        validation_results['valid'] = False
    elif hasattr(config, 'DATABASE_PATH'):
        validation_results['info'].append(f"Database file found: {config.DATABASE_PATH}")

    # Check model file using config
    if not config.MODEL_PATH.exists():
        validation_results['warnings'].append(f"Model file not found: {config.MODEL_PATH}")
        # Check if Google Drive ID is configured for download
        if not config.GDRIVE_MODEL_ID or config.GDRIVE_MODEL_ID.strip() in {'', 'YOUR_FILE_ID', 'YOUR_GOOGLE_DRIVE_FILE_ID'}:
            validation_results['errors'].append("Model file not found and GDRIVE_MODEL_ID not configured")
            validation_results['valid'] = False
        else:
            validation_results['info'].append("Model file not found locally, but GDRIVE_MODEL_ID is configured for download")
    else:
        validation_results['info'].append(f"Model file found: {config.MODEL_PATH}")

    # Check environment variables using config
    if config.SECRET_KEY == 'dev-secret-key-change-in-production':
        validation_results['warnings'].append("Using default SECRET_KEY - change in production")

    # Check log file directory using config
    log_dir = config.LOG_FILE.parent
    if not log_dir.exists():
        validation_results['warnings'].append(f"Log directory not found: {log_dir}")
    else:
        validation_results['info'].append(f"Log directory found: {log_dir}")
    
    return validation_results
