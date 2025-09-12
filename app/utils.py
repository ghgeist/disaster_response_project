"""
Utility functions for the Disaster Response application.
"""
import re
import logging
from typing import Optional, Tuple
from pathlib import Path
from flask import Flask

from .services import DataService, ModelService


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
        # Initialize data service
        app.data_service = DataService(app.config['DATABASE_URL'])
        
        # Initialize model service
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


def validate_environment() -> dict:
    """
    Validate environment configuration with comprehensive checks.
    
    Returns:
        Dictionary with validation results including errors, warnings, and status
    """
    import os
    from pathlib import Path
    
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': []
    }
    
    # Check file paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / '02_stg'
    models_dir = base_dir / 'model'  # Use correct directory name from config
    images_dir = base_dir / 'images'
    
    # Check required directories
    if not data_dir.exists():
        validation_results['errors'].append(f"Data directory not found: {data_dir}")
        validation_results['valid'] = False
    else:
        validation_results['info'].append(f"Data directory found: {data_dir}")
    
    if not models_dir.exists():
        validation_results['warnings'].append(f"Models directory not found: {models_dir}")
    else:
        validation_results['info'].append(f"Models directory found: {models_dir}")
    
    if not images_dir.exists():
        validation_results['warnings'].append(f"Images directory not found: {images_dir}")
    else:
        validation_results['info'].append(f"Images directory found: {images_dir}")
    
    # Check database file
    db_file = data_dir / 'stg_disaster_response.db'
    if not db_file.exists():
        validation_results['errors'].append(f"Database file not found: {db_file}")
        validation_results['valid'] = False
    else:
        validation_results['info'].append(f"Database file found: {db_file}")
    
    # Check model file - use the configured model filename
    from .config import Config
    model_file = Config.MODEL_PATH
    if not model_file.exists():
        validation_results['warnings'].append(f"Model file not found: {model_file}")
        # Check if Google Drive ID is configured for download
        gdrive_id = os.environ.get('GDRIVE_MODEL_ID')
        if not gdrive_id or gdrive_id.strip() in {'', 'YOUR_FILE_ID', 'YOUR_GOOGLE_DRIVE_FILE_ID'}:
            validation_results['errors'].append("Model file not found and GDRIVE_MODEL_ID not configured")
            validation_results['valid'] = False
        else:
            validation_results['info'].append("Model file not found locally, but GDRIVE_MODEL_ID is configured for download")
    else:
        validation_results['info'].append(f"Model file found: {model_file}")
    
    # Check environment variables
    secret_key = os.environ.get('SECRET_KEY')
    if not secret_key or secret_key == 'dev-secret-key-change-in-production':
        validation_results['warnings'].append("Using default SECRET_KEY - change in production")
    
    # Check log file directory
    log_file = base_dir / 'app.log'
    log_dir = log_file.parent
    if not log_dir.exists():
        validation_results['warnings'].append(f"Log directory not found: {log_dir}")
    else:
        validation_results['info'].append(f"Log directory found: {log_dir}")
    
    return validation_results
