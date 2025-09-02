"""
Utility functions for the Disaster Response application.
"""
import re
import logging
from typing import Optional, Tuple
from pathlib import Path
from flask import Flask

from app.services import DataService, ModelService


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
    Validate environment configuration.
    
    Returns:
        Dictionary with validation results
    """
    import os
    from pathlib import Path
    
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check file paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / '02_stg'
    models_dir = base_dir / 'models'
    
    if not data_dir.exists():
        validation_results['errors'].append(f"Data directory not found: {data_dir}")
        validation_results['valid'] = False
    
    if not models_dir.exists():
        validation_results['warnings'].append(f"Models directory not found: {models_dir}")
    
    # Check database file
    db_file = data_dir / 'stg_disaster_response.db'
    if not db_file.exists():
        validation_results['errors'].append(f"Database file not found: {db_file}")
        validation_results['valid'] = False
    
    return validation_results
