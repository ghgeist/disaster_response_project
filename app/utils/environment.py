"""
Environment validation utilities.
"""
from app.config import Config


def validate_environment(config_class=None) -> dict:
    """
    Validate environment configuration with comprehensive checks.

    Args:
        config_class: Configuration class to use for validation

    Returns:
        Dictionary with validation results including errors, warnings, and status
    """
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
