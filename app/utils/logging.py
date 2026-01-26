"""
Logging and service initialization utilities.
"""
import logging
import threading
from flask import Flask, has_request_context

from app.services.data_service import DataService
from app.services.model_service import ModelService
from .mocks import MockDataService, MockModelService

# Module-level logging state to prevent duplicate handlers and startup messages
_logging_lock = threading.Lock()
_logging_configured = False
_logging_startup_logged = False

# Module-level service initialization tracking
_services_initialized = False


def setup_logging(app: Flask) -> None:
    """Setup application logging."""
    global _logging_configured, _logging_startup_logged
    
    # Check if logging has already been configured in this process (thread-safe)
    with _logging_lock:
        if _logging_configured:
            # Already configured, just set the level for this app instance
            app.logger.setLevel(getattr(logging, app.config['LOG_LEVEL']))
            return
        
        # Mark as configured before proceeding to avoid race conditions
        _logging_configured = True
    
    # Perform logging configuration outside the lock to avoid holding it during I/O
    if not app.debug:
        # Check if file handler already exists to prevent duplicates
        log_file = app.config['LOG_FILE']
        log_file_path = str(log_file.resolve())
        existing_file_handlers = [
            h for h in app.logger.handlers
            if isinstance(h, logging.FileHandler) and h.baseFilename == log_file_path
        ]
        
        if not existing_file_handlers:
            # Create logs directory if it doesn't exist
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Configure file logging
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            file_handler.setLevel(getattr(logging, app.config['LOG_LEVEL']))
            app.logger.addHandler(file_handler)
        
        app.logger.setLevel(getattr(logging, app.config['LOG_LEVEL']))
    
    # Log startup message only once per process (thread-safe)
    with _logging_lock:
        if not _logging_startup_logged:
            app.logger.info('Disaster Response application startup')
            _logging_startup_logged = True


def init_services(app: Flask) -> None:
    """Initialize application services."""
    global _services_initialized
    
    # Check if services have already been initialized in this process
    if _services_initialized:
        # Reuse existing services if they exist on a previous app instance
        # For new app instances, we still need to attach services
        if not hasattr(app, 'data_service') or not hasattr(app, 'model_service'):
            try:
                if app.config.get('TESTING'):
                    app.data_service = MockDataService(app.config['DATABASE_URL'])
                    app.model_service = MockModelService()
                else:
                    app.data_service = DataService(app.config['DATABASE_URL'])
                    app.model_service = ModelService(
                        model_path=app.config['MODEL_PATH'],
                        gdrive_model_id=app.config['GDRIVE_MODEL_ID']
                    )
            except Exception as e:
                app.logger.error('Failed to initialize services: %s', e)
                raise
        return
    
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

        _services_initialized = True
        app.logger.info('Services initialized successfully')

    except Exception as e:
        app.logger.error('Failed to initialize services: %s', e)
        raise
