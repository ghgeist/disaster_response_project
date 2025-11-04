"""
Flask application for Disaster Response message classification.
A clean, scalable portfolio project.
"""
import threading
from flask import Flask, render_template, request
from flask_wtf.csrf import CSRFProtect, CSRFError

from .config import Config
from .routes import register_routes
from .utils import setup_logging, init_services, validate_environment
from .nltk_setup import setup_nltk_resources, NLTKSetupError

# Module-level initialization tracking to prevent duplicate startup messages
# across multiple app instances in the same process
_app_initialization_lock = threading.Lock()
_app_initialization_count = 0


def create_app(config_class=Config):
    """
    Create and configure the Flask application.
    
    Args:
        config_class: Configuration class to use
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Use module-level tracking to prevent duplicate logging of expensive operations
    # This handles cases where create_app() is called multiple times in the same process
    global _app_initialization_count
    
    # Setup logging first (this will check its own module-level state)
    setup_logging(app)
    
    # Track if this is the first initialization in this process (thread-safe)
    with _app_initialization_lock:
        is_first_init = (_app_initialization_count == 0)
        if is_first_init:
            _app_initialization_count += 1
    
    # Setup NLTK resources at startup for performance optimization
    # Check if NLTK has already been set up (module-level cache)
    if not hasattr(setup_nltk_resources, '_setup_completed'):
        try:
            # Only log "Setting up..." on first initialization
            if is_first_init:
                app.logger.info("Setting up NLTK resources...")
            nltk_setup_results = setup_nltk_resources()
            
            # Only log NLTK setup results on first initialization
            if is_first_init:
                if nltk_setup_results["success"]:
                    app.logger.info(f"NLTK setup completed successfully in {nltk_setup_results['setup_time_ms']}ms")
                    app.logger.info(f"Loaded resources: {[r['name'] for r in nltk_setup_results['resources_loaded']]}")
                else:
                    app.logger.warning(f"NLTK setup completed with warnings in {nltk_setup_results['setup_time_ms']}ms")
                    for error in nltk_setup_results["errors"]:
                        app.logger.warning(f"NLTK setup warning: {error}")
            
            # Mark NLTK setup as completed
            setup_nltk_resources._setup_completed = True
            setup_nltk_resources._setup_results = nltk_setup_results
            
        except NLTKSetupError as e:
            # Always log errors, but only log detailed messages on first init
            app.logger.error(f"Critical NLTK setup failure: {e}")
            if is_first_init:
                app.logger.error("Application will continue but may experience performance issues")
            nltk_setup_results = {
                "success": False,
                "error": str(e),
                "setup_time_ms": 0
            }
            setup_nltk_resources._setup_completed = True
            setup_nltk_resources._setup_results = nltk_setup_results
        except Exception as e:
            app.logger.error(f"Unexpected error during NLTK setup: {e}")
            if is_first_init:
                app.logger.error("Application will continue but may experience performance issues")
            nltk_setup_results = {
                "success": False,
                "error": str(e),
                "setup_time_ms": 0
            }
            setup_nltk_resources._setup_completed = True
            setup_nltk_resources._setup_results = nltk_setup_results
    else:
        # NLTK already set up, use cached results
        nltk_setup_results = getattr(setup_nltk_resources, '_setup_results', {})
        app.logger.debug("NLTK resources already configured (using cached setup)")
    
    # Store NLTK setup results in app config for monitoring
    app.config['NLTK_SETUP_RESULTS'] = nltk_setup_results
    
    # Validate environment configuration
    validation_results = validate_environment(config_class)
    
    # Log validation results only on first initialization to prevent duplicate logs
    if is_first_init:
        # Log validation results - consolidate info messages in production, detailed in debug
        if app.debug:
            # In debug mode, log all validation details
            for info_msg in validation_results.get('info', []):
                app.logger.info(f"Config validation: {info_msg}")
        else:
            # In production, log a summary unless there are issues
            if validation_results['info']:
                info_count = len(validation_results['info'])
                app.logger.info(f"Config validation: {info_count} checks passed")
                # Log summary of what was validated
                key_validations = [msg for msg in validation_results['info'] if any(
                    key in msg.lower() for key in ['directory', 'file', 'database', 'model']
                )]
                if key_validations:
                    app.logger.debug(f"Validation details: {', '.join(key_validations[:3])}")
        
        for warning_msg in validation_results.get('warnings', []):
            app.logger.warning(f"Config validation: {warning_msg}")
        
        for error_msg in validation_results.get('errors', []):
            app.logger.error(f"Config validation: {error_msg}")
    else:
        # On subsequent initializations, only log errors/warnings
        for warning_msg in validation_results.get('warnings', []):
            app.logger.warning(f"Config validation: {warning_msg}")
        
        for error_msg in validation_results.get('errors', []):
            app.logger.error(f"Config validation: {error_msg}")
    
    # Check if validation failed
    if not validation_results['valid']:
        error_summary = "; ".join(validation_results['errors'])
        app.logger.critical(f"Configuration validation failed: {error_summary}")
        raise RuntimeError(f"Application configuration is invalid: {error_summary}")
    
    # Initialize Flask-WTF CSRF protection
    CSRFProtect(app)

    # Ensure session is initialized for CSRF token support
    @app.before_request
    def ensure_session():
        """Ensure session is initialized for CSRF token support."""
        # Only initialize session for routes that render forms or submit forms (need CSRF)
        if request.endpoint in ('index', 'go', 'classify'):
            from flask import session
            # Touch session to initialize it (Flask sessions are lazy)
            session.permanent = True
            if 'init' not in session:
                session['init'] = True

    # CSRF error handler for better diagnostics and UX
    @app.errorhandler(CSRFError)
    def handle_csrf_error(e):
        # Log detailed CSRF failure context
        reason = getattr(e, 'reason', 'unknown')
        description = getattr(e, 'description', str(e))
        has_session_cookie = 'session' in request.cookies
        app.logger.warning(
            "CSRF error: %s; reason=%s; method=%s; path=%s; referrer=%s; origin=%s; has_session_cookie=%s",
            description,
            reason,
            request.method,
            request.path,
            request.referrer,
            request.headers.get('Origin'),
            has_session_cookie,
        )
        return render_template('error.html', message="Your session expired or the form is invalid. Please refresh and try again."), 400
    
    # Initialize services
    init_services(app)
    
    # Register routes
    register_routes(app)
    
    # Add security headers
    @app.after_request
    def add_security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        # Log if session cookie is being set for GET '/' to aid CSRF debugging
        try:
            if request.method == 'GET' and request.path in {'/', '/index'}:
                set_cookie_headers = response.headers.getlist('Set-Cookie')
                # Determine if a session cookie is being set, without logging the raw value
                has_session_set = any(h.lower().startswith('session=') for h in set_cookie_headers)
                # Redact any session cookie value from logs while keeping attributes
                def _redact_session_cookie(header_value: str) -> str:
                    lower = header_value.lower()
                    if lower.startswith('session='):
                        parts = header_value.split(';', 1)
                        # Replace the cookie value with a placeholder but preserve attributes
                        redacted_prefix = 'session=<redacted>'
                        return redacted_prefix + (';' + parts[1] if len(parts) > 1 else '')
                    return header_value

                sanitized_headers = [_redact_session_cookie(h) for h in set_cookie_headers]
                app.logger.debug(
                    "Session Set-Cookie on GET %s: %s | headers=%s",
                    request.path,
                    has_session_set,
                    [h for h in sanitized_headers if h.lower().startswith('session=')]
                )
        except Exception:
            # Non-fatal logging aid
            pass
        return response
    
    # Mark as initialized and log startup completion only on first initialization
    app._initialized = True
    if is_first_init:
        app.logger.info('Disaster Response application started')
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )
