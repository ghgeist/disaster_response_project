"""
Flask application for Disaster Response message classification.
A clean, scalable portfolio project.
"""
from flask import Flask, render_template, request
from flask_wtf.csrf import CSRFProtect, CSRFError

from .config import Config
from .routes import register_routes
from .utils import setup_logging, init_services, validate_environment


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
    
    # Setup logging
    setup_logging(app)
    
    # Validate environment configuration
    validation_results = validate_environment()
    
    # Log validation results
    for info_msg in validation_results.get('info', []):
        app.logger.info(f"Config validation: {info_msg}")
    
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
        return response
    
    app.logger.info('Disaster Response application started')
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )
