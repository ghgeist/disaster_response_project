"""
Route blueprints for the Disaster Response application.
"""
from flask import Flask
from .home import home_bp
from .classification import classification_bp
from .health import health_bp


def register_routes(app: Flask):
    """Register all route blueprints with the app."""
    app.register_blueprint(home_bp)
    app.register_blueprint(classification_bp)
    app.register_blueprint(health_bp)
