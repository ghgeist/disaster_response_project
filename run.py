#!/usr/bin/env python3
"""
Entry point for running the Disaster Response Flask application.
Run this script from the workspace root directory.
"""
import os
from app.app import create_app

if __name__ == '__main__':
    # Set environment to development for auto-reloading and debugging
    os.environ['FLASK_ENV'] = 'development'
    
    app = create_app()
    
    # Use Flask's built-in debug mode, which is more robust
    app.run(
        host=app.config.get('HOST', '0.0.0.0'),
        port=app.config.get('PORT', 3001),
        debug=True  # Enables auto-reloader and debugger
    )
