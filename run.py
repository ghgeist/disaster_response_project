#!/usr/bin/env python3
"""
Entry point for running the Disaster Response Flask application.
Run this script from the workspace root directory.
"""
import os
from app.app import create_app

if __name__ == '__main__':
    app = create_app()

    app.run(
        host=app.config.get('HOST', '0.0.0.0'),
        port=app.config.get('PORT', 3001),
        debug=app.config.get('DEBUG', False)
    )
