#!/usr/bin/env python3
"""
Entry point for running the Disaster Response Flask application.
Run this script from the workspace root directory.
"""
import os
from app.app import create_app

if __name__ == '__main__':
    app = create_app()

    # Replit Autoscale requires binding to 0.0.0.0:$PORT
    # Priority: $PORT env var (for Replit) > config > default
    port = int(os.environ.get('PORT', app.config.get('PORT', 5000)))
    
    app.run(
        host=app.config.get('HOST', '0.0.0.0'),
        port=port,
        debug=app.config.get('DEBUG', False)
    )
