#!/usr/bin/env python3
"""
Entry point for running the Disaster Response Flask application.
Run this script from the workspace root directory.
"""
from app.app import create_app

if __name__ == '__main__':
    app = create_app()
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )
