#!/usr/bin/env python3
"""
Entry point for running the Disaster Response Flask application.
Run this script from the workspace root directory.
"""
import os
from pathlib import Path

from app.app import create_app


def check_dashboard_build():
    """
    Optionally check if dashboard build is newer than deployed files.
    
    Enable by setting AUTO_BUILD_DASHBOARD_CHECK=1 environment variable.
    This will warn (but not auto-build) if the React build is out of sync.
    """
    if os.environ.get('AUTO_BUILD_DASHBOARD_CHECK') != '1':
        return

    dashboard_dir = Path('app/static/dashboard')
    vendor_dist = Path('_vendor/figma_make/dist')

    if not vendor_dist.exists() or not dashboard_dir.exists():
        return

    try:
        # Get most recent file modification time in dist/
        dist_files = [f for f in vendor_dist.rglob('*') if f.is_file()]
        dashboard_files = [f for f in dashboard_dir.rglob('*') if f.is_file()]

        if not dist_files or not dashboard_files:
            return

        dist_mtime = max(f.stat().st_mtime for f in dist_files)
        dashboard_mtime = max(f.stat().st_mtime for f in dashboard_files)

        if dist_mtime > dashboard_mtime:
            print("\n⚠️  Dashboard build is newer than deployed files!")
            print("   Run: python scripts/build_dashboard.py")
            print("   Or enable auto-build: AUTO_BUILD_DASHBOARD=1 python run.py\n")
    except Exception:
        # Silently fail if check encounters any issues
        pass


if __name__ == '__main__':
    check_dashboard_build()
    app = create_app()

    # Replit Autoscale requires binding to 0.0.0.0:$PORT
    # Priority: $PORT env var (for Replit) > config > default
    port = int(os.environ.get('PORT', app.config.get('PORT', 5000)))

    app.run(
        host=app.config.get('HOST', '0.0.0.0'),
        port=port,
        debug=app.config.get('DEBUG', False)
    )
