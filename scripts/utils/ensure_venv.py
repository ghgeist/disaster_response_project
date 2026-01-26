#!/usr/bin/env python
"""
Ensure virtual environment is activated for local development.

This utility is designed to help AI agents in Cursor distinguish between:
- Local development environments (Windows/Linux) - requires venv activation
- Replit environments (SSH or web IDE) - venv not required

Purpose:
    This is a developer experience (DX) quality-of-life improvement for AI-assisted
    development workflows. The original model was ~900 MB and had to be trained
    locally instead of within Replit, necessitating dual-environment support.

Usage:
    Run directly to check venv status:
        python scripts/utils/ensure_venv.py
    
    Or import the function:
        from scripts.utils.ensure_venv import ensure_venv
        ensure_venv()

Note:
    The underlying functionality is also available in disasterproject.utils.env,
    but this script provides a standalone, user-friendly interface with helpful
    error messages for AI agents and developers.
"""
import sys
import os
from pathlib import Path

def ensure_venv():
    """Check if venv is activated, provide helpful error if not."""
    try:
        from disasterproject.utils.env import is_replit, check_venv_activation
    except ImportError:
        # Fallback if package not installed
        def is_replit():
            replit_indicators = ['REPLIT_DB_URL', 'REPL_ID', 'REPL_SLUG', 'REPL_OWNER', 'REPLIT_POD_ID']
            return any(os.getenv(var) is not None for var in replit_indicators)
        
        def check_venv_activation():
            if is_replit():
                return True
            return sys.prefix != sys.base_prefix
    
    if is_replit():
        print("ℹ️ Running in Replit - venv check skipped")
        return True
    
    if not check_venv_activation():
        venv_path = Path('.venv')
        print("❌ Virtual environment not activated!")
        print("\nTo activate:")
        if sys.platform == 'win32':
            print(f"  PowerShell: . {venv_path}\\Scripts\\Activate.ps1")
            print(f"  CMD: {venv_path}\\Scripts\\activate.bat")
        else:
            print(f"  source {venv_path}/bin/activate")
        print("\nOr create venv if it doesn't exist:")
        print(f"  python -m venv {venv_path}")
        sys.exit(1)
    
    print("✓ Virtual environment is active")
    return True

if __name__ == '__main__':
    ensure_venv()
