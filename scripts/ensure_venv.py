#!/usr/bin/env python
"""Ensure virtual environment is activated for local development."""
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
