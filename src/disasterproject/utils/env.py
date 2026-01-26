"""Environment detection utilities."""
import os
import sys
from pathlib import Path


def is_replit() -> bool:
    """Check if running in Replit environment (web IDE or SSH).
    
    Detects Replit by checking for Replit-specific environment variables that are
    set in both the web IDE and SSH environments.
    
    Returns:
        True if running in Replit (web or SSH), False otherwise.
    """
    # Check multiple Replit environment variables for robust detection
    # These are set in both web IDE and SSH environments
    replit_indicators = [
        'REPLIT_DB_URL',  # Replit database URL (most reliable)
        'REPL_ID',        # Replit ID
        'REPL_SLUG',      # Replit slug/name
        'REPL_OWNER',     # Replit owner
        'REPLIT_POD_ID',  # Replit pod ID
    ]
    
    return any(os.getenv(var) is not None for var in replit_indicators)


def is_venv_required() -> bool:
    """Check if virtual environment is required for current environment.
    
    Returns:
        True if venv is required (local development), False if not (Replit).
    """
    return not is_replit()


def check_venv_activation() -> bool:
    """Check if virtual environment is activated.
    
    Returns:
        True if venv is active or not required, False if venv is required but not active.
    """
    if not is_venv_required():
        return True  # Venv not required in Replit
    
    # Check if venv is activated
    return sys.prefix != sys.base_prefix


def get_venv_status() -> dict:
    """Get comprehensive venv status information.
    
    Returns:
        Dictionary with environment and venv status information.
    """
    return {
        'is_replit': is_replit(),
        'venv_required': is_venv_required(),
        'venv_active': check_venv_activation(),
        'python_prefix': sys.prefix,
        'python_base_prefix': sys.base_prefix,
    }
