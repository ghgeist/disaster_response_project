"""
Configuration for the Disaster Response Flask application.
"""
import os
from pathlib import Path


def _discover_latest_model(models_dir: Path) -> str:
    """Discover the latest production model file in the models directory."""
    if not models_dir.exists():
        # Fallback to explicit filename if directory doesn't exist yet
        return 'disaster_rf_v25-09-16_prod_2025-09-19.pkl'
    
    # Find all production model files matching the pattern
    pattern = 'disaster_rf_v*_prod_*.pkl'
    model_files = list(models_dir.glob(pattern))
    
    if not model_files:
        # Fallback to explicit filename if no models found
        return 'disaster_rf_v25-09-16_prod_2025-09-19.pkl'
    
    # Sort by modification time (newest first) and take the latest
    model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return model_files[0].name


    MODEL_FILENAME = 'disaster_rf_v25-11-06_prod_2025-11-06.pkl'
    GDRIVE_MODEL_ID = None
