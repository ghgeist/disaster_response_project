"""
Legacy compatibility module for model loading.

This module handles backward compatibility for models that were trained with
the old disaster_classifier module structure. It provides a clean interface
to load these models while isolating the compatibility shims.
"""
import logging
import sys
import os
import types
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

# Cache for module mappings to avoid recreating fake modules
_module_mapping_cache = {}


def load_with_legacy_paths(pickle_path: Path) -> Any:
    """
    Load a model with legacy disaster_classifier compatibility.
    
    This function handles models that were trained with the old 'disaster_classifier'
    module structure by creating fake modules in sys.modules to satisfy the
    unpickling process. Uses caching to avoid recreating modules on every load.
    
    Args:
        pickle_path: Path to the pickled model file
        
    Returns:
        The loaded model object
        
    Raises:
        RuntimeError: If the model cannot be loaded
    """
    try:
        # Check if we've already set up the compatibility shim
        if 'disaster_classifier.data.preprocessor' in _module_mapping_cache:
            logger.debug("Using cached legacy compatibility shim")
        else:
            logger.info("Setting up legacy compatibility shim (cached for future loads)")
            
            # Import the tokenize function from its actual current location
            from disasterproject.data.preprocessor import tokenize
            
            # Create a fake module structure in sys.modules to trick the unpickler
            # This is necessary because the model was pickled with a reference to
            # 'disaster_classifier' but the code has been refactored to 'disasterproject'
            logger.info("Creating fake module structure for disaster_classifier compatibility")
            
            # Create fake parent module 'disaster_classifier'
            disaster_classifier_mod = types.ModuleType('disaster_classifier')
            sys.modules['disaster_classifier'] = disaster_classifier_mod
            
            # Create fake child module 'disaster_classifier.data' and link it
            data_mod = types.ModuleType('disaster_classifier.data')
            sys.modules['disaster_classifier.data'] = data_mod
            disaster_classifier_mod.data = data_mod
            
            # Create the target module, attach the real function, and link it
            preprocessor_module = types.ModuleType('disaster_classifier.data.preprocessor')
            preprocessor_module.tokenize = tokenize
            sys.modules['disaster_classifier.data.preprocessor'] = preprocessor_module
            data_mod.preprocessor = preprocessor_module
            
            # Also inject tokenize into __main__ as additional fallback
            import __main__
            if not hasattr(__main__, 'tokenize'):
                __main__.tokenize = tokenize
            
            # Cache the setup to avoid recreating on future loads
            _module_mapping_cache['disaster_classifier.data.preprocessor'] = {
                'disaster_classifier': disaster_classifier_mod,
                'data': data_mod,
                'preprocessor': preprocessor_module
            }
            
            logger.info("Successfully created and cached fake module structure for model compatibility")
        
        # Load the model with the compatibility shim in place
        model = joblib.load(pickle_path)
        logger.info(f"Model loaded successfully with legacy compatibility from {pickle_path}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model with legacy compatibility: {e}")
        raise RuntimeError(f"Legacy model loading failed: {e}") from e


def clear_compatibility_cache():
    """
    Clear the compatibility module cache.
    
    This is useful for testing or if you need to reset the module mappings.
    """
    global _module_mapping_cache
    _module_mapping_cache.clear()
    logger.info("Compatibility module cache cleared")


def get_compatibility_cache_status():
    """
    Get the current status of the compatibility cache.
    
    Returns:
        Dictionary with cache status information
    """
    return {
        "cache_size": len(_module_mapping_cache),
        "cached_modules": list(_module_mapping_cache.keys()),
        "is_legacy_shim_active": 'disaster_classifier.data.preprocessor' in sys.modules
    }