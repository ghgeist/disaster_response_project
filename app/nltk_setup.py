"""One-time NLTK resource setup for application startup.

This module handles NLTK resource downloading and validation during
application startup to avoid per-request performance issues. All NLTK
resources are pre-loaded and validated once at startup.
"""
import logging
import time
from typing import Dict, List, Tuple, Optional
import nltk
from nltk.corpus import stopwords, wordnet

logger = logging.getLogger(__name__)

# NLTK resources required for the application
REQUIRED_RESOURCES = {
    "corpora": ["stopwords", "wordnet"],
    "tokenizers": ["punkt"]
}

# Resource validation functions
RESOURCE_VALIDATORS = {
    "stopwords": lambda: len(stopwords.words("english")) > 0,
    "wordnet": lambda: len(list(wordnet.synsets("test"))) > 0,
    "punkt": lambda: nltk.data.find("tokenizers/punkt") is not None
}


class NLTKSetupError(Exception):
    """Exception raised when NLTK setup fails."""
    pass


def setup_nltk_resources(force_download: bool = False) -> Dict[str, any]:
    """
    Download and validate NLTK resources once at startup.
    
    Args:
        force_download: If True, force download even if resources exist
        
    Returns:
        Dictionary with setup status and timing information
        
    Raises:
        NLTKSetupError: If critical resources cannot be loaded
    """
    start_time = time.time()
    setup_results = {
        "success": True,
        "resources_loaded": [],
        "resources_failed": [],
        "downloads_attempted": 0,
        "downloads_successful": 0,
        "setup_time_ms": 0,
        "errors": []
    }
    
    logger.info("Starting NLTK resource setup...")
    
    try:
        # Process each resource type
        for resource_type, resources in REQUIRED_RESOURCES.items():
            for resource in resources:
                try:
                    resource_start = time.time()
                    success, download_attempted = _setup_single_resource(resource_type, resource, force_download)
                    resource_time = (time.time() - resource_start) * 1000

                    # Update download counters
                    if download_attempted:
                        setup_results["downloads_attempted"] += 1
                        if success:
                            setup_results["downloads_successful"] += 1

                    if success:
                        setup_results["resources_loaded"].append({
                            "name": resource,
                            "type": resource_type,
                            "load_time_ms": round(resource_time, 2)
                        })
                        logger.info(
                            "✓ %s/%s loaded successfully (%.1fms)",
                            resource_type,
                            resource,
                            resource_time,
                        )
                    else:
                        setup_results["resources_failed"].append({
                            "name": resource,
                            "type": resource_type,
                            "error": "Validation failed"
                        })
                        logger.warning(
                            "✗ %s/%s validation failed",
                            resource_type,
                            resource,
                        )

                except Exception as e:
                    error_msg = f"Failed to setup {resource_type}/{resource}: {e}"
                    setup_results["resources_failed"].append({
                        "name": resource,
                        "type": resource_type,
                        "error": str(e)
                    })
                    setup_results["errors"].append(error_msg)
                    logger.error("✗ %s", error_msg)
        
        # Check if critical resources are available
        critical_resources = ["stopwords", "punkt"]
        missing_critical = [
            r["name"] for r in setup_results["resources_failed"] 
            if r["name"] in critical_resources
        ]
        
        if missing_critical:
            error_msg = f"Critical NLTK resources missing: {missing_critical}"
            setup_results["success"] = False
            setup_results["errors"].append(error_msg)
            logger.error(error_msg)
            raise NLTKSetupError(error_msg)
        
        # Ensure WordNet is fully loaded for multiprocessing compatibility
        try:
            wordnet.ensure_loaded()
            logger.info("WordNet corpus fully loaded and ready for multiprocessing")
        except Exception as e:
            warning_msg = f"Failed to ensure WordNet is loaded: {e}"
            setup_results["errors"].append(warning_msg)
            logger.warning(warning_msg)
        
        setup_results["setup_time_ms"] = round((time.time() - start_time) * 1000, 2)
        
        if setup_results["success"]:
            logger.info(
                "NLTK setup completed successfully in %sms",
                setup_results["setup_time_ms"],
            )
        else:
            logger.warning(
                "NLTK setup completed with warnings in %sms",
                setup_results["setup_time_ms"],
            )
            
        return setup_results
        
    except Exception as e:
        setup_results["success"] = False
        setup_results["setup_time_ms"] = round((time.time() - start_time) * 1000, 2)
        setup_results["errors"].append(f"Setup failed: {e}")
        logger.error("NLTK setup failed: %s", e)
        raise NLTKSetupError(f"NLTK setup failed: {e}") from e


def _setup_single_resource(resource_type: str, resource: str, force_download: bool) -> Tuple[bool, bool]:
    """
    Setup a single NLTK resource.

    Args:
        resource_type: Type of resource (corpora, tokenizers)
        resource: Name of the resource
        force_download: If True, force download even if resource exists

    Returns:
        Tuple of (success, download_attempted)
    """
    download_attempted = False
    try:
        # Check if resource already exists
        resource_path = f"{resource_type}/{resource}"
        try:
            nltk.data.find(resource_path)
            resource_exists = True
        except LookupError:
            resource_exists = False

        # Download if needed
        if not resource_exists or force_download:
            download_attempted = True
            logger.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, quiet=True)
            logger.info("Downloaded NLTK resource: %s", resource)

        # Validate resource
        if resource in RESOURCE_VALIDATORS:
            validation_result = RESOURCE_VALIDATORS[resource]()
            if not validation_result:
                logger.warning("Resource validation failed for %s", resource)
                return False, download_attempted

        return True, download_attempted

    except Exception as e:
        logger.error("Error setting up %s/%s: %s", resource_type, resource, e)
        return False, download_attempted


def validate_nltk_resources() -> Dict[str, any]:
    """
    Validate that all required NLTK resources are available.
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "all_available": True,
        "available_resources": [],
        "missing_resources": [],
        "validation_errors": []
    }
    
    for resource_type, resources in REQUIRED_RESOURCES.items():
        for resource in resources:
            try:
                resource_path = f"{resource_type}/{resource}"
                nltk.data.find(resource_path)
                
                # Additional validation if validator exists
                if resource in RESOURCE_VALIDATORS:
                    if RESOURCE_VALIDATORS[resource]():
                        validation_results["available_resources"].append(resource)
                    else:
                        validation_results["missing_resources"].append(resource)
                        validation_results["all_available"] = False
                else:
                    validation_results["available_resources"].append(resource)
                    
            except LookupError:
                validation_results["missing_resources"].append(resource)
                validation_results["all_available"] = False
            except Exception as e:
                validation_results["validation_errors"].append(f"{resource}: {e}")
                validation_results["all_available"] = False
    
    return validation_results


def get_nltk_status() -> Dict[str, any]:
    """
    Get current NLTK status for monitoring.
    
    Returns:
        Dictionary with NLTK status information
    """
    try:
        validation_results = validate_nltk_resources()
        
        return {
            "status": "healthy" if validation_results["all_available"] else "degraded",
            "all_resources_available": validation_results["all_available"],
            "available_resources": validation_results["available_resources"],
            "missing_resources": validation_results["missing_resources"],
            "validation_errors": validation_results["validation_errors"],
            "nltk_version": nltk.__version__,
            "data_path": nltk.data.path[0] if nltk.data.path else "Not configured"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "nltk_version": getattr(nltk, '__version__', 'Unknown')
        }