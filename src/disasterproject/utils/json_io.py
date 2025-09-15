"""Input/output utility functions for loading and saving configuration files."""

from typing import Any, Dict, Optional
import json
import logging

logger = logging.getLogger(__name__)


def load_json(file_path: str) -> Optional[Dict[str, Any]]:
    """Load a JSON file and return its contents as a dictionary.

    This function opens a JSON file, decodes it into a Python object, and returns that object.
    If the file does not exist, cannot be opened, or does not contain a valid JSON object,
    an error is logged and the function returns ``None``. If the JSON object is not a dictionary,
    an error is logged and the function returns ``None``.

    Args:
        file_path: The file path of the JSON file.

    Returns:
        The contents of the JSON file as a dictionary, or ``None`` if an error occurred.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data: Any = json.load(f)
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        return None
    except json.JSONDecodeError as exc:
        logger.error("Error decoding JSON from %s: %s", file_path, exc)
        return None
    except OSError as exc:
        logger.error("Error reading %s: %s", file_path, exc)
        return None

    if not isinstance(data, dict):
        logger.error(
            "Expected a dictionary in file: %s, but got %s", file_path, type(data)
        )
        return None

    return data


def load_model_parameters(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a JSON file and return its contents as a dictionary with parameter processing.

    This function opens a JSON file, decodes it into a Python object, and returns that object. 
    It handles both flat parameter structures and nested structures with metadata.
    It also processes the parameters by converting single-item lists to their values and 
    two-item lists to tuples for sklearn compatibility.

    Args:
    file_path (str): The file path of the JSON file.

    Returns:
    data (dict): The processed parameters as a dictionary, or None if an error occurred.
    """
    data = load_json(file_path)
    if data is None:
        return None

    # Handle nested structure with metadata - extract just the parameters
    if "parameters" in data and "metadata" in data:
        parameters = data["parameters"]
        model_name = data["metadata"].get("model_name", "unknown")
        version = data["metadata"].get("version", "unknown")
        logger.info("Loaded parameters from %s version %s", model_name, version)
    else:
        # Handle flat structure (backward compatibility)
        parameters = data

    # Convert single-item lists to their values and two-item lists to tuples
    for k, v in parameters.items():
        if isinstance(v, list):
            if len(v) == 1:
                parameters[k] = v[0]
            elif len(v) == 2:
                parameters[k] = tuple(v)

    return parameters


def load_hyperparameter_optimization_config(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a JSON file and return its contents as a dictionary with hyperparameter optimization configuration.

    This function opens a JSON file, decodes it into a Python object, and returns that object. 
    It also processes the parameters by converting single-item lists to their values and 
    lists of two-item lists to lists of tuples for sklearn GridSearchCV compatibility.

    Args:
    file_path (str): The file path of the JSON file.

    Returns:
    data (dict): The contents of the JSON file as a dictionary, or None if an error occurred.
    """
    parameters = load_json(file_path)
    if parameters is None:
        return None

    # Convert single-item lists to their values and lists of two-item lists to lists of tuples
    for k, v in parameters.items():
        if isinstance(v, list):
            if len(v) == 1:
                parameters[k] = v[0]
            elif all(isinstance(i, list) and len(i) == 2 for i in v):
                parameters[k] = [tuple(i) for i in v]

    return parameters


def save_json(data: Any, file_path: str) -> None:
    """Save data to a JSON file.

    Args:
        data: The data to save (must be JSON serializable).
        file_path: The file path where the JSON file should be saved.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:  # pragma: no cover - basic error propagation
        logger.error("Error saving JSON to %s: %s", file_path, exc)
        raise
