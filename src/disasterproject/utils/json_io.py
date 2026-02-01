import json
import logging
from typing import Any, Dict, Optional


def load_json(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a JSON file and return its contents as a dictionary.

    This function opens a JSON file, decodes it into a Python object, and returns that object.
    If the file does not exist, cannot be opened, or does not contain a valid JSON object,
    an error message is logged and the function returns None.
    If the JSON object is not a dictionary, an error message is logged and the function returns None.

    Args:
        file_path (str): The file path of the JSON file.

    Returns:
        data (dict): The contents of the JSON file as a dictionary, or None if an error occurred.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from file {file_path}: {e}")
        return None
    except OSError as e:
        logging.error(f"Error opening file {file_path}: {e}")
        return None

    if not isinstance(data, dict):
        logging.error(f"Expected a dictionary in file: {file_path}, but got {type(data)} instead.")
        return None

    return data

def save_json(data: Any, file_path: str) -> None:
    """
    Save data to a JSON file.

    Legacy-compatible behavior: accepts any JSON-serializable object and
    raises exceptions on failure instead of returning a status flag.

    Args:
        data: Any JSON-serializable Python object.
        file_path (str): Destination path for the JSON file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_json_safe(data: Dict[str, Any], file_path: str) -> bool:
    """
    Save a dictionary to a JSON file, returning success as a boolean.

    This safe variant preserves the newer behavior by logging errors and
    returning False instead of raising exceptions.

    Args:
        data (dict): The dictionary to save to JSON.
        file_path (str): The file path where the JSON should be saved.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except OSError as e:
        logging.error(f"Error opening file for writing {file_path}: {e}")
        return False
    except (TypeError, ValueError) as e:
        logging.error(f"Error serializing data to JSON: {e}")
        return False


def load_model_parameters(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a JSON file and return its contents as a dictionary with model parameters.

    This function opens a JSON file, decodes it into a Python object, and returns that object.
    If the file does not exist, cannot be opened, or does not contain a valid JSON object,
    an error message is logged and the function returns None.
    If the JSON object is not a dictionary, an error message is logged and the function returns None.

    Args:
        file_path (str): The file path of the JSON file.

    Returns:
        data (dict): The contents of the JSON file as a dictionary, or None if an error occurred.
    """
    raw = load_json(file_path)
    if raw is None:
        return None

    # Unwrap nested structure and ignore metadata if present
    parameters = raw.get("parameters") if isinstance(raw, dict) and "parameters" in raw else raw

    if not isinstance(parameters, dict):
        logging.error("Invalid parameters format in %s; expected object, got %s", file_path, type(parameters))
        return None

    # Convert single-item lists to their values and two-item lists to tuples
    normalized: Dict[str, Any] = {}
    for k, v in parameters.items():
        if isinstance(v, list):
            if len(v) == 1:
                normalized[k] = v[0]
            elif len(v) == 2:
                normalized[k] = tuple(v)
            else:
                normalized[k] = v
        else:
            normalized[k] = v

    return normalized


def load_hyperparameter_optimization_config(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a JSON file and return its contents as a dictionary with hyperparameter optimization configuration.

    This function opens a JSON file, decodes it into a Python object, and returns that object.
    If the file does not exist, cannot be opened, or does not contain a valid JSON object,
    an error message is logged and the function returns None.
    If the JSON object is not a dictionary, an error message is logged and the function returns None.

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
