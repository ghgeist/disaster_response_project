"""
Input/output utility functions for loading and saving configuration files.
"""

import json
import logging


def load_json(file_path):
    """
    Load a JSON file and return its contents as a dictionary.

    This function opens a JSON file, decodes it into a Python object, and returns that object. 
    If the file does not exist, cannot be opened, or does not contain a valid JSON object, 
    an error message is printed and the function returns None. 
    If the JSON object is not a dictionary, an error message is printed and the function returns None.

    Args:
    file_path (str): The file path of the JSON file.

    Returns:
    data (dict): The contents of the JSON file as a dictionary, or None if an error occurred.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        return None

    if not isinstance(data, dict):
        print(
            f"Expected a dictionary in file: {file_path}, but got {type(data)} instead."
        )
        return None

    return data


def load_model_parameters(file_path):
    """
    Load a JSON file and return its contents as a dictionary with parameter processing.

    This function opens a JSON file, decodes it into a Python object, and returns that object. 
    It also processes the parameters by converting single-item lists to their values and 
    two-item lists to tuples for sklearn compatibility.

    Args:
    file_path (str): The file path of the JSON file.

    Returns:
    data (dict): The contents of the JSON file as a dictionary, or None if an error occurred.
    """
    parameters = load_json(file_path)
    if parameters is None:
        return None

    # Convert single-item lists to their values and two-item lists to tuples
    for k, v in parameters.items():
        if isinstance(v, list):
            if len(v) == 1:
                parameters[k] = v[0]
            elif len(v) == 2:
                parameters[k] = tuple(v)

    return parameters


def load_grid_search_parameters(file_path):
    """
    Load a JSON file and return its contents as a dictionary with grid search parameter processing.

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


def save_json(data, file_path):
    """
    Save data to a JSON file.

    Args:
    data: The data to save (must be JSON serializable)
    file_path (str): The file path where the JSON file should be saved.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving JSON to {file_path}: {e}")
        raise
