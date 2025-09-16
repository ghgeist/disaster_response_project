def load_hyperparameter_optimization_config(file_path):
    """
    Load a JSON file and return its contents as a dictionary with hyperparameter optimization configuration.
    """
    import json
    with open(file_path, "r", encoding="utf-8") as f:
        parameters = json.load(f)
    # Convert single-item lists to their values and lists of two-item lists to lists of tuples
    for k, v in parameters.items():
        if isinstance(v, list):
            if len(v) == 1:
                parameters[k] = v[0]
            elif all(isinstance(i, list) and len(i) == 2 for i in v):
                parameters[k] = [tuple(i) for i in v]
    return parameters