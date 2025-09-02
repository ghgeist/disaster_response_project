"""
Model evaluation and metrics calculation functions.
"""

import logging
import os
import pickle
import pandas as pd
from sklearn.metrics import classification_report

from ..utils.config import TARGET_COLUMNS


def evaluate_model(model, model_name, X_test, Y_test, category_names):
    """
    Evaluate a trained model and save detailed results.
    
    Args:
        model: The trained model to evaluate
        model_name (str): Name of the model for file naming
        X_test: Test features
        Y_test: Test labels
        category_names: List of category names for the target columns
    """
    try:
        Y_pred = model.predict(X_test)
        results = []

        for i, col in enumerate(category_names):
            report = classification_report(
                Y_test[:, i], Y_pred[:, i], output_dict=True, zero_division=0
            )
            for output_class, metrics in report.items():
                if isinstance(metrics, dict):
                    temp = metrics.copy()
                    temp["output_class"] = output_class
                    temp["category"] = col
                    results.append(temp)

        results_df = pd.DataFrame(results)
        results_df = results_df[
            ["category", "output_class", "precision", "recall", "f1-score", "support"]
        ]

        results_file_path = os.path.join(
            "data", "04_fct", f"fct_{model_name}_prediction_results.csv"
        )
        results_df.to_csv(results_file_path, index=False)
        logging.info("Evaluation results saved to: %s", results_file_path)

    except Exception as e:
        logging.error("Error evaluating model: %s", e)


def save_model(model, model_filepath):
    """
    Save a machine learning model to a file using pickle.

    Parameters:
    model (sklearn.base.BaseEstimator): The machine learning model to save.
    model_filepath (str): The file path where the model should be saved.

    If an error occurs while saving the model, a message is printed to the console.
    """
    try:
        with open(model_filepath, "wb") as f:
            pickle.dump(model, f)
    except Exception as e:
        logging.error("Error saving model: %s", e)


def save_gs_results(cv, output_file_path):
    """
    Save grid search results to a JSON file.

    Args:
    cv (GridSearchCV): The fitted GridSearchCV instance.
    output_file_path (str): The file path where the results should be saved.
    """
    import json
    
    cv_results = cv.cv_results_
    results = []

    for params, mean_score in zip(cv_results["params"], cv_results["mean_test_score"]):
        results.append({"params": params, "score": mean_score})

    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(results, f)
    except FileNotFoundError as e:
        logging.error(f"Error saving results: {e}")


def save_best_parameters(cv, output_file_path):
    """
    Save the best parameters of a grid search to a JSON file.

    This function extracts the best parameters from the results of a GridSearchCV, 
    and saves them to a JSON file. If the file cannot be opened for writing (for example, if the directory does not exist), 
    an error message is logged and the function returns without saving the parameters.

    Args:
    cv (GridSearchCV): The fitted GridSearchCV instance.
    output_file_path (str): The file path where the parameters should be saved.

    Raises:
    FileNotFoundError: If the file cannot be opened for writing.
    """
    import json
    
    best_params = cv.best_params_

    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f)
    except FileNotFoundError as e:
        logging.error(f"Error saving best parameters: {e}")
