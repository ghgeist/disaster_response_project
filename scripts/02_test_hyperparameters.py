#!/usr/bin/env python3
"""
Test hyperparameter optimization for disaster response classification.

This script provides hyperparameter tuning using GridSearchCV to find optimal
parameters for the disaster response classification model.

Use this script when you want to:
- Find optimal hyperparameters for the model
- Compare different parameter combinations
- Optimize model performance systematically

For baseline models, use create_baseline_model.py instead.
For class weighting, use create_weighted_model.py instead.
For sampling experiments, use test_sampling_strategies.py instead.

Usage:
    python scripts/02_test_hyperparameters.py data/02_stg/stg_disaster_response.db [model_output.pkl]
"""

# Standard library imports
import json
import logging
import multiprocessing
import os
import pickle
import sys
from datetime import datetime

# Third-party imports
import numpy as np
import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

# Local imports
from disasterproject.models.hyperparameter_search import run_parameter_search
from disasterproject.utils.config import FEATURE_COLUMNS, TARGET_COLUMNS
from disasterproject.data.preprocessor import tokenize
# Removed unused sampling imports - not compatible with multi-label classification

# Download required NLTK resources
nltk_resources = {
    "corpora": ["stopwords", "wordnet"],
    "tokenizers": ["punkt"]
}

for resource_type, resources in nltk_resources.items():
    for resource in resources:
        try:
            if resource_type == "corpora":
                nltk.data.find(f"corpora/{resource}")
            elif resource_type == "tokenizers":
                nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.download(resource)
                logging.info(f"Downloaded NLTK resource: {resource}")
            except Exception as e:
                logging.warning(f"Failed to download NLTK resource {resource}: {e}")
                # Continue execution as some resources might be optional

# Ensure WordNet is fully loaded to prevent multiprocessing race conditions
try:
    from nltk.corpus import wordnet as wn
    wn.ensure_loaded()  # This forces complete loading in the main thread
    logging.info("WordNet corpus fully loaded and ready for multiprocessing")
except Exception as e:
    logging.warning(f"Failed to ensure WordNet is loaded: {e}")
    # Continue execution but multiprocessing may have issues


# Logging is configured in main() via logging.basicConfig to avoid duplicate handlers

# Constants and tokenizer now imported from disasterproject modules

# Experiment paths structure
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Base experiment paths
BASE_PARAMETERS = os.path.join(PROJECT_ROOT, "experiments", "model_candidates", "parameters.json")
HYPERPARAMETER_OPTIMIZATION = os.path.join(PROJECT_ROOT, "experiments", "experimental_configs", "hyperparameters", "2025-09-16_comprehensive-grid_search.json")

# Dynamic paths based on config filename - maintains config->results link
DATE_PREFIX = datetime.now().strftime("%Y-%m-%d")
HYPERPARAMETER_LOG = os.path.join(PROJECT_ROOT, "experiments", "logs", f"{DATE_PREFIX}_hyperparameter_search.log")

logging.info("Setting random seed...")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_data(db_filepath):
    """
    Load data from a SQLite database.

    This function reads a table from a SQLite database and splits it into features (X) and labels (y). 
    The features are the 'message' column of the table, and the labels are the columns specified by TARGET_COLUMNS.
    If any of the TARGET_COLUMNS contain NaN values, a ValueError is raised.

    Args:
    db_filepath (str): The file path of the SQLite database.

    Returns:
    X (numpy.ndarray): The features from the 'message' column of the table.
    y (numpy.ndarray): The labels from the columns specified by TARGET_COLUMNS.

    Raises:
    ValueError: If any of the TARGET_COLUMNS contain NaN values.
    """
    try:
        database_url = "sqlite:///" + db_filepath.replace("\\", "/")
        engine = create_engine(database_url)
    except OperationalError:
        logging.error("Error connecting to database at %s", db_filepath)
        return None, None

    table_name = os.path.splitext(os.path.basename(db_filepath))[0]

    try:
        df = pd.read_sql_table(table_name, engine)
    except ValueError:
        logging.error("Table %s not found in database", table_name)
        return None, None

    try:
        X = df.message.values
        y = df[TARGET_COLUMNS].values

        nan_columns = df[TARGET_COLUMNS].isna().any()
        nan_columns_list = nan_columns[nan_columns == True].index.tolist()

        if len(nan_columns_list) > 0:
            logging.error("Columns with NaN values: %s", nan_columns_list)
            raise ValueError(
                "NaN values found in columns: %s. Check the TARGET_COLUMNS to make sure they are set up correctly "
                "or the underlying data" % nan_columns_list
            )

    except KeyError as e:
        logging.error("Column %s not found in table", e.args[0])
        return None, None
    except ValueError as e:
        logging.error(e)
        return None, None

    return X, y


# Tokenizer now imported from disasterproject.data.preprocessor


def load_json(file_path):
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
        logging.error("File not found: %s", file_path)
        return None
    except json.JSONDecodeError:
        logging.error("Error decoding JSON from file: %s", file_path)
        return None

    if not isinstance(data, dict):
        logging.error(
            "Expected a dictionary in file: %s, but got %s instead.", file_path, type(data)
        )
        return None

    return data


def generate_output_paths(config_path):
    """
    Generate output file paths based on config filename following naming convention.

    Example:
    Input:  "2025-09-16_comprehensive-grid_search.json"
    Output: "2025-09-16-comprehensive-grid-search-optimized-hyperparameters.json"
    """
    config_filename = os.path.basename(config_path)
    config_name = os.path.splitext(config_filename)[0]  # Remove .json

    # Transform: "2025-09-16_comprehensive-grid_search" -> "2025-09-16-comprehensive-grid-search-optimized-hyperparameters"
    output_base = config_name.replace('_', '-') + "-optimized-hyperparameters"

    results_file = f"{output_base}.json"
    detailed_results = config_name.replace('_', '-') + "-detailed-results.json"

    return {
        'optimized_params': os.path.join(PROJECT_ROOT, "experiments", "model_candidates", results_file),
        'detailed_results': os.path.join(PROJECT_ROOT, "experiments", "results", detailed_results)
    }


def load_parameters(file_path, config_type="model"):
    """
    Load and normalize parameters from a JSON configuration file.

    This function handles both model parameters and hyperparameter optimization configurations.
    It unwraps nested structures, normalizes list formats, and handles different parameter types.

    Args:
    file_path (str): The file path of the JSON file.
    config_type (str): Type of config - "model" for model parameters, "hyperopt" for optimization configs.

    Returns:
    dict: The normalized parameters, or None if an error occurred.

    """
    raw = load_json(file_path)
    if raw is None:
        logging.error("load_json returned None for file: %s", file_path)
        return None

    # For model parameters, unwrap nested structure if present
    if config_type == "model":
        parameters = raw.get("parameters") if isinstance(raw, dict) and "parameters" in raw else raw
    else:
        parameters = raw

    if not isinstance(parameters, dict):
        logging.error("Invalid parameters format in %s; expected object, got %s", file_path, type(parameters))
        return None

    # Normalize parameter formats
    normalized = {}
    for k, v in parameters.items():
        if isinstance(v, list):
            if len(v) == 1:
                normalized[k] = v[0]
            elif len(v) == 2 and config_type == "model":
                normalized[k] = tuple(v)
            elif config_type == "hyperopt" and all(isinstance(i, list) and len(i) == 2 for i in v):
                normalized[k] = [tuple(i) for i in v]
            else:
                normalized[k] = v
        else:
            normalized[k] = v

    return normalized


def create_pipeline():
    """
    Create a machine learning pipeline.

    This function creates a pipeline that first vectorizes the text data using CountVectorizer and a custom tokenizer,
    then applies a TF-IDF transformation, and finally uses a multi-output classifier.

    Returns:
    pipeline (sklearn.pipeline.Pipeline): The machine learning pipeline.

    If an error occurs while creating the pipeline, None is returned.
    """
    try:
        # Instantiate and configure the pipeline
        pipeline = Pipeline(
            [
                (
                    "vect",
                    CountVectorizer(analyzer=tokenize, token_pattern=None, lowercase=False),
                ),  # Tokenize and vectorize text without token_pattern warnings
                (
                    "tfidf",
                    TfidfTransformer(smooth_idf=False),
                ),  # Apply TF-IDF transformation
                (
                    "clf",
                    MultiOutputClassifier(
                        RandomForestClassifier(n_jobs=1)
                    ),
                ),  # Use MultiOutputClassifier with RandomForest, n_jobs specifies cores
            ]
        )
    except (TypeError, ValueError) as e:
        logging.error("Error creating pipeline: %s", e)
        return None

    return pipeline


def build_model(pipeline, parameters):
    """
    Build a machine learning model.

    This function configures the RandomForestClassifier in the pipeline with the given parameters.

    Args:
    pipeline (sklearn.pipeline.Pipeline): The machine learning pipeline.
    parameters (dict): The parameters for the RandomForestClassifier.

    Returns:
    pipeline (sklearn.pipeline.Pipeline): The machine learning pipeline.

    If an error occurs while building the model, None is returned.
    """
    try:
        # Configure the RandomForestClassifier with the given parameters
        if parameters is None:
            pipeline.set_params(clf__estimator__random_state=RANDOM_STATE)
        else:
            pipeline.set_params(clf__estimator__random_state=RANDOM_STATE, **parameters)
    except (ValueError, ImportError, TypeError) as e:
        logging.error("Error building model: %s", e)
        return None

    return pipeline


def evaluate_model(model, model_name, X_test, Y_test, category_names):
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
            PROJECT_ROOT, "experiments", "results", f"fct_{model_name}_prediction_results.csv"
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
    Run a grid search to find the best parameters for a pipeline.

    This function uses GridSearchCV to find the best parameters for the specified pipeline using the provided training data.
    The function measures the time it takes to run the grid search and logs the runtime.
    If use_small_subset is True, the function uses only the first 100 samples of the training data and estimates the total runtime based on this subset.

    Args:
    pipeline (Pipeline): The pipeline for which to find the best parameters.
    parameters (dict): The parameters to try in the grid search.
    X_train (numpy.ndarray): The features for the training data.
    y_train (numpy.ndarray): The labels for the training data.
    use_small_subset (bool, optional): Whether to use only the first 100 samples of the training data. Defaults to False.

    Returns:
    cv (GridSearchCV): The fitted GridSearchCV instance.

    """
    cv_results = cv.cv_results_
    results = []

    # Handle multi-metric results: capture both and include refit metric
    mean_weighted = cv_results.get("mean_test_f1_weighted")
    mean_micro = cv_results.get("mean_test_f1_micro")

    for idx, params in enumerate(cv_results["params"]):
        entry = {"params": params}
        if mean_weighted is not None:
            entry["mean_test_f1_weighted"] = float(mean_weighted[idx])
        if mean_micro is not None:
            entry["mean_test_f1_micro"] = float(mean_micro[idx])
        results.append(entry)

    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(results, f)
    except FileNotFoundError as e:
        logging.error("Error saving results: %s", e)


def save_best_parameters(cv, output_file_path):
    """
    Save the best parameters of a grid search to a JSON file in expected format.

    This function extracts the best parameters from the results of a GridSearchCV,
    and saves them to a JSON file with metadata structure that matches the expected format.

    Args:
    cv (GridSearchCV): The fitted GridSearchCV instance.
    output_file_path (str): The file path where the parameters should be saved.

    Raises:
    FileNotFoundError: If the file cannot be opened for writing.
    """
    best_params = cv.best_params_

    # Create payload in expected format with metadata
    payload = {
        "metadata": {
            "model_name": "optimized_disaster_classifier",
            "version": "1.1.0",
            "created_date": datetime.now().strftime("%Y-%m-%d"),
            "description": "Optimized parameters from comprehensive grid search using RandomizedSearchCV",
            "model_type": "RandomForestClassifier with TfidfVectorizer",
            "last_modified": datetime.now().strftime("%Y-%m-%d"),
            "experiment_id": f"grid_search_{datetime.now().strftime('%Y-%m-%d')}",
            "refit_metric": getattr(cv, "refit", None),
            "best_score": float(getattr(cv, "best_score_", float("nan"))),
            "search_method": "RandomizedSearchCV"
        },
        "parameters": best_params
    }

    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except FileNotFoundError as e:
        logging.error("Error saving best parameters: %s", e)


def get_user_input(prompt):
    """
    Get user input with validation.

    This function prompts the user for input and validates it. 
    The function continues to prompt the user until they enter 'yes', 'no', or 'exit' (case insensitive).

    Args:
    prompt (str): The prompt to display to the user.

    Returns:
    user_input (str): The validated user input, converted to lowercase.

    """
    while True:
        user_input = input(prompt)
        if user_input.lower() in ["yes", "no", "exit"]:
            return user_input.lower()
        else:
            print("Invalid input. Please enter 'yes', 'no', or 'exit'.")


# Removed apply_smote_sampling function - SMOTE doesn't work with multi-label data


# Removed apply_multi_label_aware_sampling function - SMOTE/ADASYN don't support multi-label classification
# Multi-label sampling requires specialized approaches that are beyond the scope of hyperparameter tuning


def main():
    """
    Main function to train a classifier.

    This function loads data from a database file, splits it into training and test sets, 
    and trains a classifier using a pipeline. The user is given the option to retrain the base model, 
    estimate the grid search runtime, run a grid search, and retrain the model using the optimized parameters found by the grid search. 
    The trained model is then saved to a pickle file.

    Args:
    None

    Returns:
    None
    """
    import argparse

    # Configure logging to write to both console and log file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(HYPERPARAMETER_LOG),
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("database_filepath")
    parser.add_argument("model_filepath")
    parser.add_argument("--config", default=HYPERPARAMETER_OPTIMIZATION)
    args = parser.parse_args()

    database_filepath = args.database_filepath
    model_filepath = args.model_filepath
    hyperparameter_config_path = args.config

    # Generate output paths based on config filename following naming convention
    output_paths = generate_output_paths(hyperparameter_config_path)

    # Debug logging for path resolution
    logging.info("Using hyperparameter config path: %s", hyperparameter_config_path)
    logging.info("Config file exists: %s", os.path.exists(hyperparameter_config_path))
    logging.info("Will save optimized parameters to: %s", output_paths['optimized_params'])
    logging.info("Will save detailed results to: %s", output_paths['detailed_results'])

    logging.info("Loading data from database: %s", database_filepath)
    X, Y = load_data(database_filepath)
    if X is None or Y is None:
        logging.error("Error loading data from database")
        return

    logging.info("Splitting data into training and test sets...")
    # Use consistent random seed for reproducible results (matches production script approach)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_STATE)

    logging.info("Using original training data (no oversampling for multi-label classification)...")

    logging.info("Creating ML pipeline...")
    pipeline = create_pipeline()

    estimate_runtime = get_user_input(
        "Do you want to estimate the grid search runtime? (yes/no/exit): "
    )
    if estimate_runtime == "exit":
        sys.exit()
    elif estimate_runtime == "yes":
        logging.info("Loading grid search parameters...")
        hyperparameter_config = load_parameters(hyperparameter_config_path, "hyperopt")
        logging.info("Estimating grid search runtime (using small subset)...")
        estimated_grid_search = run_parameter_search(
            pipeline,
            hyperparameter_config,
            X_train,
            Y_train,
            use_small_subset=True,
        )
        logging.info("Grid search runtime estimate complete!")

    do_grid_search = get_user_input(
        "Do you want to run a grid search? (yes/no/exit): "
    )
    if do_grid_search == "exit":
        sys.exit()
    elif do_grid_search == "yes":
        logging.info("Starting full grid search...")
        hyperparameter_config = load_parameters(hyperparameter_config_path, "hyperopt")
        grid_search = run_parameter_search(
            pipeline,
            hyperparameter_config,
            X_train,
            Y_train,
            use_small_subset=False,
        )
        logging.info("Grid search complete!")
        save_gs_results(grid_search, output_paths['detailed_results'])
        save_best_parameters(grid_search, output_paths['optimized_params'])
        logging.info("Grid search results and optimized parameters saved!")

        print(f"\n🎉 HYPERPARAMETER SEARCH COMPLETE!")
        print(f"=" * 50)
        print(f"📄 Detailed results saved to: {output_paths['detailed_results']}")
        print(f"⚙️  Optimized parameters saved to: {output_paths['optimized_params']}")
        print(f"📊 Best score achieved: {grid_search.best_score_:.4f}")
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Review the optimized parameters in: {output_paths['optimized_params']}")
        print(f"   2. Create optimized model with:")
        print(f"      python scripts/03_create_experimental_model.py \\")
        print(f"         --params {output_paths['optimized_params']}")
        print(f"=" * 50)


if __name__ == "__main__":
    main()
