# Standard library imports
import datetime
import json
import logging
import multiprocessing
import os
import pickle
import re
import string
import sys
from time import time

# Third-party imports
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import MultiOutputSampler

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


# Set up logging
logFormatter = logging.Formatter(
    "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
)
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler("app.log")
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

rootLogger.setLevel(logging.INFO)

FEATURE_COLUMNS = ["message"]
TARGET_COLUMNS = [
    "related",
    "request",
    "offer",
    "aid_related",
    "medical_help",
    "medical_products",
    "search_and_rescue",
    "security",
    "military",
    "child_alone",
    "water",
    "food",
    "shelter",
    "clothing",
    "money",
    "missing_people",
    "refugees",
    "death",
    "other_aid",
    "infrastructure_related",
    "transport",
    "buildings",
    "electricity",
    "tools",
    "hospitals",
    "shops",
    "aid_centers",
    "other_infrastructure",
    "weather_related",
    "floods",
    "storm",
    "fire",
    "earthquake",
    "cold",
    "other_weather",
    "direct_report",
]

STOPWORDS_SET = set(stopwords.words("english"))
URL_REGEX = (
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)
URL_PLACE_HOLDER = "urlplaceholder"

SCRIPT_DIR = os.path.dirname(__file__)
BASE_PARAMETERS = os.path.join(SCRIPT_DIR, "base_parameters.json")
GRID_SEARCH_PARAMETERS = os.path.join(SCRIPT_DIR, "grid_search_parameters.json")
GRID_SEARCH_RESULTS = os.path.join(SCRIPT_DIR, "gs_results.json")
OPTIMIZED_PARAMETERS = os.path.join(SCRIPT_DIR, "optimized_parameters.json")

logging.info("Setting random seed...")
np.random.seed(0)


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


def tokenize(text):
    """
    Tokenize the message data.

    This function detects and replaces URLs, removes punctuation, tokenizes the text, removes stop words, and lemmatizes the tokens.

    Parameters:
    text (str): The text to be tokenized.

    Returns:
    cleaned_tokens (list of str): The tokenized and cleaned text.

    If an error occurs during tokenization, an empty list is returned.
    """
    try:
        # Detect and replace URLs
        text = re.sub(URL_REGEX, URL_PLACE_HOLDER, text)
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Tokenize text
        tokens = word_tokenize(text)
        # Remove stop words
        tokens = [token for token in tokens if token not in STOPWORDS_SET]
        # Lemmatize tokens
        lemmatizer = WordNetLemmatizer()
        cleaned_tokens = [
            lemmatizer.lemmatize(token.lower().strip()) for token in tokens
        ]
    except Exception as e:
        logging.error("Error tokenizing text: %s", e)
        return []

    return cleaned_tokens


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
                    CountVectorizer(tokenizer=tokenize),
                ),  # Tokenize and vectorize text
                (
                    "tfidf",
                    TfidfTransformer(smooth_idf=False),
                ),  # Apply TF-IDF transformation
                (
                    "clf",
                    MultiOutputClassifier(
                        RandomForestClassifier(n_jobs=multiprocessing.cpu_count() - 1)
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
            pipeline.set_params(clf__estimator__random_state=42)
        else:
            pipeline.set_params(clf__estimator__random_state=42, **parameters)
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


def run_grid_search(pipeline, parameters, X_train, y_train, use_small_subset=False):
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
    start_time = time()
    cv = GridSearchCV(
        pipeline,
        param_grid=parameters,
        scoring="accuracy",
        n_jobs=multiprocessing.cpu_count() - 1,
        verbose=1,
    )

    if use_small_subset:
        X_train_size = len(X_train)
        X_train = X_train[:100]
        y_train = y_train[:100]
        cv.fit(X_train, y_train)
        end_time = time()
        runtime = (end_time - start_time) * (
            X_train_size / 100
        )  # keep the time in seconds
        formatted_runtime = f"{runtime:.2f} seconds (estimated)"
    else:
        cv.fit(X_train, y_train)
        end_time = time()
        runtime = end_time - start_time  # keep the time in seconds
        hours, remainder = divmod(runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_runtime = f"{int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds (actual)"

    logging.info(f"Runtime: {formatted_runtime}")

    return cv


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
    best_params = cv.best_params_

    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f)
    except FileNotFoundError as e:
        logging.error(f"Error saving best parameters: {e}")


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


def apply_smote_sampling(X_train, y_train):
    """
    Apply SMOTE oversampling to handle class imbalance in multi-label classification.
    
    This function applies SMOTE to each target column individually using MultiOutputSampler
    to handle the multi-label classification properly. It also handles cases where SMOTE
    cannot be applied (e.g., single class) and provides before/after class distribution statistics.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
        
    Returns:
        tuple: (X_train_resampled, y_train_resampled) - The resampled training data
        
    Raises:
        ValueError: If SMOTE cannot be applied to any target column
    """
    try:
        # Print before class distribution statistics
        logging.info("Class distribution BEFORE SMOTE:")
        for i, col in enumerate(TARGET_COLUMNS):
            unique, counts = np.unique(y_train[:, i], return_counts=True)
            class_dist = dict(zip(unique, counts))
            logging.info(f"  {col}: {class_dist}")
        
        # Apply SMOTE using MultiOutputSampler for multi-label classification
        sampler = MultiOutputSampler(
            SMOTE(random_state=42, k_neighbors=1),
            random_state=42
        )
        
        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
        
        # Print after class distribution statistics
        logging.info("Class distribution AFTER SMOTE:")
        for i, col in enumerate(TARGET_COLUMNS):
            unique, counts = np.unique(y_train_resampled[:, i], return_counts=True)
            class_dist = dict(zip(unique, counts))
            logging.info(f"  {col}: {class_dist}")
        
        logging.info(f"Training samples: {len(X_train)} -> {len(X_train_resampled)}")
        
        return X_train_resampled, y_train_resampled
        
    except Exception as e:
        logging.error(f"Error applying SMOTE: {e}")
        logging.warning("SMOTE could not be applied. Using original training data.")
        return X_train, y_train


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
    if len(sys.argv) != 3:
        logging.info(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )
    else:
        database_filepath, model_filepath = sys.argv[1:]

        logging.info("Loading data...\n    DATABASE: %s", database_filepath)
        X, Y = load_data(database_filepath)
        if X is None or Y is None:
            logging.error("Error loading data from database")
            return

        logging.info("Splitting the data into training and test sets...")
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        logging.info("Applying SMOTE oversampling to handle class imbalance...")
        X_train, Y_train = apply_smote_sampling(X_train, Y_train)

        logging.info("Creating pipeline...")
        pipeline = create_pipeline()

        retrain_base_model = get_user_input(
            "Do you want to retrain the base model? (yes/no/exit): "
        )
        if retrain_base_model == "exit":
            sys.exit()
        elif retrain_base_model == "yes":
            logging.info("Loading base parameters from %s...", BASE_PARAMETERS)
            base_parameters = load_model_parameters(BASE_PARAMETERS)
            if base_parameters is None:
                logging.error(
                    f"Failed to load base parameters from {BASE_PARAMETERS}. Please ensure the file exists and is a valid JSON file."
                )
            else:
                logging.info(
                    f"Base parameters loaded successfully from {BASE_PARAMETERS} ."
                )

            logging.info("Building model from base parameters...")
            base_model = build_model(pipeline, base_parameters)

            logging.info("Training model with base parameters...")
            base_model.fit(X_train, Y_train)

            logging.info("Evaluating model with base parameters...")
            evaluate_model(base_model, "base_model", X_test, Y_test, TARGET_COLUMNS)

            logging.info(
                "Saving model built from base parameters...\n    MODEL: %s",
                model_filepath,
            )
            save_model(base_model, model_filepath)

            logging.info("Trained model saved!")

        estimate_runtime = get_user_input(
            "Do you want to estimate the grid search runtime? (yes/no/exit): "
        )
        if estimate_runtime == "exit":
            sys.exit()
        elif estimate_runtime == "yes":
            logging.info("Loading GridSearch parameters from %s...", GRID_SEARCH_PARAMETERS)
            grid_search_parameters = load_grid_search_parameters(GRID_SEARCH_PARAMETERS)
            logging.info("Estimating grid search runtime...")
            estimated_grid_search = run_grid_search(
                pipeline,
                grid_search_parameters,
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
            logging.info("Starting grid search...")
            grid_search_parameters = load_grid_search_parameters(GRID_SEARCH_PARAMETERS)
            grid_search = run_grid_search(
                pipeline,
                grid_search_parameters,
                X_train,
                Y_train,
                use_small_subset=False,
            )
            logging.info("Grid search complete!")
            logging.info("Loading GridSearch parameters from %s...", GRID_SEARCH_PARAMETERS)
            save_gs_results(grid_search, GRID_SEARCH_RESULTS)
            logging.info("Grid search results saved!")
            save_best_parameters(grid_search, OPTIMIZED_PARAMETERS)
            logging.info("Optimized parameters saved!")

        retrain_optimized_model = get_user_input(
            "Do you want to retrain the model using the optimized parameters found by the grid search? (yes/no/exit): "
        )
        if retrain_optimized_model == "exit":
            sys.exit()
        elif retrain_optimized_model == "yes":
            logging.info("Building model from optimized parameters...")
            optimized_parameters = load_model_parameters(OPTIMIZED_PARAMETERS)
            if optimized_parameters is None:
                logging.error(
                    "Failed to load optimized parameters from %s. Please ensure the file exists and is a valid JSON file.", 
                    OPTIMIZED_PARAMETERS
                    )
            else:
                logging.info(
                    "Optimized parameters loaded successfully from %s.", 
                    OPTIMIZED_PARAMETERS
                    )
            logging.info("Building model from optimized parameters...")
            optimized_model = build_model(pipeline, optimized_parameters)
            logging.info("Training model with optimized parameters...")
            optimized_model.fit(X_train, Y_train)
            logging.info("Evaluating model with optimized parameters...")
            evaluate_model(
                optimized_model, "optimized_model", X_test, Y_test, TARGET_COLUMNS
            )
            logging.info(
                "Saving model built from optimized parameters...\n    MODEL: %s",
                model_filepath,
            )
            save_model(optimized_model, model_filepath)
            logging.info("Trained model saved!")


if __name__ == "__main__":
    main()
