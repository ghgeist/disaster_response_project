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

nltk_resources = ["stopwords", "wordnet"]
for resource in nltk_resources:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource)


# Set up logging
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler('app.log')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

rootLogger.setLevel(logging.INFO)

FEATURE_COLUMNS =['message']
TARGET_COLUMNS = [
    'related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products',
    'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food',
    'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death',
    'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity',
    'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
    'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
    'other_weather', 'direct_report']

STOPWORDS_SET = set(stopwords.words('english'))
URL_REGEX = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
URL_PLACE_HOLDER = "urlplaceholder"

logging.info('Setting random seed...')
np.random.seed(0)

def load_data(db_filepath):
    try:
        database_url = 'sqlite:///' + db_filepath.replace('\\', '/')
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
        cleaned_tokens = [lemmatizer.lemmatize(token.lower().strip()) for token in tokens]
    except Exception as e:
        logging.error("Error tokenizing text: %s", e)
        return []

    return cleaned_tokens

def load_model_parameters(file_path):
    try:
        with open(file_path, 'r') as f:
            parameters = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        return None

    if not isinstance(parameters, dict):
        print(f"Expected a dictionary in file: {file_path}, but got {type(parameters)} instead.")
        return None

    # Convert single-item lists to their values and two-item lists to tuples
    for k, v in parameters.items():
        if isinstance(v, list) and len(v) == 2:
            parameters[k] = tuple(v)

    return parameters

def load_grid_search_parameters(file_path):
    try:
        with open(file_path, 'r') as f:
            parameters = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        return None

    if not isinstance(parameters, dict):
        print(f"Expected a dictionary in file: {file_path}, but got {type(parameters)} instead.")
        return None

    # Convert single-item lists to their values and lists of two-item lists to lists of tuples
    for k, v in parameters.items():
        if isinstance(v, list) and all(isinstance(i, list) and len(i) == 2 for i in v):
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
        pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)), # Tokenize and vectorize text
            ('tfidf', TfidfTransformer(smooth_idf=False)), # Apply TF-IDF transformation
            ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=multiprocessing.cpu_count() - 1))) # Use MultiOutputClassifier with RandomForest, n_jobs specifies cores
        ])
    except Exception as e:
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
        pipeline.set_params(clf__estimator__random_state=42, **parameters)
    except Exception as e:
        logging.error("Error building model: %s", e)
        return None

    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate a machine learning model.

    This function predicts the labels for the test data and generates a classification report for each category.
    The results are saved to a CSV file.

    Parameters:
    model (sklearn.base.BaseEstimator): The machine learning model to evaluate.
    X_test (numpy.ndarray): The test features.
    Y_test (numpy.ndarray): The test labels.
    category_names (list of str): The names of the categories.

    If an error occurs during evaluation, no file is saved and a message is printed to the console.
    """
    try:
        # Assuming Y_test and Y_pred are your test labels and predicted labels respectively
        Y_pred = model.predict(X_test)

        # Create an empty list to store the results
        results = []

        for i, col in enumerate(category_names):
            report = classification_report(Y_test[:, i], Y_pred[:, i], output_dict=True, zero_division=0)
            for output_class, metrics in report.items():
                if isinstance(metrics, dict):  # Ensure metrics is a dictionary
                    temp = metrics.copy()  # Create a copy of metrics to avoid modifying the original dictionary
                    temp['output_class'] = output_class
                    temp['category'] = col
                    results.append(temp)

        # Convert the results list to a DataFrame
        results_df = pd.DataFrame(results)

        # Re-arrange the columns so that category and output_class are the first two columns
        results_df = results_df[['category', 'output_class', 'precision', 'recall', 'f1-score', 'support']]

        results_file_path = f'data\\04_fct\\fct_{model}_prediction_results.csv'
        results_df.to_csv(results_file_path, index=False)
        logging.info('Evaluation results saved to: %s', results_file_path)

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
        with open(model_filepath, 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        logging.error("Error saving model: %s", e)

##############################################Grid Search Code######################################################
def estimate_grid_search_runtime(estimator, parameters_file, X_train, y_train):
    
    # Load parameters from JSON file
    with open(parameters_file, 'r') as f:
        parameters = json.load(f)
    # Take the first 100 samples from your training set
    X_train_small = X_train[:100]
    y_train_small = y_train[:100]

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator, parameters)

    # Measure the time before training
    start_time = time()

    # Train the model
    grid_search.fit(X_train_small, y_train_small)

    # Measure the time after training
    end_time = time()

    # Calculate the time it took to train the model
    time_for_small_train = end_time - start_time

    # Estimate the time for the full training set
    estimated_runtime = time_for_small_train * (len(X_train) / 100)

    return estimated_runtime

def run_grid_search(pipeline, parameters_file, X_train, y_train):
    confirmation = input("Are you sure you want proceed with the grid search? (yes/no): ")
    if confirmation.lower() != 'yes':
        logging.info("Grid search cancelled by user.")
        return None

    # Load parameters from JSON file
    with open(parameters_file, 'r') as f:
        parameters = json.load(f)

    start_time = time.time()
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='accuracy', n_jobs=multiprocessing.cpu_count() - 1)
    cv.fit(X_train, y_train)
    end_time = time.time()

    actual_runtime = (end_time - start_time) / 60  # convert seconds to minutes
    logging.info(f"Actual runtime: {actual_runtime} minutes")

    return cv

def save_gs_results(cv, output_file_path):
    cv_results = cv.cv_results_
    results = []

    for params, mean_score in zip(cv_results['params'], cv_results['mean_test_score']):
        results.append({'params': params, 'score': mean_score})

    try:
        with open(output_file_path, 'w') as f:
            json.dump(results, f)
    except FileNotFoundError as e:
        logging.error(f"Error saving results: {e}")

def save_best_parameters(cv, output_file_path):
    best_params = cv.best_params_

    try:
        with open(output_file_path, 'w') as f:
            json.dump(best_params, f)
    except FileNotFoundError as e:
        logging.error(f"Error saving best parameters: {e}")
##############################################End Grid Search Code######################################################

def main():
    """
    Main function to train and save a machine learning model.

    This function performs the following steps:
    1. Load data from a SQLite database.
    2. Split the data into a training set and a test set.
    3. Build a machine learning pipeline.
    4. Train the model using the training data.
    5. Evaluate the model using the test data.
    6. Save the trained model as a pickle file.

    The filepaths for the database and the pickle file are provided as command line arguments.

    Usage:
    python train_classifier.py <database_filepath> <model_filepath>

    Arguments:
    database_filepath (str): Filepath for the SQLite database containing preprocessed data.
    model_filepath (str): Filepath for the output pickle file for the trained model.

    Returns:
    None
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        logging.info('Loading data...\n    DATABASE: %s', database_filepath)
        X, Y = load_data(database_filepath)
        if X is None or Y is None:
            logging.error('Error loading data from database')
            return

        logging.info('Splitting the data into training and test sets...')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        logging.info('Loading base parameters from models\\base_parameters.json...')
        base_parameters = load_model_parameters(r'models\initial_parameters.json')

        if base_parameters is None:
            logging.error("Failed to load base parameters from models\\base_parameters.json. Please ensure the file exists and is a valid JSON file.")
        else:
            logging.info("Base parameters loaded successfully.")
            
        logging.info('Creating pipeline...')
        pipeline = create_pipeline()

        logging.info('Building model from base parameters...')
        base_model = build_model(pipeline, base_parameters)

        logging.info('Training model with base parameters...')
        base_model.fit(X_train, Y_train)

        logging.info('Evaluating model with base parameters...')
        evaluate_model(base_model, X_test, Y_test, TARGET_COLUMNS)

        logging.info('Saving model built from base parameters...\n    MODEL: %s', model_filepath)
        save_model(base_model, model_filepath)

        logging.info('Trained model saved!')

    # Ask the user if they want to run a grid search to improve the model
        if input("Do you want to run a grid search to improve the model? (yes/no): ").lower() == 'yes':
            file_path = r'models\grid_search_parameters.json'
            logging.info(f'Loading GridSearch parameters from {file_path}...')
            grid_search_parameters = load_grid_search_parameters(file_path)
            
            logging.info('Creating pipeline...')
            pipeline = create_pipeline()
            if pipeline is None:
                logging.error('Error creating pipeline')
                return
            
            logging.info('Splitting the data into training and test sets...')
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
            
            # Estimate grid search runtime
            logging.info('Estimating grid search runtime...')

            estimated_runtime = estimate_grid_search_runtime(pipeline, grid_search_parameters, X_train, Y_train)
            logging.info(f'Estimated runtime: {estimated_runtime} minutes')

            # Ask the user if they want to run the grid search
            if input(f"Do you want to run a grid search to improve the model? It is estimated to take approximately {estimated_runtime} minutes. (yes/no): ").lower() == 'yes':
                logging.info('Starting grid search...')
                grid_search = run_grid_search(pipeline, grid_search_parameters, X_train, Y_train)
                logging.info('Grid search complete!')

                gs_results_path = r'models\gs_results.json'
                logging.info(f'Saving grid search results to {gs_results_path}...')
                save_gs_results(grid_search, gs_results_path)

                gs_best_params_path = r'models\gs_best_parameters.json'
                logging.info(f'Saving best parameters to {gs_best_params_path}...')
                save_best_parameters(grid_search, gs_best_params_path)

        # Ask the user if they want to retrain the model using the best parameters found by the grid search
        if input("Do you want to retrain the model using the best parameters found by the grid search? (yes/no): ").lower() == 'yes':
            logging.info(f'Loading best parameters from {gs_best_params_path}...')
            best_parameters = load_model_parameters(r'models\gs_best_parameters.json')

            if best_parameters is not None:
                logging.info('Building model from best parameters...')
                optimized_model = build_model(pipeline, best_parameters)

                logging.info('Retraining model with best parameters...')
                optimized_model.fit(X_train, Y_train)

                logging.info('Evaluating model with best parameters...')
                evaluate_model(optimized_model, X_test, Y_test, TARGET_COLUMNS)

                logging.info('Saving model built from best parameters...\n    MODEL: %s', model_filepath)
                save_model(optimized_model, model_filepath)

                logging.info('Retrained model saved!')
    else:
        logging.info('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()