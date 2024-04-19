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

consoleHandler = logging.StreamHandler(sys.stdout)
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

BASE_PARAMETERS = r'models\base_parameters.json'
GRID_SEARCH_PARAMETERS = r'models\grid_search_parameters.json'
GRID_SEARCH_RESULTS = r'models\gs_results.json'
OPTIMIZED_PARAMETERS = r'models\optimized_parameters.json'

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

def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        return None

    if not isinstance(data, dict):
        print(f"Expected a dictionary in file: {file_path}, but got {type(data)} instead.")
        return None

    return data

def load_model_parameters(file_path):
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

def evaluate_model(model, model_name, X_test, Y_test, category_names):
    try:
        Y_pred = model.predict(X_test)
        results = []

        for i, col in enumerate(category_names):
            report = classification_report(Y_test[:, i], Y_pred[:, i], output_dict=True, zero_division=0)
            for output_class, metrics in report.items():
                if isinstance(metrics, dict):
                    temp = metrics.copy()
                    temp['output_class'] = output_class
                    temp['category'] = col
                    results.append(temp)

        results_df = pd.DataFrame(results)
        results_df = results_df[['category', 'output_class', 'precision', 'recall', 'f1-score', 'support']]

        results_file_path = f'data/04_fct/fct_{model_name}_prediction_results.csv'
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

##############################################Grid Search Code#####################################################
def run_grid_search(pipeline, parameters, X_train, y_train, use_small_subset=False):
    start_time = time()
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='accuracy', n_jobs=multiprocessing.cpu_count() - 1, verbose=1)

    if use_small_subset:
        X_train_size = len(X_train)
        X_train = X_train[:100]
        y_train = y_train[:100]
        cv.fit(X_train, y_train)
        end_time = time()
        runtime = (end_time - start_time) * (X_train_size / 100)  # keep the time in seconds
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
def get_user_input(prompt):
    while True:
        user_input = input(prompt)
        if user_input.lower() in ['yes', 'no', 'exit']:
            return user_input.lower()
        else:
            print("Invalid input. Please enter 'yes', 'no', or 'exit'.")
            
def main():    
    if len(sys.argv) != 3:
        logging.info('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
    else:
        database_filepath, model_filepath = sys.argv[1:]
    
        logging.info('Loading data...\n    DATABASE: %s', database_filepath)
        X, Y = load_data(database_filepath)
        if X is None or Y is None:
            logging.error('Error loading data from database')
            return
        
        logging.info('Splitting the data into training and test sets...')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        logging.info('Creating pipeline...')
        pipeline = create_pipeline()
        
        retrain_base_model = get_user_input("Do you want to retrain the base model? (yes/no/exit): ")
        if retrain_base_model == 'exit':
            sys.exit()
        elif retrain_base_model == 'yes':
            logging.info('Loading base parameters from models\\base_parameters.json...')
            base_parameters = load_model_parameters(BASE_PARAMETERS)
            if base_parameters is None:
                logging.error(f"Failed to load base parameters from {BASE_PARAMETERS}. Please ensure the file exists and is a valid JSON file.")
            else:
                logging.info(f"Base parameters loaded successfully from {BASE_PARAMETERS} .")
            
            logging.info('Building model from base parameters...')
            base_model = build_model(pipeline, base_parameters)

            logging.info('Training model with base parameters...')
            base_model.fit(X_train, Y_train)

            logging.info('Evaluating model with base parameters...')
            evaluate_model(base_model,'base_model', X_test, Y_test, TARGET_COLUMNS)

            logging.info('Saving model built from base parameters...\n    MODEL: %s', model_filepath)
            save_model(base_model, model_filepath)

            logging.info('Trained model saved!')
        
        estimate_runtime = get_user_input("Do you want to estimate the grid search runtime? (yes/no/exit): ")
        if estimate_runtime == 'exit':
            sys.exit()
        elif estimate_runtime == 'yes':
            logging.info(f'Loading GridSearch parameters from {GRID_SEARCH_PARAMETERS}...')
            grid_search_parameters = load_grid_search_parameters(GRID_SEARCH_PARAMETERS)
            logging.info('Estimating grid search runtime...')
            estimated_grid_search = run_grid_search(pipeline, grid_search_parameters, X_train, Y_train, use_small_subset=True)
            logging.info('Grid search runtime estimate complete!')

        do_grid_search = get_user_input("Do you want to run a grid search? (yes/no/exit): ")
        if do_grid_search == 'exit':
            sys.exit()
        elif do_grid_search == 'yes':
            logging.info('Starting grid search...')
            grid_search_parameters = load_grid_search_parameters(GRID_SEARCH_PARAMETERS)
            grid_search = run_grid_search(pipeline, grid_search_parameters, X_train, Y_train, use_small_subset=False)
            logging.info('Grid search complete!')
            logging.info(f'Saving grid search results to {GRID_SEARCH_RESULTS}...')
            save_gs_results(grid_search, GRID_SEARCH_RESULTS)
            logging.info('Grid search results saved!')
            save_best_parameters(grid_search, OPTIMIZED_PARAMETERS)
            logging.info('Optimized parameters saved!')

        retrain_optimized_model = get_user_input("Do you want to retrain the model using the optimized parameters found by the grid search? (yes/no/exit): ")
        if retrain_optimized_model == 'exit':
            sys.exit()
        elif retrain_optimized_model == 'yes':
            logging.info('Building model from optimized parameters...')
            optimized_parameters = load_model_parameters(OPTIMIZED_PARAMETERS)
            if optimized_parameters is None:
                logging.error(f"Failed to load optimized parameters from {OPTIMIZED_PARAMETERS}. Please ensure the file exists and is a valid JSON file.")
            else:
                logging.info(f"Optimized parameters loaded successfully from {OPTIMIZED_PARAMETERS}.")
            logging.info('Building model from optimized parameters...')
            optimized_model = build_model(pipeline, optimized_parameters)
            logging.info('Training model with optimized parameters...')
            optimized_model.fit(X_train, Y_train)
            logging.info('Evaluating model with optimized parameters...')
            evaluate_model(optimized_model, 'optimized_model', X_test, Y_test, TARGET_COLUMNS)
            logging.info('Saving model built from optimized parameters...\n    MODEL: %s', model_filepath)
            save_model(optimized_model, model_filepath)
            logging.info('Trained model saved!')

if __name__ == '__main__':
    main()