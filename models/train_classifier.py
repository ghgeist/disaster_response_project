# Standard library imports
import logging
import multiprocessing
import os
import pickle
import re
import string
import sys

# Third-party imports
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError

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

def load_data(db_filepath):
    """
    Load data from a SQLite database and return features and labels for machine learning.

    The table name is assumed to be the same as the file name (without the extension).

    Parameters:
    db_filepath (str): The file path of the SQLite database.

    Returns:
    X (numpy.ndarray): The features for the machine learning model.
    y (numpy.ndarray): The labels for the machine learning model.

    If an error occurs while loading the data, both X and y will be None.
    """
    try:
        # Create a valid SQLAlchemy URL for a SQLite database
        database_url = 'sqlite:///' + db_filepath.replace('\\', '/')
        engine = create_engine(database_url)
    except OperationalError:
        logging.error("Error connecting to database at %s", db_filepath)
        return None, None

    # Extract table name from the file name
    table_name = os.path.splitext(os.path.basename(db_filepath))[0]

    try:
        # Create a dataframe from the engine
        df = pd.read_sql_table(table_name, engine)
    except ValueError:
        logging.error("Table %s not found in database", table_name)
        return None, None

    try:
        X = df.message.values
        y = df[TARGET_COLUMNS].values

        # For debugging
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

def build_model():
    """
    Build a machine learning pipeline.

    This function builds a pipeline that first vectorizes the text data using CountVectorizer and a custom tokenizer,
    then applies a TF-IDF transformation, and finally uses a multi-output classifier with a random forest classifier.

    Returns:
    pipeline (sklearn.pipeline.Pipeline): The machine learning pipeline.

    If an error occurs while building the pipeline, None is returned.
    """
    try:
        # Instantiate and configure the pipeline
        pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)), # Tokenize and vectorize text
            ('tfidf', TfidfTransformer(smooth_idf=False)), # Apply TF-IDF transformation
            ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=multiprocessing.cpu_count() - 1))) # Use MultiOutputClassifier with RandomForest, n_jobs specifies cores
        ])
    except Exception as e:
        print("Error building model: %s", e)
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
        results_file_path = 'data\\04_fct\\fct_prediction_results.csv'
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
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        logging.info('Building model...')
        model = build_model()

        logging.info('Training model...')
        model.fit(X_train, Y_train)

        logging.info('Evaluating model...')
        evaluate_model(model, X_test, Y_test, TARGET_COLUMNS)

        logging.info('Saving model...\n    MODEL: %s', model_filepath)
        save_model(model, model_filepath)

        logging.info('Trained model saved!')
    else:
        logging.info('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()