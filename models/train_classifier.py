# Standard library imports
import re
import string
from collections.abc import Iterable

# Related third party imports
import nltk
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

# Local application/library specific imports
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary nltk packages
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Constants
STOPWORDS_SET = set(stopwords.words('english'))
URL_REGEX = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
URL_PLACE_HOLDER = "urlplaceholder"
FEATURE_COLUMNS =['message']
TARGET_COLUMNS = [
    'related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 
    'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food', 
    'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 
    'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 
    'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
    'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 
    'other_weather', 'direct_report']


def load_data(db_filepath, table_name, feature_columns, target_columns):
    """
    Load data from a database and return feature and target arrays.

    Parameters:
    db_filepath (str): The file path to the database.
    table_name (str): The name of the table to load data from.
    feature_columns (list of str): The names of the feature columns.
    target_columns (list of str): The names of the target columns.

    Returns:
    tuple: A tuple containing the feature array and the target array. 
           If an error occurs, returns (None, None).
    """
    try:
        # Load data from database
        engine = create_engine(db_filepath)

        # Create a dataframe from the engine
        df = pd.read_sql_table(table_name, engine) 
        if df.empty:
            print(f"The table '{table_name}' is empty or does not exist.")
            return None, None
    except SQLAlchemyError as e:
        print(f"Error loading data from database: {e}")
        return None, None

    try:
        x = df[feature_columns].copy().values
        y = df[target_columns].copy().values
        print(f"The data from '{table_name}' has been successfully loaded.")
        #To Do
        # - Add some more print statements that describe what's being loaded
    except (KeyError, AttributeError) as e:
        print(f"Error extracting features/targets: {e}")
        return None, None

    return x, y

def tokenize(text):
    """
    Tokenizes the input text by performing the following steps:
    1. Replaces URLs with a placeholder.
    2. Removes punctuation.
    3. Tokenizes the text into individual words.
    4. Removes stop words.
    5. Lemmatizes the tokens.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        list: A list of cleaned and lemmatized tokens.
    """
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
    return cleaned_tokens
