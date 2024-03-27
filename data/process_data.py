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
URL_REGEX = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
URL_PLACE_HOLDER = "urlplaceholder"
STOPWORDS_SET = set(stopwords.words("english"))
FEATURE_COLUMNS =['message']
TARGET_COLUMNS = [
    'related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 
    'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food', 
    'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 
    'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 
    'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
    'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 
    'other_weather', 'direct_report']


def load_data_from_db(db_filepath, table_name, feature_columns, target_columns):
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
        # Define the features and target variables X and y
        x = df[feature_columns].copy().values
        y = df[target_columns].copy().values
        print(f"The data from '{table_name}' has been successfully loaded.")   
    except (KeyError, AttributeError) as e:
        print(f"Error extracting features/targets: {e}")
        return None, None

    return x, y

def detect_and_replace_urls(text_array):
    """
    Detects URLs in each element of the text array and replaces them with a placeholder.
    
    :param text_array: numpy array of strings containing the original text.
    :return: numpy array of text with URLs replaced by placeholder.
    """
    if not isinstance(text_array, np.ndarray):
        raise ValueError("Input must be a numpy array")
    vectorized_replace = np.vectorize(lambda text: re.sub(URL_REGEX, URL_PLACE_HOLDER, text))
    print(f"Replaced URLs in text (if any) with the following placeholder: {URL_PLACE_HOLDER}")
    return vectorized_replace(text_array)

def remove_punctuation(text_array):
    """
    Removes punctuation from the input numpy array of strings.
    
    :param text_array: Numpy array containing the original texts.
    :return: Numpy array with punctuation removed from each text.
    """
    if not isinstance(text_array, np.ndarray):
        raise ValueError("Input must be a numpy array")

    vfunc = np.vectorize(
        lambda text: text.translate(str.maketrans("", "", string.punctuation)) 
        if isinstance(text, str) else None)
    try:
        result = vfunc(text_array)
        print("All punctuation has been successfully removed.")
        return result
    except TypeError as e:
        print(f"Error removing punctuation: {e}")
        return None

def tokenize_text(text, encoding='utf-8'):
    """
    Tokenizes the input text.

    :param text: The text to tokenize. Can be a string or a numpy array of strings.
    :param encoding: The encoding to use if the text is bytes. Default is 'utf-8'.
    :return: A numpy array of tokens if the input is a string, or a numpy array of arrays of tokens if the input is a numpy array.
    """
    if isinstance(text, np.ndarray):
        result = np.array(
            [word_tokenize(t.decode(encoding)) if isinstance(t, bytes) 
                else word_tokenize(t) 
                for t in text.flatten()], 
            dtype=object)
        print("All text has been tokenized.")
        return result
    elif isinstance(text, str) or isinstance(text, bytes):
        result = np.array(
            word_tokenize(text.decode(encoding)) if isinstance(text, bytes) 
            else word_tokenize(text), 
            dtype=object)
        print("All text has been tokenized.")
        return result
    else:
        raise ValueError("Input must be a string or a numpy array of strings")

def remove_stop_words(words_array, stopwords_set=STOPWORDS_SET):
    """
    Removes English stopwords from the numpy array of words.
    
    :param words_array: Numpy array of words. Each element of the array should be a list of words.
    :param stopwords_set: Set of stopwords to remove. Default is English stopwords.
    :return: List of lists of words with stopwords removed. Each inner list corresponds to a sentence.
    """
    if not isinstance(words_array, np.ndarray):
        raise ValueError("Input must be a numpy array of words")

    result = [[word for word in words if word not in stopwords_set] for words in words_array]
    print("Stop words have been removed.")
    return result

def clean_tokens_generator(tokens):
    """
    Cleans and normalizes tokens using lemmatization. Uses a generator to yield tokens one at a time.
    :param tokens: Iterable of tokens to clean.
    :yield: Cleaned token one at a time.
    """
    if not isinstance(tokens, Iterable):
        raise ValueError("Input must be an iterable of tokens")

    lemmatizer = WordNetLemmatizer()
    for token in tokens:
        yield lemmatizer.lemmatize(token.lower().strip())   

def tokenize(text_array):
    """
    Processes an array of text: tokenizes, cleans, and normalizes.

    This function performs the following steps:
    1. Detects and replaces URLs in the text.
    2. Removes punctuation from the text.
    3. Tokenizes the text into words.
    4. Removes English stopwords from the tokens.
    5. Cleans and normalizes the tokens using lemmatization.

    :param text_array: Numpy array of text to process. Each element of the array should be a string.
    :return: List of lists of processed tokens. Each inner list corresponds to a sentence.
    """
    if not isinstance(text_array, np.ndarray):
        raise ValueError("Input must be a numpy array of strings")

    text_array = detect_and_replace_urls(text_array)
    text_array = remove_punctuation(text_array)
    tokenized_text = tokenize_text(text_array)
    tokenized_text = remove_stop_words(tokenized_text)
    cleaned_tokens = [list(clean_tokens_generator(tokens)) for tokens in tokenized_text]
    print("Tokens have been processed.")
    return cleaned_tokens

def main():
    # Load the data
    x, y = load_data_from_db(
        'sqlite:///data/02_stg//stg_disaster_response.db',
        'stg_disaster_response',
        FEATURE_COLUMNS, 
        TARGET_COLUMNS
    )

    # Process the data
    x = tokenize(x)
    print("Data processing complete!")

if __name__ == "__main__":
    main()
