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

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Download necessary nltk packages
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

FEATURE_COLUMNS =['message']
TARGET_COLUMNS = [
    'related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 
    'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food', 
    'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 
    'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 
    'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
    'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 
    'other_weather', 'direct_report']

URL_REGEX = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
URL_PLACE_HOLDER = "urlplaceholder"


def load_data(db_filepath, table_name, feature_columns, target_columns):
    try:
        engine = create_engine(db_filepath)
        df = pd.read_sql_table(table_name, engine) 
        if df.empty:
            print(f"The table '{table_name}' is empty or does not exist.")
            return None, None
    except SQLAlchemyError as e:
        print(f"Error loading data from database: {e}")
        return None, None

    try:
        X = df[feature_columns].copy().values
        Y = df[target_columns].copy().values
        print(f"The data from '{table_name}' has been successfully loaded.")
    except (KeyError, AttributeError) as e:
        print(f"Error extracting features/targets: {e}")
        return None, None

    return X, Y


def tokenize(text):
    detected_urls = re.findall(URL_REGEX, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def display_results(y_test, y_pred):
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)


def main():
    X, Y = load_data('sqlite:///data/02_stg//stg_disaster_response.db', 'stg_disaster_response', FEATURE_COLUMNS, TARGET_COLUMNS)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier()),
    ])

    pipeline.fit(X_train, Y_train)
    Y_pred = pipeline.predict(X_test)

    display_results(Y_test, Y_pred)
    
main()
