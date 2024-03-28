#Import libraries
import pandas as pd
import numpy as np
import string
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pandas as pd

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

def tokenize(text):
    """
    This function is designed to tokenize the message data
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
