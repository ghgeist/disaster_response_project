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
import unittest
from unittest.mock import MagicMock, patch
import warnings
warnings.filterwarnings("ignore", message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None'")

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

#Local Application Imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from train_classifier import load_json, load_model_parameters, load_grid_search_parameters, estimate_grid_search_runtime, run_grid_search

class TestTrainClassifier(unittest.TestCase):
    @patch('builtins.open', new_callable=MagicMock)
    @patch('json.load')
    def test_load_json(self, mock_json_load, mock_open):
        # Define the mock return value
        mock_json_load.return_value = {"key": "value"}

        # Call the function
        result = load_json('dummy_file_path')

        # Check the result
        expected_result = {"key": "value"}
        self.assertEqual(result, expected_result)

        # Check that open and json.load were called with the right arguments
        mock_open.assert_called_once_with('dummy_file_path', 'r')
        mock_json_load.assert_called_once()

    @patch('train_classifier.load_json')
    def test_load_model_parameters(self, mock_load_json):
        # Define the mock return value
        mock_load_json.return_value = {
            "param1": [1],
            "param2": [1, 2]
        }

        # Call the function
        result = load_model_parameters('dummy_file_path')

        # Check the result
        expected_result = {
            "param1": 1,
            "param2": (1, 2)
        }
        self.assertEqual(result, expected_result)

        # Check that load_json was called with the right arguments
        mock_load_json.assert_called_once_with('dummy_file_path')

    @patch('train_classifier.load_json')
    def test_load_grid_search_parameters(self, mock_load_json):
        # Define the mock return value
        mock_load_json.return_value = {
            "param1": [1],
            "param2": [[1, 2], [3, 4]]
        }

        # Call the function
        result = load_grid_search_parameters('dummy_file_path')

        # Check the result
        expected_result = {
            "param1": 1,
            "param2": [(1, 2), (3, 4)]
        }
        self.assertEqual(result, expected_result)

        # Check that load_json was called with the right arguments
        mock_load_json.assert_called_once_with('dummy_file_path')
    
    @patch('sklearn.model_selection.GridSearchCV')
    @patch('time.time', side_effect=[1, 2])  # Mocking time to simulate 1 second passing
    def test_estimate_grid_search_runtime(self, mock_time, mock_GridSearchCV):
        # Arrange
        estimator = RandomForestClassifier()
        parameters = {'n_estimators': [50, 100, 200]}
        X_train = np.random.rand(500, 10)  # 500 samples, 10 features each
        y_train = np.random.randint(0, 2, size=(500,))  # Binary labels for 500 samples
        mock_GridSearchCV.return_value.fit.return_value = None

        # Act
        result = estimate_grid_search_runtime(estimator, parameters, X_train, y_train)

        # Assert
        mock_GridSearchCV.assert_called_once_with(estimator, parameters)
        mock_GridSearchCV.return_value.fit.assert_called_once_with(X_train, y_train)
        self.assertEqual(result, 1.0)

    
    # @patch('sklearn.model_selection.GridSearchCV')
    # def test_run_grid_search(self, mock_GridSearchCV):
    #     # Create a mock object to be returned when fit is called
    #     mock_cv = MagicMock(spec=GridSearchCV)
    #     mock_GridSearchCV.return_value = mock_cv

    #     # Create a mock Pipeline object
    #     mock_pipeline = MagicMock(spec=Pipeline)

    #     # Define your inputs
    #     pipeline = mock_pipeline
    #     parameters = load_grid_search_parameters(r'models\grid_search_parameters.json')
    #     X_train =[
    #         ['weather', 'update', 'cold', 'front', 'cuba', 'could', 'pas', 'haiti'],
    #         ['is', 'hurricane'], ['looking', 'someone', 'name'],
    #         ['un', 'report', 'leogane', '8090', 'destroyed', 'only', 'hospital', 'st', 'croix', 'functioning', 'need', 'supply', 'desperately'],
    #         ['say', 'west', 'side', 'haiti', 'rest', 'country', 'today', 'tonight']
    #         ]
    #     y_train = [
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     ]

    #     # Call your function
    #     result = run_grid_search(pipeline, parameters, X_train, y_train)

    #     # Assert that GridSearchCV was called with the right arguments
    #     mock_GridSearchCV.assert_called_once_with(pipeline, param_grid=parameters, scoring='accuracy', n_jobs=-1)

    #     # Assert that fit was called with the right arguments
    #     mock_cv.fit.assert_called_once_with(X_train, y_train)

    #     # Assert that the result is as expected
    #     self.assertEqual(result, mock_cv)


if __name__ == '__main__':
    unittest.main()