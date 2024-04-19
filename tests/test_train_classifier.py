import unittest
from unittest.mock import patch, Mock
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.train_classifier import *

class TestTrainClassifier(unittest.TestCase):

    def test_tokenize(self):
        # Arrange
        text = "This is a sample message with a URL: http://example.com"
        expected_tokens = ['this', 'sample', 'message', 'url', 'urlplaceholder']

        # Act
        tokens = tokenize(text)

        # Assert
        self.assertEqual(tokens, expected_tokens)



if __name__ == '__main__':
    unittest.main()