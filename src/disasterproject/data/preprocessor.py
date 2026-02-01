"""Text preprocessing utilities for disaster response classification."""
import logging
import re
import string

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from ..utils.config import STOPWORDS_SET, URL_PLACE_HOLDER, URL_REGEX


def _normalize_negation_contractions(text: str) -> str:
    """
    Expand common English negation contractions to explicit forms prior to tokenization.
    Examples: can't -> can not, won't -> will not, isn't -> is not
    """
    # Work on lowercase to simplify patterns
    lowered = text.lower()
    # Specific exceptions first
    replacements = {
        "won't": "will not",
        "can't": "can not",
    }
    for k, v in replacements.items():
        lowered = lowered.replace(k, v)
    # Generic n't -> not (e.g., isn't -> is not)
    lowered = re.sub(r"n\'t\b", " not", lowered)
    # Return with original casing irrelevant because we lowercase during lemmatization
    return lowered


def tokenize(text):
    """
    Tokenize with disaster-aware stopword filtering.

    This function detects and replaces URLs, removes punctuation, tokenizes the text,
    removes stop words while preserving disaster-critical words, and lemmatizes the tokens.

    Parameters:
    text (str): The text to be tokenized.

    Returns:
    cleaned_tokens (list of str): The tokenized and cleaned text.

    If an error occurs during tokenization, an empty list is returned.
    """
    try:
        # Detect and replace URLs
        text = re.sub(URL_REGEX, URL_PLACE_HOLDER, text)
        # Normalize negation contractions before removing punctuation
        text = _normalize_negation_contractions(text)
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Tokenize text
        tokens = word_tokenize(text)

        # DISASTER-AWARE stopword removal
        disaster_critical = {
            'me', 'us', 'we', 'i', 'my', 'our', 'help', 'please', 'save', 'rescue',
            # Negations and related
            'no', 'not', 'never', 'none', 'without', 'nor'
        }
        tokens = [token for token in tokens
                 if token.lower() not in STOPWORDS_SET or token.lower() in disaster_critical]

        # Lemmatize tokens
        lemmatizer = WordNetLemmatizer()
        cleaned_tokens = [
            lemmatizer.lemmatize(token.lower().strip()) for token in tokens
        ]
    except Exception as e:
        logging.error("Error tokenizing text: %s", e)
        return []

    return cleaned_tokens
