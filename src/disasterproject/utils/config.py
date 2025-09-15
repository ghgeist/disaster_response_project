"""
Configuration constants and settings for the disaster response classification system.
"""

import os
import logging
import sys
import numpy as np
import nltk
from nltk.corpus import stopwords

# NLTK resources are now managed by app/nltk_setup.py during application startup
# This prevents per-request downloads and improves performance
# The resources are validated and loaded once at startup instead of on every import

# Set up logging
def setup_logging():
    """Set up logging configuration with file and console handlers."""
    # File handler with detailed format
    file_formatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    )
    # Console handler with clean format (no timestamps/thread info)
    console_formatter = logging.Formatter("%(message)s")

    root_logger = logging.getLogger()

    # File handler for detailed logging
    file_handler = logging.FileHandler("app.log")
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console handler for clean output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    root_logger.setLevel(logging.INFO)

# Data configuration
FEATURE_COLUMNS = ["message"]
TARGET_COLUMNS = [
    "related",
    "request",
    "offer",
    "aid_related",
    "medical_help",
    "medical_products",
    "search_and_rescue",
    "security",
    "military",
    "child_alone",
    "water",
    "food",
    "shelter",
    "clothing",
    "money",
    "missing_people",
    "refugees",
    "death",
    "other_aid",
    "infrastructure_related",
    "transport",
    "buildings",
    "electricity",
    "tools",
    "hospitals",
    "shops",
    "aid_centers",
    "other_infrastructure",
    "weather_related",
    "floods",
    "storm",
    "fire",
    "earthquake",
    "cold",
    "other_weather",
    "direct_report",
]

# Text processing configuration
STOPWORDS_SET = set(stopwords.words("english"))
URL_REGEX = (
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)
URL_PLACE_HOLDER = "urlplaceholder"

# File paths
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
BASE_PARAMETERS = os.path.join(SCRIPT_DIR, "models", "base_parameters.json")
HYPERPARAMETER_OPTIMIZATION = os.path.join(SCRIPT_DIR, "models", "hyperparameter_optimization.json")
GRID_SEARCH_RESULTS = os.path.join(SCRIPT_DIR, "models", "gs_results.json")
OPTIMIZED_PARAMETERS = os.path.join(SCRIPT_DIR, "models", "optimized_parameters.json")

# Set random seed for reproducibility
logging.info("Setting random seed...")
np.random.seed(0)
