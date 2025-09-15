"""
Configuration constants and settings for the disaster response classification system.

Focus areas:
- Centralized constants used across data, models, and evaluation
- Lightweight imports and safe initialization (no heavy side effects at import)
- Idempotent logging setup
"""

import os
import logging
import sys
import numpy as np
from nltk.corpus import stopwords

# NLTK resources are now managed by app/nltk_setup.py during application startup
# This prevents per-request downloads and improves performance
# The resources are validated and loaded once at startup instead of on every import

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging with file and console handlers (idempotent).

    Adds a detailed file handler and a clean console handler if not already present,
    and sets the root logging level to INFO.
    """
    root_logger = logging.getLogger()

    # Avoid duplicate handlers if setup_logging is called multiple times
    if getattr(root_logger, "_disasterproject_logging_configured", False):
        return

    file_formatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    )
    console_formatter = logging.Formatter("%(message)s")

    file_handler = logging.FileHandler("app.log")
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    root_logger.setLevel(logging.INFO)
    # Mark as configured to prevent duplicates
    setattr(root_logger, "_disasterproject_logging_configured", True)

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
# Load stopwords defensively to avoid hard crashes when NLTK resources
# aren't available (e.g., fresh environments before `nltk_setup`).
try:  # Prefer full set when resources are available
    STOPWORDS_SET = set(stopwords.words("english"))
except Exception as exc:  # LookupError, OSError, etc.
    logger.warning("NLTK stopwords unavailable (%s); using empty fallback set", exc)
    STOPWORDS_SET = set()

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

# Set random seed for reproducibility without noisy import-time logs
np.random.seed(0)
