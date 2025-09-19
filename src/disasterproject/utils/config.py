"""
Configuration constants and settings for the disaster response classification system.

Focus areas:
- Centralized constants used across data, models, and evaluation
- Lightweight imports and safe initialization (no heavy side effects at import)
- Idempotent logging setup
"""

import logging
import sys
from pathlib import Path

import numpy as np
from nltk.corpus import stopwords

# NLTK resources are now managed by app/nltk_setup.py during application startup
# This prevents per-request downloads and improves performance
# The resources are validated and loaded once at startup instead of on every import

logger = logging.getLogger(__name__)

# Resource Management Configuration
# These constants are used primarily by hyperparameter_search.py but centralized here
# for system-wide consistency and easy adjustment
MEMORY_LIMIT_GB = 12
MEMORY_WARNING_GB = 10
MIN_AVAILABLE_MEMORY_GB = 2.0

# Hyperparameter Search Configuration
DEFAULT_N_ITER = 20
SEARCH_N_JOBS = 2  # Parallelism for hyperparameter search
DEFAULT_CV_SPLITS = 3
ESTIMATION_CV_SPLITS = 2
ESTIMATION_MAX_ITER = 5
ESTIMATION_SUBSET_SIZE = 100

# Random State Configuration
RANDOM_STATE = 42

# Machine Learning Pipeline Configuration
# Used across scripts for consistent train/test splitting and model behavior
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_SEED = 42  # Alias for RANDOM_STATE for script arguments
RF_N_JOBS = 1  # Conservative default for RF estimators to prevent CPU oversubscription


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

    file_handler = logging.FileHandler("app.log", encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    # Handle Windows console encoding issues
    if hasattr(console_handler.stream, "reconfigure"):
        try:
            console_handler.stream.reconfigure(encoding="utf-8")
        except Exception:
            pass  # Fallback for older Python versions or restricted environments
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

URL_REGEX = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
URL_PLACE_HOLDER = "urlplaceholder"

# File paths
# ``SCRIPT_DIR`` historically pointed at the repository root even though the
# name implies the package directory. Keep a string form for backwards
# compatibility while introducing more explicit Path-based constants for new
# call sites.
SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SRC_ROOT.parent
SCRIPT_DIR = str(PROJECT_ROOT)
FCT_DIR = PROJECT_ROOT / "data" / "04_fct"
BASE_METRICS_PATH = FCT_DIR / "fct_median_metrics_by_output_class_base.csv"
OPT_METRICS_PATH = FCT_DIR / "fct_median_metrics_by_output_class_optimized.csv"
MODEL_DIR = PROJECT_ROOT / "model"
BASE_PARAMETERS = MODEL_DIR / "base_parameters.json"
HYPERPARAMETER_OPTIMIZATION = MODEL_DIR / "hyperparameter_optimization.json"
GRID_SEARCH_RESULTS = MODEL_DIR / "gs_results.json"
OPTIMIZED_PARAMETERS = MODEL_DIR / "optimized_parameters.json"

# Hierarchy Configuration
# Label taxonomy (parent -> children) for enforcing hierarchical consistency
TAXONOMY = {
    "aid_related": [
        "medical_help",
        "medical_products",
        "search_and_rescue",
        "water",
        "food",
        "shelter",
        "clothing",
        "money",
        "other_aid",
    ],
    "infrastructure_related": [
        "transport",
        "buildings",
        "electricity",
        "tools",
        "hospitals",
        "shops",
        "aid_centers",
        "other_infrastructure",
    ],
    # Keep weather strictly weather; treat earthquake/fire as independent
    "weather_related": ["floods", "storm", "cold", "other_weather"],
    # Treat "related" as a root with siblings; apply child→parent at decision level only
    "related": ["request", "offer", "direct_report"],
}

# Critical leaves (use softer thresholds in the fixer)
CRITICAL_LABELS = {
    "medical_help",
    "medical_products",
    "search_and_rescue",
    "water",
    "food",
    "security",
}

# Labels excluded from hierarchy constraints (documented data limitations)
EXCLUDE_FROM_CONSTRAINTS = {"child_alone"}  # 0 positives in source + train data

# Hierarchy threshold behavior
# Default reduction applied to critical labels during hierarchy decisioning.
# Kept configurable for clarity in experiments; 0.0 per latest evaluation.
HIERARCHY_CRITICAL_THRESHOLD_REDUCTION = 0.0

# Model Promotion Configuration
# Performance thresholds for model validation during promotion
PERFORMANCE_THRESHOLDS = {
    'min_f1_weighted': 0.5,  # Minimum weighted F1 score for promotion
    'min_f1_micro': 0.6,     # Minimum micro F1 score for promotion
    'max_model_size_mb': 1000  # Maximum model size in MB for production
}

# Set random seed for reproducibility without noisy import-time logs
np.random.seed(0)
