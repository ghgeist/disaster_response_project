"""
Hyperparameter optimization pipeline with resource management and monitoring.

This module provides functionality for running parameter searches with proper
resource monitoring, memory management, and progress tracking.
"""

import logging
from time import time

import psutil
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV

# Import centralized configuration constants
from disasterproject.utils.config import (
    DEFAULT_CV_SPLITS,
    DEFAULT_N_ITER,
    ESTIMATION_CV_SPLITS,
    ESTIMATION_MAX_ITER,
    ESTIMATION_SUBSET_SIZE,
    MEMORY_WARNING_GB,
    MIN_AVAILABLE_MEMORY_GB,
    RANDOM_STATE,
    SEARCH_N_JOBS,
)

logger = logging.getLogger(__name__)


def create_scorers_from_config(optimization_config):
    """
    Create sklearn scorer objects from optimization configuration.

    Args:
        optimization_config (dict): Configuration containing scoring metrics

    Returns:
        tuple: (scoring_dict, refit_metric)

    Example config:
        {
            "scoring": {
                "f1_weighted": {"scorer_type": "f1_score", "average": "weighted", "zero_division": 0},
                "accuracy": "accuracy"
            },
            "refit_metric": "f1_weighted"
        }
    """
    if not optimization_config or "scoring" not in optimization_config:
        raise ValueError("optimization_config must contain 'scoring' section")

    scoring_dict = {}
    scorer_functions = {
        "f1_score": f1_score,
        "accuracy_score": accuracy_score,
        "precision_score": precision_score,
        "recall_score": recall_score
    }

    for metric_name, metric_config in optimization_config["scoring"].items():
        if isinstance(metric_config, str):
            # Built-in sklearn scorer (e.g., "accuracy")
            scoring_dict[metric_name] = metric_config
        elif isinstance(metric_config, dict):
            # Custom scorer configuration
            scorer_type = metric_config["scorer_type"]
            if scorer_type not in scorer_functions:
                raise ValueError(f"Unknown scorer_type: {scorer_type}")

            scorer_func = scorer_functions[scorer_type]
            scorer_kwargs = {k: v for k, v in metric_config.items() if k != "scorer_type"}
            scoring_dict[metric_name] = make_scorer(scorer_func, **scorer_kwargs)
        else:
            raise ValueError(f"Invalid metric config for {metric_name}: {metric_config}")

    refit_metric = optimization_config.get("refit_metric")
    if refit_metric and refit_metric not in scoring_dict:
        raise ValueError(f"refit_metric '{refit_metric}' not found in scoring configuration")

    return scoring_dict, refit_metric


def run_parameter_search(pipeline, parameters, X_train, y_train, use_small_subset=False, optimization_config=None):
    """
    Run a randomized search to find the best parameters for a pipeline with resource management.

    This function uses RandomizedSearchCV to find the best parameters for the specified pipeline using the provided training data.
    The function includes resource monitoring, timeout protection, and progress tracking.
    If use_small_subset is True, the function uses only the first 100 samples of the training data and estimates the total runtime based on this subset.

    Args:
    pipeline (Pipeline): The pipeline for which to find the best parameters.
    parameters (dict): The parameters to try in the search.
    X_train (numpy.ndarray): The features for the training data.
    y_train (numpy.ndarray): The labels for the training data.
    use_small_subset (bool, optional): Whether to use only the first 100 samples of the training data. Defaults to False.
    optimization_config (dict, optional): Configuration for scoring metrics and refit strategy. If None, uses default F1 weighted.

    Returns:
    cv (RandomizedSearchCV): The fitted RandomizedSearchCV instance.

    Raises:
    ValueError: If input parameters are invalid.
    RuntimeError: If insufficient memory is available.
    """
    # Validate inputs
    if pipeline is None or parameters is None:
        raise ValueError("Pipeline and parameters cannot be None")
    if X_train is None or y_train is None or len(X_train) == 0:
        raise ValueError("Training data cannot be None or empty")

    start_time = time()

    # Check system resources before starting
    initial_memory = psutil.Process().memory_info().rss / (1024**3)
    available_memory = psutil.virtual_memory().available / (1024**3)

    logger.info("Initial memory usage: %.1fGB", initial_memory)
    logger.info("Available system memory: %.1fGB", available_memory)

    if available_memory < MIN_AVAILABLE_MEMORY_GB:
        raise RuntimeError(
            f"Insufficient memory available: {available_memory:.1f}GB. "
            f"Need at least {MIN_AVAILABLE_MEMORY_GB}GB."
        )

    # Use MultilabelStratifiedKFold for proper multi-label cross-validation
    cv_strategy = MultilabelStratifiedKFold(
        n_splits=DEFAULT_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )

    # Configure scoring metrics based on optimization config or use defaults
    if optimization_config:
        scoring_dict, refit_metric = create_scorers_from_config(optimization_config)
        logger.info("Using configured optimization metrics: %s", list(scoring_dict.keys()))
        logger.info("Refit metric: %s", refit_metric)
    else:
        # Default to original hardcoded behavior for backward compatibility
        f1_weighted_scorer = make_scorer(f1_score, average="weighted", zero_division=0)
        f1_micro_scorer = make_scorer(f1_score, average="micro", zero_division=0)
        scoring_dict = {"f1_weighted": f1_weighted_scorer, "f1_micro": f1_micro_scorer}
        refit_metric = "f1_weighted"
        logger.info("Using default optimization metrics (backward compatibility)")

    # Configure RandomizedSearchCV with conservative resource settings
    cv = RandomizedSearchCV(
        pipeline,
        param_distributions=parameters,
        n_iter=DEFAULT_N_ITER,
        scoring=scoring_dict,
        refit=refit_metric,
        cv=cv_strategy,
        n_jobs=SEARCH_N_JOBS,  # Parallelism for hyperparameter search
        verbose=2,  # Detailed progress output
        random_state=RANDOM_STATE
    )

    if use_small_subset:
        print("\n🔍 RUNTIME ESTIMATION MODE")
        print("=" * 50)
        X_train_size = len(X_train)
        print(f"📊 Full dataset size: {X_train_size:,} samples")
        print(f"🔬 Using subset: {ESTIMATION_SUBSET_SIZE} samples ({ESTIMATION_SUBSET_SIZE/X_train_size*100:.1f}%)")

        X_train = X_train[:ESTIMATION_SUBSET_SIZE]
        y_train = y_train[:ESTIMATION_SUBSET_SIZE]

        # Memory check before fitting
        current_memory = psutil.Process().memory_info().rss / (1024**3)
        if current_memory > MEMORY_WARNING_GB:
            logger.warning("Memory usage %.1fGB exceeds warning threshold", current_memory)

        # Configure for faster estimation
        _configure_estimation_mode(cv)

        print("⚙️  Estimation settings:")
        print(f"   • CV folds: {ESTIMATION_CV_SPLITS}")
        print(f"   • Parameter trials: {cv.n_iter}")
        print(f"   • Total CV fits: {ESTIMATION_CV_SPLITS * cv.n_iter}")
        print("\n🚀 Starting estimation...")

        cv.fit(X_train, y_train)
        end_time = time()

        subset_time = end_time - start_time

        # Extrapolate runtime estimate
        full_runtime = subset_time * (X_train_size / ESTIMATION_SUBSET_SIZE)
        hours, remainder = divmod(full_runtime, 3600)
        minutes, seconds = divmod(remainder, 60)

        print("\n📈 ESTIMATION RESULTS:")
        print("=" * 50)
        print(f"⏱️  Subset completed in: {subset_time:.1f} seconds")
        print(f"🎯 Best score found: {cv.best_score_:.4f}")
        print(f"🏆 Best parameters: {cv.best_params_}")
        print("\n⚡ FULL SEARCH ESTIMATE:")
        print(f"   📅 Expected runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"   🔢 Full parameter trials: {DEFAULT_N_ITER}")
        print(f"   📊 Full CV folds: {DEFAULT_CV_SPLITS}")
        print(f"   🔄 Total CV fits: {DEFAULT_CV_SPLITS * DEFAULT_N_ITER}")

        if hours > 4:
            print("\n⚠️  WARNING: Estimated runtime > 4 hours")
            print("   Consider reducing n_iter or using smaller parameter grid")
        elif hours < 0.5:
            print("\n✅ Good news: Fast runtime expected!")

        print("=" * 50)
    else:
        logger.info("Starting hyperparameter search with resource monitoring")
        logger.info("Configuration: n_iter=%d, n_jobs=%d", DEFAULT_N_ITER, SEARCH_N_JOBS)

        # Configure pipeline to prevent CPU oversubscription
        _configure_production_mode(cv)

        # Execute search with error handling
        try:
            logger.info("Starting RandomizedSearchCV.fit()")
            cv.fit(X_train, y_train)
        except Exception as e:
            current_memory = psutil.Process().memory_info().rss / (1024**3)
            logger.error("Search failed with error: %s", e)
            logger.error("Memory usage at failure: %.1fGB", current_memory)
            raise

        # Report results
        end_time = time()
        final_memory = psutil.Process().memory_info().rss / (1024**3)
        runtime_hours = (end_time - start_time) / 3600

        logger.info("Search completed successfully!")
        logger.info("Total runtime: %.2f hours", runtime_hours)
        logger.info("Peak memory usage: %.1fGB", final_memory)
        logger.info("Best score: %.4f", cv.best_score_)

    return cv


def _configure_estimation_mode(cv):
    """Configure RandomizedSearchCV for fast estimation mode."""
    try:
        # Try to cap vectorizer features and reduce RF trees for speed
        if hasattr(cv, "estimator"):
            cv.estimator.set_params(**{
                "vect__max_features": 10000,
                "clf__estimator__n_estimators": 50,
            })
    except Exception:
        logger.debug("Could not set estimation mode parameters")

    # Use lighter CV strategy for estimate
    cv.cv = MultilabelStratifiedKFold(
        n_splits=ESTIMATION_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    cv.n_iter = min(getattr(cv, "n_iter", DEFAULT_N_ITER), ESTIMATION_MAX_ITER)


def _configure_production_mode(cv):
    """Configure RandomizedSearchCV for production mode to prevent resource contention."""
    try:
        # Ensure inner RF does not oversubscribe CPUs during CV
        if hasattr(cv, "estimator"):
            cv.estimator.set_params(**{
                "clf__estimator__n_jobs": 1,
            })
    except Exception:
        logger.debug("Could not set production mode parameters")
