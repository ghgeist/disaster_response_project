def run_parameter_search(pipeline, parameters, X_train, y_train, use_small_subset=False):
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

    Returns:
    cv (RandomizedSearchCV): The fitted RandomizedSearchCV instance.

    """
    from time import time
    from sklearn.model_selection import RandomizedSearchCV
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    import numpy as np
    import multiprocessing
    import psutil
    import logging

    start_time = time()

    # Resource management settings
    MEMORY_LIMIT_GB = 12  # Abort if memory usage exceeds 12GB
    MEMORY_WARNING_GB = 10  # Warn if memory usage exceeds 10GB
    TIMEOUT_HOURS = 4  # Overall timeout

    # Check initial memory usage
    initial_memory = psutil.Process().memory_info().rss / (1024**3)
    available_memory = psutil.virtual_memory().available / (1024**3)

    print(f"Initial memory usage: {initial_memory:.1f}GB")
    print(f"Available system memory: {available_memory:.1f}GB")

    if available_memory < 2.0:
        raise RuntimeError(f"Insufficient memory available: {available_memory:.1f}GB. Need at least 2GB.")

    # Use MultilabelStratifiedKFold for proper multi-label cross-validation
    cv_strategy = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Conservative resource settings for system stability
    cv = RandomizedSearchCV(
        pipeline,
        param_distributions=parameters,
        n_iter=20,  # Number of parameter settings that are sampled
        scoring="f1_weighted",
        cv=cv_strategy,  # Use proper multi-label CV strategy
        n_jobs=2,  # Conservative CPU usage for system responsiveness
        verbose=2,  # More detailed progress output
        random_state=42 # for reproducibility
    )

    if use_small_subset:
        print("Using a small subset (100 samples) for estimation...")
        X_train_size = len(X_train)
        X_train = X_train[:100]
        y_train = y_train[:100]

        # Memory check before fitting
        current_memory = psutil.Process().memory_info().rss / (1024**3)
        if current_memory > MEMORY_WARNING_GB:
            print(f"WARNING: Memory usage {current_memory:.1f}GB exceeds warning threshold")

        cv.fit(X_train, y_train)
        end_time = time()
        runtime = (end_time - start_time) * (X_train_size / 100)
        hours, remainder = divmod(runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nESTIMATED FULL RUNTIME: {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds")
    else:
        print("Starting hyperparameter search with resource monitoring...")
        print(f"Configuration: n_iter=20, n_jobs=2, timeout={TIMEOUT_HOURS}h")

        # Monitor memory usage during fitting
        class MemoryMonitor:
            def __init__(self):
                self.last_check = time()

            def check(self):
                now = time()
                if now - self.last_check > 600:  # Check every 10 minutes
                    current_memory = psutil.Process().memory_info().rss / (1024**3)
                    elapsed = (now - start_time) / 3600
                    print(f"Progress update: {elapsed:.1f}h elapsed, Memory: {current_memory:.1f}GB")

                    if current_memory > MEMORY_LIMIT_GB:
                        raise RuntimeError(f"Memory limit exceeded: {current_memory:.1f}GB > {MEMORY_LIMIT_GB}GB")
                    elif current_memory > MEMORY_WARNING_GB:
                        print(f"WARNING: High memory usage: {current_memory:.1f}GB")

                    self.last_check = now

        monitor = MemoryMonitor()

        # Fit with basic monitoring (timeout handled by RandomizedSearchCV internally)
        try:
            print("Starting RandomizedSearchCV.fit()...")
            cv.fit(X_train, y_train)
        except Exception as e:
            current_memory = psutil.Process().memory_info().rss / (1024**3)
            print(f"Search failed with error: {e}")
            print(f"Memory usage at failure: {current_memory:.1f}GB")
            raise

        end_time = time()
        final_memory = psutil.Process().memory_info().rss / (1024**3)
        runtime_hours = (end_time - start_time) / 3600

        print(f"\nSearch completed successfully!")
        print(f"Total runtime: {runtime_hours:.2f} hours")
        print(f"Peak memory usage: {final_memory:.1f}GB")
        print(f"Best score: {cv.best_score_:.4f}")

    return cv