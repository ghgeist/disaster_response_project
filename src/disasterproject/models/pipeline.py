def run_parameter_search(pipeline, parameters, X_train, y_train, use_small_subset=False):
    """
    Run a randomized search to find the best parameters for a pipeline.

    This function uses RandomizedSearchCV to find the best parameters for the specified pipeline using the provided training data.
    The function measures the time it takes to run the search and logs the runtime.
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

    start_time = time()

    # Use MultilabelStratifiedKFold for proper multi-label cross-validation
    cv_strategy = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    cv = RandomizedSearchCV(
        pipeline,
        param_distributions=parameters,
        n_iter=20,  # Number of parameter settings that are sampled
        scoring="f1_weighted",
        cv=cv_strategy,  # Use proper multi-label CV strategy
        n_jobs=multiprocessing.cpu_count() - 1,
        verbose=1,
        random_state=42 # for reproducibility
    )

    if use_small_subset:
        print("Using a small subset (100 samples) for estimation...")
        X_train_size = len(X_train)
        X_train = X_train[:100]
        y_train = y_train[:100]
        cv.fit(X_train, y_train)
        end_time = time()
        runtime = (end_time - start_time) * (X_train_size / 100)
        hours, remainder = divmod(runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nESTIMATED FULL RUNTIME: {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds")
    else:
        cv.fit(X_train, y_train)

    return cv