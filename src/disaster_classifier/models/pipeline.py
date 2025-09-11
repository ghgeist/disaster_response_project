"""
Machine learning pipeline creation and model building functions.
"""

import logging
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from ..data.preprocessor import tokenize


def create_pipeline(use_class_weights=False):
    """
    Create a machine learning pipeline.

    This function creates a pipeline that first vectorizes the text data using CountVectorizer and a custom tokenizer,
    then applies a TF-IDF transformation, and finally uses a multi-output classifier.

    Args:
        use_class_weights (bool): Whether to enable class weight support in the pipeline

    Returns:
    pipeline (sklearn.pipeline.Pipeline): The machine learning pipeline.

    If an error occurs while creating the pipeline, None is returned.
    """
    try:
        # Instantiate and configure the pipeline
        pipeline = Pipeline(
            [
                (
                    "vect",
                    CountVectorizer(tokenizer=tokenize),
                ),  # Tokenize and vectorize text
                (
                    "tfidf",
                    TfidfTransformer(smooth_idf=False),
                ),  # Apply TF-IDF transformation
                (
                    "clf",
                    MultiOutputClassifier(
                        RandomForestClassifier(
                            n_jobs=multiprocessing.cpu_count() - 1,
                            class_weight='balanced' if use_class_weights else None
                        )
                    ),
                ),  # Use MultiOutputClassifier with RandomForest, n_jobs specifies cores
            ]
        )
    except (TypeError, ValueError) as e:
        logging.error("Error creating pipeline: %s", e)
        return None

    return pipeline


def create_pipeline_with_custom_weights(class_weights_dict=None):
    """
    Create a machine learning pipeline with custom class weights for multi-label classification.

    Note: For multi-label classification with MultiOutputClassifier, class weights are applied
    automatically when using 'balanced' class_weight. This function serves as a foundation
    for more advanced weight customization if needed.

    Args:
        class_weights_dict (dict, optional): Dictionary mapping label indices to class weight dictionaries

    Returns:
        pipeline (sklearn.pipeline.Pipeline): The machine learning pipeline with class weights
    """
    try:
        # For multi-label classification, use balanced class weights
        # MultiOutputClassifier will handle per-label weighting automatically
        pipeline = create_pipeline(use_class_weights=True)
        
        if pipeline is None:
            return None
            
        logging.info("Pipeline configured with balanced class weights for multi-label classification")
        return pipeline
        
    except Exception as e:
        logging.error("Error creating pipeline with custom weights: %s", e)
        return None


def build_model(pipeline, parameters):
    """
    Build a machine learning model.

    This function configures the RandomForestClassifier in the pipeline with the given parameters.

    Args:
    pipeline (sklearn.pipeline.Pipeline): The machine learning pipeline.
    parameters (dict): The parameters for the RandomForestClassifier.

    Returns:
    pipeline (sklearn.pipeline.Pipeline): The machine learning pipeline.

    If an error occurs while building the model, None is returned.
    """
    try:
        # Configure the RandomForestClassifier with the given parameters
        default_params = {
            "clf__estimator__n_estimators": 100,
            "clf__estimator__max_depth": 25,
            "clf__estimator__min_samples_leaf": 2,
            "clf__estimator__max_features": "sqrt",
        }

        if parameters is None:
            pipeline.set_params(clf__estimator__random_state=42, **default_params)
        else:
            # Merge defaults for any missing downsizing guard params
            merged = {**default_params, **parameters}
            pipeline.set_params(clf__estimator__random_state=42, **merged)
    except (ValueError, ImportError, TypeError) as e:
        logging.error("Error building model: %s", e)
        return None

    return pipeline


def run_grid_search(pipeline, parameters, X_train, y_train, use_small_subset=False):
    """
    Run a grid search to find the best parameters for a pipeline.

    This function uses GridSearchCV to find the best parameters for the specified pipeline using the provided training data.
    The function measures the time it takes to run the grid search and logs the runtime.
    If use_small_subset is True, the function uses only the first 100 samples of the training data and estimates the total runtime based on this subset.

    Args:
    pipeline (Pipeline): The pipeline for which to find the best parameters.
    parameters (dict): The parameters to try in the grid search.
    X_train (numpy.ndarray): The features for the training data.
    y_train (numpy.ndarray): The labels for the training data.
    use_small_subset (bool, optional): Whether to use only the first 100 samples of the training data. Defaults to False.

    Returns:
    cv (GridSearchCV): The fitted GridSearchCV instance.
    """
    from time import time
    
    start_time = time()
    cv = GridSearchCV(
        pipeline,
        param_grid=parameters,
        scoring="accuracy",
        n_jobs=multiprocessing.cpu_count() - 1,
        verbose=1,
    )

    if use_small_subset:
        X_train_size = len(X_train)
        X_train = X_train[:100]
        y_train = y_train[:100]
        cv.fit(X_train, y_train)
        end_time = time()
        runtime = (end_time - start_time) * (
            X_train_size / 100
        )  # keep the time in seconds
        formatted_runtime = f"{runtime:.2f} seconds (estimated)"
    else:
        cv.fit(X_train, y_train)
        end_time = time()
        runtime = end_time - start_time  # keep the time in seconds
        hours, remainder = divmod(runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_runtime = f"{int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds (actual)"

    logging.info(f"Runtime: {formatted_runtime}")

    return cv
