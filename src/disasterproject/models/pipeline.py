"""
ML pipeline creation and model building functions.

This module provides functions to create sklearn pipelines and build models
for the disaster response classification system.
"""

import logging
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from disasterproject.data.preprocessor import tokenize

logger = logging.getLogger(__name__)


def create_pipeline(use_class_weights=False):
    """
    Create a basic sklearn pipeline for disaster response classification.

    Args:
        use_class_weights (bool): Whether to use class weights (ignored in basic pipeline)

    Returns:
        Pipeline: Configured sklearn pipeline
    """
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ])

    logger.info("Created basic pipeline without class weighting")
    return pipeline


def create_pipeline_with_custom_weights():
    """
    Create a sklearn pipeline with class weighting enabled.

    Returns:
        Pipeline: Configured sklearn pipeline with class weighting
    """
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(
            RandomForestClassifier(
                random_state=42,
                class_weight='balanced'
            )
        ))
    ])

    logger.info("Created pipeline with balanced class weighting")
    return pipeline


def build_model(pipeline, parameters):
    """
    Build a model by setting parameters on the pipeline.

    Args:
        pipeline (Pipeline): The sklearn pipeline to configure
        parameters (dict): Dictionary of parameters to set on the pipeline

    Returns:
        Pipeline: The configured pipeline (same object as input)
    """
    try:
        if parameters:
            pipeline.set_params(**parameters)
            logger.info("Applied %s parameters to pipeline", len(parameters))
        else:
            logger.info("No parameters provided, using pipeline defaults")

        return pipeline

    except Exception as e:
        logger.error("Failed to build model with parameters: %s", e)
        return None


class WeightedMultiOutputClassifier(MultiOutputClassifier):
    """
    MultiOutputClassifier that applies per-label class weights.
    
    Handles zero-positive labels by using DummyClassifier.
    """
    
    def __init__(self, estimator, class_weights_list=None, n_jobs=None):
        """
        Parameters:
        -----------
        estimator : estimator object
            Base estimator (e.g., LogisticRegression)
        class_weights_list : list of dict or None
            List of class weight dicts, one per label.
            Format: [{0: 1.0, 1: 2.5}, {0: 1.0, 1: 1.8}, ...]
        n_jobs : int or None
            Number of parallel jobs
        """
        super().__init__(estimator, n_jobs=n_jobs)
        self.class_weights_list = class_weights_list
    
    def fit(self, X, y, sample_weight=None):
        """Fit one estimator per label with appropriate class weights."""
        from sklearn.utils.validation import check_array
        
        # Validate input
        X = check_array(X, accept_sparse=True, force_all_finite=False)
        # Ensure y is 2D array for multi-output
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Always use custom fitting to handle single-class labels
        self.estimators_ = []
        self.classes_ = []
        
        for i, column in enumerate(y.T):
            # Get unique classes for this label
            classes = np.unique(column)
            self.classes_.append(classes)
            
            # Handle zero-positive labels (only class 0 present)
            if len(classes) == 1:
                logger.warning(f"Label {i} has only class {classes[0]}, using DummyClassifier")
                estimator = DummyClassifier(strategy='constant', constant=classes[0])
            else:
                # Clone base estimator
                estimator = clone(self.estimator)
                
                # Set class weights if available
                if self.class_weights_list and i < len(self.class_weights_list) and hasattr(estimator, 'class_weight'):
                    estimator.class_weight = self.class_weights_list[i]
                    logger.info(f"Label {i}: Applied weights {self.class_weights_list[i]}")
            
            # Fit estimator
            estimator.fit(X, column, sample_weight=sample_weight)
            self.estimators_.append(estimator)
        
        return self


def create_pipeline_logistic_regression(use_ngrams=True):
    """
    Create text processing pipeline with LogisticRegression classifier.
    
    Uses higher max_iter to handle convergence with imbalanced data.
    Uses WeightedMultiOutputClassifier (with no weights) to handle zero-positive labels.
    
    Args:
        use_ngrams (bool): Whether to use bigrams (1,2) or unigrams only (1,1)
    
    Returns:
        Pipeline: Configured sklearn pipeline with LogisticRegression
    """
    lr = LogisticRegression(
        max_iter=5000,
        solver='saga',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    # Use WeightedMultiOutputClassifier even without weights to handle single-class labels
    pipeline = Pipeline([
        ('vect', CountVectorizer(
            tokenizer=tokenize,
            ngram_range=(1, 2) if use_ngrams else (1, 1)
        )),
        ('tfidf', TfidfTransformer()),
        ('clf', WeightedMultiOutputClassifier(lr, class_weights_list=None, n_jobs=-1))
    ])
    
    logger.info("Created LogisticRegression pipeline with ngrams=%s", use_ngrams)
    return pipeline


def create_pipeline_logistic_regression_weighted(class_weights_list=None, use_ngrams=True):
    """
    Create text processing pipeline with weighted LogisticRegression.
    
    Parameters:
    -----------
    class_weights_list : list of dict or None
        Per-label class weights. Format: [{0: 1.0, 1: 2.5}, ...]
    use_ngrams : bool
        Whether to use bigrams
    
    Returns:
        Pipeline: Configured sklearn pipeline with weighted LogisticRegression
    """
    lr = LogisticRegression(
        max_iter=5000,
        solver='saga',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(
            tokenizer=tokenize,
            ngram_range=(1, 2) if use_ngrams else (1, 1)
        )),
        ('tfidf', TfidfTransformer()),
        ('clf', WeightedMultiOutputClassifier(
            lr, 
            class_weights_list=class_weights_list, 
            n_jobs=-1
        ))
    ])
    
    logger.info("Created weighted LogisticRegression pipeline with %s labels", 
                len(class_weights_list) if class_weights_list else 0)
    return pipeline