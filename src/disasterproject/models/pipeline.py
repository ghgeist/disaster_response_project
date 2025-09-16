"""
ML pipeline creation and model building functions.

This module provides functions to create sklearn pipelines and build models
for the disaster response classification system.
"""

import logging
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
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
            logger.info(f"Applied {len(parameters)} parameters to pipeline")
        else:
            logger.info("No parameters provided, using pipeline defaults")

        return pipeline

    except Exception as e:
        logger.error(f"Failed to build model with parameters: {e}")
        return None