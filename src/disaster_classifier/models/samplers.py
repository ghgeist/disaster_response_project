"""
Sampling strategies for handling class imbalance in multi-label classification.
"""

import logging
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN

from ..utils.config import TARGET_COLUMNS
from .multilabel_samplers import (
    apply_mlsmote,
    apply_label_powerset_sampling,
    apply_random_oversampling_multilabel,
    get_class_weights_multilabel,
    print_multilabel_class_distribution
)


def apply_smote_sampling(X_train, y_train):
    """
    Apply SMOTE oversampling to handle class imbalance in multi-label classification.
    
    This function applies SMOTE to the multi-label dataset using a more conservative approach
    that works better with text data. It uses regular SMOTE with better parameters and
    handles the multi-label nature more carefully.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
        
    Returns:
        tuple: (X_train_resampled, y_train_resampled) - The resampled training data
        
    Raises:
        ValueError: If SMOTE cannot be applied to any target column
    """
    try:
        # Print before class distribution statistics (concise format)
        logging.info("Class distribution BEFORE SMOTE:")
        for i, col in enumerate(TARGET_COLUMNS):
            unique, counts = np.unique(y_train[:, i], return_counts=True)
            class_dist = dict(zip(unique, counts))
            # Show only classes with significant imbalance (ratio > 10:1)
            if len(class_dist) == 2:
                ratio = max(class_dist.values()) / min(class_dist.values())
                if ratio > 10:
                    logging.info(f"  {col}: {class_dist} (ratio: {ratio:.1f}:1)")
        
        # Use regular SMOTE with more conservative parameters for text data
        # k_neighbors=3 is more conservative than 1, and sampling_strategy='auto' 
        # will balance classes to the majority class size
        sampler = SMOTE(
            random_state=42, 
            k_neighbors=3,  # More conservative than 1
            sampling_strategy='auto'  # Balance to majority class
        )
        
        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
        
        logging.info(f"Training samples: {len(X_train)} -> {len(X_train_resampled)}")
        
        # Print after class distribution statistics
        logging.info("Class distribution AFTER SMOTE:")
        for i, col in enumerate(TARGET_COLUMNS):
            unique, counts = np.unique(y_train_resampled[:, i], return_counts=True)
            class_dist = dict(zip(unique, counts))
            if len(class_dist) == 2:
                ratio = max(class_dist.values()) / min(class_dist.values())
                logging.info(f"  {col}: {class_dist} (ratio: {ratio:.1f}:1)")
        
        return X_train_resampled, y_train_resampled
        
    except Exception as e:
        logging.error(f"Error applying SMOTE: {e}")
        logging.warning("SMOTE could not be applied. Using original training data.")
        return X_train, y_train


def apply_multi_label_aware_sampling(X_train, y_train, method='smote'):
    """
    Apply oversampling that's more aware of multi-label classification.
    
    This function tries different approaches to handle the multi-label nature
    of the data more effectively than standard SMOTE.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
        method (str): Sampling method ('smote', 'adasyn', 'conservative')
        
    Returns:
        tuple: (X_train_resampled, y_train_resampled) - The resampled training data
    """
    try:
        logging.info(f"Applying {method.upper()} sampling for multi-label classification...")
        
        # Print before class distribution statistics
        logging.info("Class distribution BEFORE sampling:")
        for i, col in enumerate(TARGET_COLUMNS):
            unique, counts = np.unique(y_train[:, i], return_counts=True)
            class_dist = dict(zip(unique, counts))
            if len(class_dist) == 2:
                ratio = max(class_dist.values()) / min(class_dist.values())
                if ratio > 5:  # Show all imbalanced classes
                    logging.info(f"  {col}: {class_dist} (ratio: {ratio:.1f}:1)")
        
        if method == 'smote':
            # Conservative SMOTE with moderate oversampling
            sampler = SMOTE(
                random_state=42, 
                k_neighbors=5,  # More conservative
                sampling_strategy=0.5  # Only oversample to 50% of majority class
            )
        elif method == 'adasyn':
            # ADASYN focuses on harder-to-learn examples
            sampler = ADASYN(
                random_state=42,
                n_neighbors=5,
                sampling_strategy=0.5
            )
        elif method == 'conservative':
            # Very conservative SMOTE
            sampler = SMOTE(
                random_state=42, 
                k_neighbors=7,  # Very conservative
                sampling_strategy=0.3  # Only oversample to 30% of majority class
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
        
        logging.info(f"Training samples: {len(X_train)} -> {len(X_train_resampled)}")
        
        # Print after class distribution statistics
        logging.info("Class distribution AFTER sampling:")
        for i, col in enumerate(TARGET_COLUMNS):
            unique, counts = np.unique(y_train_resampled[:, i], return_counts=True)
            class_dist = dict(zip(unique, counts))
            if len(class_dist) == 2:
                ratio = max(class_dist.values()) / min(class_dist.values())
                logging.info(f"  {col}: {class_dist} (ratio: {ratio:.1f}:1)")
        
        return X_train_resampled, y_train_resampled
        
    except Exception as e:
        logging.error(f"Error applying {method} sampling: {e}")
        logging.warning(f"{method.upper()} could not be applied. Using original training data.")
        return X_train, y_train


def apply_proper_multilabel_sampling(X_train, y_train, method='mlsmote', **kwargs):
    """
    Apply proper multi-label aware sampling strategies.
    
    This function uses sampling methods specifically designed for multi-label
    classification, avoiding the limitations of standard SMOTE/ADASYN.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels (multi-label)
        method (str): Sampling method to use:
            - 'mlsmote': Multi-Label SMOTE using binary relevance
            - 'label_powerset': Label Powerset transformation sampling
            - 'random_oversample': Random oversampling per label
            - 'none': No sampling (for comparison)
        **kwargs: Additional arguments for specific methods
        
    Returns:
        tuple: (X_train_resampled, y_train_resampled) - The resampled training data
    """
    try:
        logging.info(f"Applying {method} for proper multi-label sampling...")
        
        # Print initial class distribution
        print_multilabel_class_distribution(y_train, TARGET_COLUMNS, top_n=10)
        
        if method == 'none':
            logging.info("No sampling applied (baseline)")
            return X_train, y_train
            
        elif method == 'mlsmote':
            # Multi-Label SMOTE
            k_neighbors = kwargs.get('k_neighbors', 5)
            sampling_strategy = kwargs.get('sampling_strategy', 0.5)
            X_resampled, y_resampled = apply_mlsmote(
                X_train, y_train, 
                k_neighbors=k_neighbors,
                sampling_strategy=sampling_strategy
            )
            
        elif method == 'label_powerset':
            # Label Powerset sampling
            sampling_ratio = kwargs.get('sampling_ratio', 0.5)
            X_resampled, y_resampled = apply_label_powerset_sampling(
                X_train, y_train,
                sampling_ratio=sampling_ratio
            )
            
        elif method == 'random_oversample':
            # Random oversampling
            sampling_strategy = kwargs.get('sampling_strategy', 0.5)
            min_samples = kwargs.get('min_samples_threshold', 10)
            X_resampled, y_resampled = apply_random_oversampling_multilabel(
                X_train, y_train,
                sampling_strategy=sampling_strategy,
                min_samples_threshold=min_samples
            )
            
        else:
            logging.warning(f"Unknown sampling method: {method}")
            return X_train, y_train
            
        # Print final class distribution
        logging.info("Class distribution after sampling:")
        print_multilabel_class_distribution(y_resampled, TARGET_COLUMNS, top_n=10)
        
        return X_resampled, y_resampled
        
    except Exception as e:
        logging.error(f"Error in multi-label sampling: {e}")
        logging.warning("Returning original training data.")
        return X_train, y_train


def get_multilabel_class_weights(y_train, strategy='balanced'):
    """
    Get class weights for multi-label classification.
    
    Use this as an alternative to resampling by passing weights to the classifier.
    
    Args:
        y_train (numpy.ndarray): Training labels
        strategy (str): Weight calculation strategy
        
    Returns:
        dict: Class weights for each label
    """
    return get_class_weights_multilabel(y_train, weight_strategy=strategy)
