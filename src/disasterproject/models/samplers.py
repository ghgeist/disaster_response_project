"""
Sampling strategies for handling class imbalance in multi-label classification.
"""

import logging
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN

from ..utils.config import TARGET_COLUMNS


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
        raise ValueError(f"SMOTE sampling failed: {e}") from e


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
        raise ValueError(f"{method.upper()} sampling failed: {e}") from e


def get_multilabel_class_weights(y_train, strategy='balanced'):
    """
    Compute per-label class weights for a multi-label dataset.

    This returns a list of dictionaries, one per label column, mapping class value
    (0 or 1) to its weight. For "balanced", weights are computed as:
        w_c = N / (n_classes * n_c)
    where N is the number of samples and n_c is the count of class c for that label.

    Args:
        y_train (numpy.ndarray): Training labels of shape (n_samples, n_labels)
        strategy (str): Currently only "balanced" is supported

    Returns:
        list[dict[int, float]]: Weights per label column
    """
    try:
        if y_train is None or len(y_train) == 0:
            logging.warning("Empty y_train passed to get_multilabel_class_weights")
            return []

        if strategy != 'balanced':
            logging.warning("Only 'balanced' strategy is supported; defaulting to 'balanced'.")

        num_labels = y_train.shape[1]
        weights_per_label = []

        for label_index in range(num_labels):
            column = y_train[:, label_index]
            unique, counts = np.unique(column, return_counts=True)
            class_counts = dict(zip(unique.astype(int), counts.astype(int)))

            count_0 = class_counts.get(0, 0)
            count_1 = class_counts.get(1, 0)

            if count_0 > 0 and count_1 > 0:
                total = count_0 + count_1
                w0 = total / (2.0 * count_0)
                w1 = total / (2.0 * count_1)
            else:
                # If a class is missing, avoid extreme/undefined weights
                w0 = 1.0
                w1 = 1.0

            weights_per_label.append({0: float(w0), 1: float(w1)})

        return weights_per_label
    except Exception as e:
        logging.error(f"Error computing multilabel class weights: {e}")
        return []


def apply_proper_multilabel_sampling(X_train, y_train, method='none', **kwargs):
    """
    Compatibility wrapper for sampling methods referenced in scripts.

    Maps higher-level method names to the available implementations. For now,
    this routes to conservative, SMOTE, or ADASYN variants, or returns the
    original data when no sampling is requested.

    Supported methods:
      - 'none': return inputs unchanged
      - 'mlsmote': use SMOTE-based approach
      - 'random_oversample': use a conservative SMOTE fallback
      - 'label_powerset': use ADASYN as a proxy

    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
        method (str): Method name
        **kwargs: Unused extra parameters for compatibility

    Returns:
        tuple: (X_resampled, y_resampled)
    """
    try:
        method = (method or 'none').lower()

        if method == 'none':
            logging.info("No sampling selected; returning original training data.")
            return X_train, y_train
        if method == 'mlsmote':
            return apply_multi_label_aware_sampling(X_train, y_train, method='smote')
        if method == 'random_oversample':
            # Fallback to a conservative SMOTE approach as a proxy for simple oversampling
            return apply_multi_label_aware_sampling(X_train, y_train, method='conservative')
        if method == 'label_powerset':
            # Use ADASYN as a proxy; underlying implementation treats each unique label vector as a class
            return apply_multi_label_aware_sampling(X_train, y_train, method='adasyn')

        # Default: delegate to the underlying implementation if the name matches
        return apply_multi_label_aware_sampling(X_train, y_train, method=method)
    except Exception as e:
        logging.error(f"Error in apply_proper_multilabel_sampling: {e}")
        raise ValueError(f"Multilabel sampling failed: {e}") from e