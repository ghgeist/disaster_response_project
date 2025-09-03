"""
Multi-label aware sampling strategies for handling class imbalance.

This module provides implementations that properly handle multi-label classification
with severe class imbalance, using approaches that work with the multi-label nature
of the data.
"""

import logging
import numpy as np
from collections import Counter
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import warnings


def apply_mlsmote(X_train, y_train, k_neighbors=5, sampling_strategy='auto'):
    """
    Apply ML-SMOTE (Multi-Label SMOTE) using binary relevance transformation.
    
    This approach applies SMOTE to each label independently, then combines
    the results to maintain label correlations as much as possible.
    
    Args:
        X_train: Training features
        y_train: Training labels (multi-label)
        k_neighbors: Number of neighbors for SMOTE
        sampling_strategy: Sampling strategy for SMOTE
        
    Returns:
        Resampled X_train and y_train
    """
    try:
        logging.info("Applying ML-SMOTE (Binary Relevance approach)...")
        n_labels = y_train.shape[1]
        
        # Store original samples
        X_resampled_all = []
        y_resampled_all = []
        
        # Apply SMOTE to each label independently
        for label_idx in range(n_labels):
            # Get current label
            y_current = y_train[:, label_idx]
            
            # Check if label has minority class
            unique, counts = np.unique(y_current, return_counts=True)
            if len(unique) < 2 or min(counts) < k_neighbors + 1:
                continue
                
            try:
                # Apply SMOTE to current label
                smote = SMOTE(k_neighbors=min(k_neighbors, min(counts)-1), 
                             sampling_strategy=sampling_strategy,
                             random_state=42)
                X_res, y_res = smote.fit_resample(X_train, y_current)
                
                # Find new samples (those added by SMOTE)
                n_original = len(X_train)
                if len(X_res) > n_original:
                    new_indices = list(range(n_original, len(X_res)))
                    X_new = X_res[new_indices]
                    
                    # Create multi-label y for new samples
                    y_new = np.zeros((len(X_new), n_labels))
                    y_new[:, label_idx] = y_res[new_indices]
                    
                    X_resampled_all.append(X_new)
                    y_resampled_all.append(y_new)
                    
            except Exception as e:
                logging.debug(f"Could not apply SMOTE to label {label_idx}: {e}")
                continue
        
        # Combine original data with all resampled data
        if X_resampled_all:
            X_combined = np.vstack([X_train] + X_resampled_all)
            y_combined = np.vstack([y_train] + y_resampled_all)
            
            logging.info(f"ML-SMOTE: {len(X_train)} -> {len(X_combined)} samples")
            return X_combined, y_combined
        else:
            logging.warning("ML-SMOTE could not be applied to any label")
            return X_train, y_train
            
    except Exception as e:
        logging.error(f"Error in ML-SMOTE: {e}")
        return X_train, y_train


def apply_label_powerset_sampling(X_train, y_train, sampling_ratio=0.5):
    """
    Apply sampling using Label Powerset transformation.
    
    This approach treats each unique label combination as a single class
    and performs oversampling on the transformed problem.
    
    Args:
        X_train: Training features  
        y_train: Training labels (multi-label)
        sampling_ratio: Ratio of minority to majority class after sampling
        
    Returns:
        Resampled X_train and y_train
    """
    try:
        logging.info("Applying Label Powerset sampling...")
        
        # Convert label combinations to single labels
        y_powerset = [''.join(map(str, row.astype(int))) for row in y_train]
        
        # Count label combinations
        label_counts = Counter(y_powerset)
        
        # Identify minority combinations (less than median count)
        median_count = np.median(list(label_counts.values()))
        minority_labels = [label for label, count in label_counts.items() 
                          if count < median_count]
        
        # Resample minority combinations
        X_resampled = X_train.copy()
        y_resampled = y_train.copy()
        
        for minority_label in minority_labels:
            # Find indices of minority label
            indices = [i for i, label in enumerate(y_powerset) if label == minority_label]
            
            if len(indices) > 0:
                # Calculate number of samples to add
                current_count = len(indices)
                target_count = int(median_count * sampling_ratio)
                n_samples_to_add = max(0, target_count - current_count)
                
                if n_samples_to_add > 0:
                    # Resample with replacement
                    new_indices = resample(indices, n_samples=n_samples_to_add, 
                                         random_state=42, replace=True)
                    
                    X_resampled = np.vstack([X_resampled, X_train[new_indices]])
                    y_resampled = np.vstack([y_resampled, y_train[new_indices]])
        
        logging.info(f"Label Powerset: {len(X_train)} -> {len(X_resampled)} samples")
        return X_resampled, y_resampled
        
    except Exception as e:
        logging.error(f"Error in Label Powerset sampling: {e}")
        return X_train, y_train


def apply_random_oversampling_multilabel(X_train, y_train, sampling_strategy='auto', 
                                        min_samples_threshold=10):
    """
    Apply random oversampling for each label independently.
    
    This is a simple but effective approach that randomly duplicates
    minority class samples for each label.
    
    Args:
        X_train: Training features
        y_train: Training labels (multi-label)
        sampling_strategy: 'auto' or float ratio
        min_samples_threshold: Minimum samples to consider a class as minority
        
    Returns:
        Resampled X_train and y_train
    """
    try:
        logging.info("Applying Random Oversampling for multi-label data...")
        
        X_resampled = X_train.copy()
        y_resampled = y_train.copy()
        
        n_labels = y_train.shape[1]
        
        for label_idx in range(n_labels):
            y_current = y_train[:, label_idx]
            
            # Get class distribution
            unique, counts = np.unique(y_current, return_counts=True)
            if len(unique) < 2:
                continue
                
            # Find minority class
            minority_class = unique[np.argmin(counts)]
            minority_count = np.min(counts)
            majority_count = np.max(counts)
            
            # Skip if minority class has too few samples
            if minority_count < min_samples_threshold:
                continue
            
            # Calculate target count
            if sampling_strategy == 'auto':
                target_count = majority_count
            elif isinstance(sampling_strategy, float):
                target_count = int(majority_count * sampling_strategy)
            else:
                target_count = majority_count
                
            # Find minority class indices
            minority_indices = np.where(y_current == minority_class)[0]
            
            # Calculate how many samples to add
            n_samples_to_add = target_count - minority_count
            
            if n_samples_to_add > 0:
                # Randomly sample with replacement
                new_indices = resample(minority_indices, n_samples=n_samples_to_add,
                                     random_state=42 + label_idx)
                
                # Add to resampled data
                X_resampled = np.vstack([X_resampled, X_train[new_indices]])
                
                # Create label vector for new samples
                new_labels = np.zeros((n_samples_to_add, n_labels))
                new_labels[:, label_idx] = 1  # Set current label
                
                # Copy other labels from original samples
                for i, idx in enumerate(new_indices):
                    new_labels[i] = y_train[idx]
                    
                y_resampled = np.vstack([y_resampled, new_labels])
        
        logging.info(f"Random Oversampling: {len(X_train)} -> {len(X_resampled)} samples")
        return X_resampled, y_resampled
        
    except Exception as e:
        logging.error(f"Error in Random Oversampling: {e}")
        return X_train, y_train


def get_class_weights_multilabel(y_train, weight_strategy='balanced'):
    """
    Calculate class weights for multi-label classification.
    
    This function returns weights that can be used in classifiers
    instead of resampling the data.
    
    Args:
        y_train: Training labels (multi-label)
        weight_strategy: 'balanced' or 'inverse_frequency'
        
    Returns:
        Dictionary mapping label indices to class weight dictionaries
    """
    try:
        n_samples = len(y_train)
        n_labels = y_train.shape[1]
        
        class_weights = {}
        
        for label_idx in range(n_labels):
            y_current = y_train[:, label_idx]
            
            # Count classes
            unique, counts = np.unique(y_current, return_counts=True)
            
            if weight_strategy == 'balanced':
                # sklearn-style balanced weights
                weights = n_samples / (len(unique) * counts)
            elif weight_strategy == 'inverse_frequency':
                # Inverse frequency weights
                weights = 1.0 / counts
                weights = weights / weights.sum()  # Normalize
            else:
                weights = np.ones_like(counts)
                
            # Create weight dictionary for this label
            class_weights[label_idx] = {cls: weight for cls, weight in zip(unique, weights)}
            
        return class_weights
        
    except Exception as e:
        logging.error(f"Error calculating class weights: {e}")
        return None


def print_multilabel_class_distribution(y, label_names=None, top_n=10):
    """
    Print class distribution statistics for multi-label data.
    
    Args:
        y: Label matrix
        label_names: List of label names
        top_n: Number of most imbalanced labels to show
    """
    n_labels = y.shape[1]
    imbalance_ratios = []
    
    for i in range(n_labels):
        unique, counts = np.unique(y[:, i], return_counts=True)
        if len(unique) == 2:
            ratio = max(counts) / min(counts)
            label_name = label_names[i] if label_names else f"Label_{i}"
            imbalance_ratios.append((label_name, ratio, dict(zip(unique, counts))))
    
    # Sort by imbalance ratio
    imbalance_ratios.sort(key=lambda x: x[1], reverse=True)
    
    logging.info(f"Top {top_n} most imbalanced labels:")
    for label, ratio, dist in imbalance_ratios[:top_n]:
        logging.info(f"  {label}: {dist} (ratio: {ratio:.1f}:1)")
