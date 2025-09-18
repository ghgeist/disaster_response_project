"""
Hierarchy post-processor for enforcing parent-child consistency in multi-label predictions.

This module implements a lightweight hierarchy fixer that:
1. Enforces parent ≥ child probabilities for taxonomy groups
2. Ensures child=1 forces parent=1 (decision-level consistency)
3. Applies gentle threshold softening for critical safety labels
4. Respects exclusions for labels with data limitations (e.g., child_alone)

Key design decisions:
- Post-processing approach for efficiency (no model retraining required)
- Conservative approach: ignores reverse violations (parent=1, all children=0)
- Special handling for 'related' group: decision-level only, no probability clamping
- Deterministic behavior with clear logging of adjustments
"""

import logging
from typing import Dict, List, Set, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


def apply_hierarchy(
    probs: Dict[str, float],
    thresholds: Dict[str, float],
    taxonomy: Dict[str, List[str]],
    critical_labels: Set[str],
    exclude: Set[str],
    critical_threshold_reduction: float = 0.10
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Apply hierarchical consistency to probabilities and generate binary predictions.

    Args:
        probs: Dictionary mapping label names to probabilities (0.0-1.0)
        thresholds: Dictionary mapping label names to decision thresholds
        taxonomy: Dictionary mapping parent labels to list of child labels
        critical_labels: Set of labels requiring softer thresholds for safety
        exclude: Set of labels to exclude from hierarchy constraints
        critical_threshold_reduction: Amount to reduce thresholds for critical labels

    Returns:
        Tuple of (adjusted_probs, binary_predictions)
        - adjusted_probs: Probabilities after hierarchy enforcement
        - binary_predictions: Binary decisions (0/1) after threshold application

    Hierarchy Rules:
        1. For taxonomy groups (except 'related'): enforce p(parent) ≥ p(child)
        2. For 'related' group: apply decision-level child→parent only
        3. Decision forcing: any child=1 forces parent=1
        4. Critical labels get reduced thresholds for higher recall
        5. Excluded labels are unchanged by hierarchy constraints
    """
    # Work with copies to avoid modifying inputs
    adjusted_probs = probs.copy()
    adjusted_thresholds = thresholds.copy()

    # Apply critical threshold reduction
    for label in critical_labels:
        if label in adjusted_thresholds:
            original_threshold = adjusted_thresholds[label]
            adjusted_thresholds[label] = max(0.0, original_threshold - critical_threshold_reduction)
            logger.debug(f"Critical label {label}: threshold {original_threshold:.3f} → {adjusted_thresholds[label]:.3f}")

    # Track adjustments for logging
    prob_adjustments = 0
    parent_activations = 0

    # Phase 1: Enforce probability monotonicity for taxonomy groups (except 'related')
    for parent, children in taxonomy.items():
        if parent == "related":
            continue  # Skip probability clamping for 'related' group

        if parent in exclude:
            continue

        # Find valid children (present in probs and not excluded)
        valid_children = [child for child in children
                         if child in adjusted_probs and child not in exclude]

        if not valid_children or parent not in adjusted_probs:
            continue

        # Enforce parent ≥ max(children) for probabilities
        max_child_prob = max(adjusted_probs[child] for child in valid_children)

        if adjusted_probs[parent] < max_child_prob:
            old_prob = adjusted_probs[parent]
            adjusted_probs[parent] = max_child_prob
            prob_adjustments += 1
            logger.debug(f"Boosted {parent} prob: {old_prob:.3f} → {max_child_prob:.3f}")

        # Enforce children ≤ parent for probabilities
        for child in valid_children:
            if adjusted_probs[child] > adjusted_probs[parent]:
                old_prob = adjusted_probs[child]
                adjusted_probs[child] = adjusted_probs[parent]
                prob_adjustments += 1
                logger.debug(f"Clamped {child} prob: {old_prob:.3f} → {adjusted_probs[parent]:.3f}")

    # Phase 2: Generate binary predictions using adjusted thresholds
    binary_predictions = {}
    for label, prob in adjusted_probs.items():
        threshold = adjusted_thresholds.get(label, 0.5)  # Default threshold if not specified
        binary_predictions[label] = 1 if prob >= threshold else 0

    # Phase 3: Enforce decision-level child→parent for all taxonomy groups
    for parent, children in taxonomy.items():
        if parent in exclude:
            continue

        # Find valid children
        valid_children = [child for child in children
                         if child in binary_predictions and child not in exclude]

        if not valid_children or parent not in binary_predictions:
            continue

        # If any child=1, force parent=1
        if any(binary_predictions[child] == 1 for child in valid_children):
            if binary_predictions[parent] == 0:
                binary_predictions[parent] = 1
                parent_activations += 1
                logger.debug(f"Forced {parent}=1 due to active children: {[c for c in valid_children if binary_predictions[c] == 1]}")

    # Log summary of adjustments
    if prob_adjustments > 0 or parent_activations > 0:
        logger.info(f"Hierarchy adjustments: {prob_adjustments} probability fixes, {parent_activations} parent activations")

    return adjusted_probs, binary_predictions


def count_violations(
    probs: Dict[str, float],
    taxonomy: Dict[str, List[str]],
    exclude: Set[str]
) -> int:
    """
    Count parent < child probability violations in the given predictions.

    Args:
        probs: Dictionary mapping label names to probabilities
        taxonomy: Dictionary mapping parent labels to list of child labels
        exclude: Set of labels to exclude from violation counting

    Returns:
        Total number of parent < child violations found
    """
    violations = 0

    for parent, children in taxonomy.items():
        if parent == "related":
            continue  # Skip 'related' group as it doesn't use probability constraints

        if parent in exclude:
            continue

        # Find valid children (present in probs and not excluded)
        valid_children = [child for child in children
                         if child in probs and child not in exclude]

        if not valid_children or parent not in probs:
            continue

        # Count violations where child > parent
        for child in valid_children:
            if probs[child] > probs[parent]:
                violations += 1
                logger.debug(f"Violation: {child} ({probs[child]:.3f}) > {parent} ({probs[parent]:.3f})")

    return violations


def optimize_critical_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label_names: List[str],
    critical_labels: Set[str],
    target_recall: float = 0.8
) -> Dict[str, float]:
    """
    Find validation-optimized thresholds for critical labels targeting desired recall.

    Args:
        y_true: True binary labels, shape (n_samples, n_labels)
        y_proba: Predicted probabilities, shape (n_samples, n_labels)
        label_names: List of label names corresponding to columns
        critical_labels: Set of critical label names
        target_recall: Target recall for critical labels

    Returns:
        Dictionary mapping critical label names to optimized thresholds
    """
    from sklearn.metrics import precision_recall_curve

    thresholds = {}
    name_to_idx = {name: i for i, name in enumerate(label_names)}

    for label in critical_labels:
        if label not in name_to_idx:
            logger.warning(f"Critical label {label} not found in label_names")
            continue

        idx = name_to_idx[label]
        y_true_label = y_true[:, idx]
        y_proba_label = y_proba[:, idx]

        # Skip if no positive examples
        if np.sum(y_true_label) == 0:
            logger.warning(f"No positive examples for critical label {label}, using default threshold")
            thresholds[label] = 0.5
            continue

        try:
            precision, recall, thresh = precision_recall_curve(y_true_label, y_proba_label)

            # Find threshold with recall nearest to target
            recall_diff = np.abs(recall - target_recall)
            best_idx = int(np.argmin(recall_diff))

            # precision_recall_curve returns thresholds one shorter than recall
            chosen = float(thresh[max(0, min(best_idx, len(thresh)-1))]) if len(thresh) else 0.5
            thresholds[label] = chosen

            logger.info(f"Optimized threshold for {label}: {chosen:.3f} (target recall: {target_recall})")

        except Exception as e:
            logger.warning(f"Failed to optimize threshold for {label}: {e}, using default")
            thresholds[label] = 0.5

    return thresholds