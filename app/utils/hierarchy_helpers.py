"""
Hierarchy-related helper functions for classification routes.
"""
from typing import Dict, Iterable, List, Mapping, Set, Tuple

from disasterproject.hierarchy import apply_hierarchy
from disasterproject.utils.config import (
    CRITICAL_LABELS,
    EXCLUDE_FROM_CONSTRAINTS,
    HIERARCHY_CRITICAL_THRESHOLD_REDUCTION,
    TAXONOMY,
)


def compute_violations(
    probs: Mapping[str, float],
    taxonomy: Mapping[str, Iterable[str]],
    exclude_set: Set[str],
) -> List[Dict[str, float]]:
    """
    Compute parent < child violations for display in the diff table.

    Args:
        probs: Dictionary mapping label names to probabilities.
        taxonomy: Dictionary mapping parent labels to list of child labels.
        exclude_set: Set of labels to exclude from violation checks.

    Returns:
        List of violation dictionaries with parent, child, parent_prob, child_prob.
    """
    violations: List[Dict[str, float]] = []

    for parent, children in taxonomy.items():
        if parent == "related":
            continue

        if parent in exclude_set:
            continue

        valid_children = [
            child for child in children if child in probs and child not in exclude_set
        ]

        if not valid_children or parent not in probs:
            continue

        for child in valid_children:
            if probs[child] > probs[parent]:
                violations.append(
                    {
                        "parent": parent,
                        "child": child,
                        "parent_prob": probs[parent],
                        "child_prob": probs[child],
                    }
                )

    return violations


def run_hierarchy_correction(
    probs: Dict[str, float],
    thresholds: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Apply production taxonomy hierarchy correction to probabilities and labels.

    Uses the shared TAXONOMY / CRITICAL_LABELS / EXCLUDE_FROM_CONSTRAINTS config
    so /api/classify and /classify stay aligned.
    """
    return apply_hierarchy(
        probs=probs,
        thresholds=thresholds,
        taxonomy=TAXONOMY,
        critical_labels=CRITICAL_LABELS,
        exclude=EXCLUDE_FROM_CONSTRAINTS,
        critical_threshold_reduction=HIERARCHY_CRITICAL_THRESHOLD_REDUCTION,
    )
