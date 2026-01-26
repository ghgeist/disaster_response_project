"""
Hierarchy-related helper functions for classification routes.
"""
from typing import Dict, Iterable, List, Mapping, Set


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
