"""
Unit tests for hierarchy post-processor functionality.

Tests cover:
- Probability monotonicity enforcement
- Decision-level child→parent forcing
- Exclusion handling for problematic labels
- Critical threshold optimization
- Special handling for 'related' group
"""

import numpy as np

from src.disasterproject.hierarchy import (
    apply_hierarchy,
    count_violations,
    optimize_critical_thresholds,
)


class TestApplyHierarchy:
    """Test the main apply_hierarchy function."""

    def test_probability_monotonicity(self):
        """Test that parent probabilities are boosted to max(children)."""
        probs = {
            "aid_related": 0.3,
            "medical_help": 0.8,
            "water": 0.6,
            "other_aid": 0.2
        }

        thresholds = {label: 0.5 for label in probs}

        taxonomy = {
            "aid_related": ["medical_help", "water", "other_aid"]
        }

        adjusted_probs, predictions = apply_hierarchy(
            probs, thresholds, taxonomy, set(), set()
        )

        # Parent should be boosted to max child probability
        assert adjusted_probs["aid_related"] == 0.8
        # Children should remain unchanged (they were already ≤ parent after boost)
        assert adjusted_probs["medical_help"] == 0.8
        assert adjusted_probs["water"] == 0.6
        assert adjusted_probs["other_aid"] == 0.2

    def test_child_probability_clamping(self):
        """Test that child probabilities are clamped to parent when they exceed it."""
        probs = {
            "infrastructure_related": 0.4,
            "hospitals": 0.7,
            "transport": 0.3
        }

        thresholds = {label: 0.5 for label in probs}

        taxonomy = {
            "infrastructure_related": ["hospitals", "transport"]
        }

        adjusted_probs, predictions = apply_hierarchy(
            probs, thresholds, taxonomy, set(), set()
        )

        # Parent should be boosted to max child
        assert adjusted_probs["infrastructure_related"] == 0.7
        # High child should remain at max (equals parent after boost)
        assert adjusted_probs["hospitals"] == 0.7
        # Low child should remain unchanged
        assert adjusted_probs["transport"] == 0.3

    def test_decision_level_forcing(self):
        """Test that child=1 forces parent=1 regardless of probability."""
        probs = {
            "aid_related": 0.2,
            "medical_help": 0.8,  # Above threshold
            "water": 0.3          # Below threshold
        }

        thresholds = {
            "aid_related": 0.5,
            "medical_help": 0.5,
            "water": 0.5
        }

        taxonomy = {
            "aid_related": ["medical_help", "water"]
        }

        adjusted_probs, predictions = apply_hierarchy(
            probs, thresholds, taxonomy, set(), set()
        )

        # medical_help should predict 1 (0.8 ≥ 0.5)
        assert predictions["medical_help"] == 1
        # water should predict 0 (0.3 < 0.5)
        assert predictions["water"] == 0
        # aid_related should be forced to 1 due to medical_help=1
        assert predictions["aid_related"] == 1

    def test_related_group_special_handling(self):
        """Test that 'related' group only applies decision-level forcing, no probability clamping."""
        probs = {
            "related": 0.2,
            "request": 0.8,
            "offer": 0.3
        }

        thresholds = {label: 0.5 for label in probs}

        taxonomy = {
            "related": ["request", "offer"]
        }

        adjusted_probs, predictions = apply_hierarchy(
            probs, thresholds, taxonomy, set(), set()
        )

        # Probabilities should NOT be adjusted for 'related' group
        assert adjusted_probs["related"] == 0.2  # Unchanged
        assert adjusted_probs["request"] == 0.8  # Unchanged
        assert adjusted_probs["offer"] == 0.3    # Unchanged

        # But decision-level forcing should still apply
        assert predictions["request"] == 1  # 0.8 ≥ 0.5
        assert predictions["offer"] == 0    # 0.3 < 0.5
        assert predictions["related"] == 1  # Forced due to request=1

    def test_exclusion_handling(self):
        """Test that excluded labels are not affected by hierarchy constraints."""
        probs = {
            "aid_related": 0.2,
            "child_alone": 0.9,  # High prob but excluded
            "medical_help": 0.3
        }

        thresholds = {label: 0.5 for label in probs}

        taxonomy = {
            "aid_related": ["child_alone", "medical_help"]
        }

        exclude = {"child_alone"}

        adjusted_probs, predictions = apply_hierarchy(
            probs, thresholds, taxonomy, set(), exclude
        )

        # child_alone should be excluded from hierarchy constraints
        assert adjusted_probs["child_alone"] == 0.9  # Unchanged
        assert predictions["child_alone"] == 1       # Based on threshold only

        # aid_related should only consider non-excluded children (medical_help)
        assert adjusted_probs["aid_related"] == 0.3  # Boosted to medical_help
        assert predictions["aid_related"] == 0       # Not forced by excluded child

    def test_critical_threshold_reduction(self):
        """Test that critical labels get reduced thresholds."""
        probs = {
            "medical_help": 0.4,  # Below normal threshold but critical
            "water": 0.4          # Below threshold, not critical
        }

        thresholds = {
            "medical_help": 0.5,
            "water": 0.5
        }

        critical_labels = {"medical_help"}

        adjusted_probs, predictions = apply_hierarchy(
            probs, thresholds, {}, critical_labels, set(), critical_threshold_reduction=0.2
        )

        # medical_help should predict 1 with reduced threshold (0.4 ≥ 0.3)
        assert predictions["medical_help"] == 1
        # water should predict 0 with normal threshold (0.4 < 0.5)
        assert predictions["water"] == 0

    def test_no_changes_when_consistent(self):
        """Test that already consistent hierarchies are not modified."""
        probs = {
            "aid_related": 0.8,
            "medical_help": 0.6,
            "water": 0.4
        }

        thresholds = {label: 0.5 for label in probs}

        taxonomy = {
            "aid_related": ["medical_help", "water"]
        }

        adjusted_probs, predictions = apply_hierarchy(
            probs, thresholds, taxonomy, set(), set()
        )

        # All probabilities should remain unchanged
        assert adjusted_probs == probs
        # Predictions should be based on original thresholds
        assert predictions["aid_related"] == 1
        assert predictions["medical_help"] == 1
        assert predictions["water"] == 0


class TestCountViolations:
    """Test the violation counting function."""

    def test_violation_counting(self):
        """Test that violations are correctly counted."""
        probs = {
            "aid_related": 0.3,
            "medical_help": 0.8,  # Violation: child > parent
            "water": 0.2,         # No violation: child < parent
            "infrastructure_related": 0.6,
            "hospitals": 0.7      # Violation: child > parent
        }

        taxonomy = {
            "aid_related": ["medical_help", "water"],
            "infrastructure_related": ["hospitals"]
        }

        violations = count_violations(probs, taxonomy, set())
        assert violations == 2  # medical_help and hospitals violations

    def test_related_group_skipped(self):
        """Test that 'related' group violations are not counted."""
        probs = {
            "related": 0.2,
            "request": 0.8  # Would be violation but 'related' is skipped
        }

        taxonomy = {
            "related": ["request"]
        }

        violations = count_violations(probs, taxonomy, set())
        assert violations == 0  # 'related' group is skipped

    def test_exclusions_respected(self):
        """Test that excluded labels don't contribute to violation count."""
        probs = {
            "aid_related": 0.2,
            "child_alone": 0.8  # Would be violation but excluded
        }

        taxonomy = {
            "aid_related": ["child_alone"]
        }

        exclude = {"child_alone"}

        violations = count_violations(probs, taxonomy, exclude)
        assert violations == 0  # child_alone is excluded


class TestOptimizeCriticalThresholds:
    """Test threshold optimization for critical labels."""

    def test_threshold_optimization(self):
        """Test basic threshold optimization functionality."""
        # Create simple test data
        np.random.seed(42)
        n_samples = 100
        n_labels = 2

        # Create y_true with some positive examples
        y_true = np.zeros((n_samples, n_labels))
        y_true[:20, 0] = 1  # First 20 samples positive for label 0

        # Create y_proba that should allow good threshold finding
        y_proba = np.random.random((n_samples, n_labels))
        y_proba[:20, 0] = np.random.uniform(0.6, 1.0, 20)  # High probs for positives

        label_names = ["medical_help", "water"]
        critical_labels = {"medical_help"}

        thresholds = optimize_critical_thresholds(
            y_true, y_proba, label_names, critical_labels, target_recall=0.8
        )

        assert "medical_help" in thresholds
        assert 0.0 <= thresholds["medical_help"] <= 1.0

    def test_no_positive_examples(self):
        """Test handling of labels with no positive examples."""
        y_true = np.zeros((50, 2))  # No positive examples
        y_proba = np.random.random((50, 2))

        label_names = ["medical_help", "water"]
        critical_labels = {"medical_help"}

        thresholds = optimize_critical_thresholds(
            y_true, y_proba, label_names, critical_labels
        )

        # Should default to 0.5 when no positive examples
        assert thresholds["medical_help"] == 0.5

    def test_missing_label(self):
        """Test handling of critical labels not in label_names."""
        y_true = np.random.randint(0, 2, (50, 1))
        y_proba = np.random.random((50, 1))

        label_names = ["water"]
        critical_labels = {"medical_help"}  # Not in label_names

        thresholds = optimize_critical_thresholds(
            y_true, y_proba, label_names, critical_labels
        )

        # Should skip missing labels
        assert "medical_help" not in thresholds


# Integration test
def test_end_to_end_hierarchy_processing():
    """Test complete hierarchy processing pipeline."""
    # Setup test data mimicking real scenario
    probs = {
        "aid_related": 0.2,
        "medical_help": 0.7,      # Critical, should be easier to activate
        "water": 0.4,             # Critical, should be easier to activate
        "other_aid": 0.1,
        "infrastructure_related": 0.3,
        "hospitals": 0.8,
        "child_alone": 0.9        # Excluded from constraints
    }

    thresholds = {label: 0.5 for label in probs}

    taxonomy = {
        "aid_related": ["medical_help", "water", "other_aid"],
        "infrastructure_related": ["hospitals"]
    }

    critical_labels = {"medical_help", "water"}
    exclude = {"child_alone"}

    # Apply hierarchy processing
    adjusted_probs, predictions = apply_hierarchy(
        probs, thresholds, taxonomy, critical_labels, exclude,
        critical_threshold_reduction=0.2
    )

    # Verify hierarchy consistency
    violations = count_violations(adjusted_probs, taxonomy, exclude)
    assert violations == 0, "Should have no violations after processing"

    # Verify critical labels benefit from reduced thresholds
    assert predictions["medical_help"] == 1  # 0.7 ≥ (0.5-0.2)
    assert predictions["water"] == 1         # 0.4 ≥ (0.5-0.2)

    # Verify decision-level forcing
    assert predictions["aid_related"] == 1   # Forced by children
    assert predictions["infrastructure_related"] == 1  # Forced by hospitals

    # Verify exclusions are respected
    assert adjusted_probs["child_alone"] == 0.9  # Unchanged
    assert predictions["child_alone"] == 1       # Based on threshold only
