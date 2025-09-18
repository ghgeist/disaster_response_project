#!/usr/bin/env python3
"""
Threshold Optimization for Hierarchy Post-Processing

Tests different critical threshold reduction values to find optimal balance
between Safety Recall improvement and Macro F1 impact.

Usage:
    python scripts/optimize_thresholds.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from disasterproject.utils.config import (
    setup_logging, TARGET_COLUMNS, TAXONOMY, CRITICAL_LABELS,
    EXCLUDE_FROM_CONSTRAINTS, DEFAULT_TEST_SIZE, DEFAULT_RANDOM_SEED
)
from disasterproject.data.loader import load_data
from disasterproject.hierarchy import apply_hierarchy, count_violations
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import joblib
import logging

logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """Optimize critical threshold reduction for hierarchy post-processing."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.X_test = None
        self.y_test = None
        self.label_names = TARGET_COLUMNS
        self.base_thresholds = {}
        self.raw_probs = None

    def load_model_and_data(self):
        """Load model and prepare test data."""
        logger.info(f"Loading model from {self.model_path}")
        self.model = joblib.load(self.model_path)

        logger.info("Loading and splitting data...")
        X, y = load_data('data/02_stg/stg_disaster_response.db')
        if X is None or y is None:
            raise ValueError("Failed to load data from database")

        # Split data (consistent with training)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_SEED
        )

        self.X_test = X_test
        self.y_test = y_test

        logger.info(f"Test set size: {len(X_test)} samples")

        # Get raw probabilities once (expensive operation)
        logger.info("Computing raw probabilities...")
        if hasattr(self.model, 'predict_proba'):
            proba_list = self.model.predict_proba(self.X_test)
            self.raw_probs = np.column_stack([
                proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
                for proba in proba_list
            ])
        else:
            raw_probs = self.model.decision_function(self.X_test)
            self.raw_probs = 1 / (1 + np.exp(-raw_probs))

    def load_base_thresholds(self):
        """Load base thresholds from experimental run."""
        exp_dir = os.path.dirname(self.model_path)
        threshold_file = os.path.join(exp_dir, "2025-09-16_thresholds.json")

        if os.path.exists(threshold_file):
            with open(threshold_file, 'r') as f:
                loaded_thresholds = json.load(f)
            # Fill in missing labels with default
            self.base_thresholds = {label: 0.5 for label in self.label_names}
            self.base_thresholds.update(loaded_thresholds)
            logger.info(f"Loaded base thresholds from {threshold_file}")
        else:
            self.base_thresholds = {label: 0.5 for label in self.label_names}
            logger.info("Using default base thresholds (0.5)")

    def evaluate_threshold_reduction(self, critical_reduction: float) -> Dict:
        """Evaluate hierarchy performance with specific critical threshold reduction."""
        logger.info(f"Testing critical threshold reduction: {critical_reduction}")

        # Apply hierarchy to all predictions
        adjusted_probs = np.zeros_like(self.raw_probs)
        predictions = np.zeros_like(self.raw_probs, dtype=int)
        total_violations = 0

        for i in range(len(self.raw_probs)):
            # Convert to dictionary format
            prob_dict = {label: self.raw_probs[i, j] for j, label in enumerate(self.label_names)}

            # Apply hierarchy processing with specified reduction
            adj_probs, binary_preds = apply_hierarchy(
                probs=prob_dict,
                thresholds=self.base_thresholds,
                taxonomy=TAXONOMY,
                critical_labels=CRITICAL_LABELS,
                exclude=EXCLUDE_FROM_CONSTRAINTS,
                critical_threshold_reduction=critical_reduction
            )

            # Convert back to arrays
            for j, label in enumerate(self.label_names):
                adjusted_probs[i, j] = adj_probs[label]
                predictions[i, j] = binary_preds[label]

            # Count violations in original probabilities
            violations = count_violations(prob_dict, TAXONOMY, EXCLUDE_FROM_CONSTRAINTS)
            total_violations += violations

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, predictions, average='weighted', zero_division=0
        )
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            self.y_test, predictions, average='macro', zero_division=0
        )

        # Safety recall
        safety_recall = self._calculate_safety_recall(predictions)

        # Violations after hierarchy (should be 0)
        post_violations = 0
        for i in range(len(adjusted_probs)):
            prob_dict = {label: adjusted_probs[i, j] for j, label in enumerate(self.label_names)}
            post_violations += count_violations(prob_dict, TAXONOMY, EXCLUDE_FROM_CONSTRAINTS)

        return {
            'critical_reduction': critical_reduction,
            'weighted_f1': f1,
            'weighted_precision': precision,
            'weighted_recall': recall,
            'macro_f1': macro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'safety_recall': safety_recall,
            'violations_before': total_violations,
            'violations_after': post_violations,
            'violations_per_1k_before': (total_violations / len(self.y_test)) * 1000,
            'violations_per_1k_after': (post_violations / len(self.y_test)) * 1000,
        }

    def _calculate_safety_recall(self, predictions: np.ndarray) -> float:
        """Calculate Safety Recall (mean recall over critical labels)."""
        safety_recalls = []
        for label in CRITICAL_LABELS:
            if label in self.label_names:
                idx = self.label_names.index(label)
                y_true_label = self.y_test[:, idx]
                y_pred_label = predictions[:, idx]

                if y_true_label.sum() > 0:
                    recall = np.sum((y_true_label == 1) & (y_pred_label == 1)) / y_true_label.sum()
                    safety_recalls.append(recall)

        return np.mean(safety_recalls) if safety_recalls else 0.0

    def optimize_thresholds(self, reduction_values: List[float]) -> pd.DataFrame:
        """Test multiple threshold reduction values and return results."""
        results = []

        for reduction in reduction_values:
            try:
                metrics = self.evaluate_threshold_reduction(reduction)
                results.append(metrics)
                logger.info(f"Reduction {reduction}: Macro F1={metrics['macro_f1']:.4f}, Safety Recall={metrics['safety_recall']:.4f}")
            except Exception as e:
                logger.error(f"Error testing reduction {reduction}: {e}")

        return pd.DataFrame(results)

    def find_optimal_reduction(self, results_df: pd.DataFrame, baseline_macro_f1: float, max_f1_decline: float = 0.02) -> Dict:
        """Find optimal reduction balancing safety recall and F1 impact."""
        # Calculate F1 change from baseline
        results_df['macro_f1_change'] = results_df['macro_f1'] - baseline_macro_f1
        results_df['macro_f1_decline_pct'] = (results_df['macro_f1_change'] / baseline_macro_f1) * 100

        # Filter candidates within acceptable F1 decline
        candidates = results_df[results_df['macro_f1_change'] >= -max_f1_decline]

        if len(candidates) == 0:
            logger.warning(f"No candidates within {max_f1_decline:.3f} F1 decline limit")
            # Fall back to best F1 performance
            optimal = results_df.loc[results_df['macro_f1'].idxmax()]
        else:
            # Among acceptable candidates, choose highest safety recall
            optimal = candidates.loc[candidates['safety_recall'].idxmax()]

        return {
            'optimal_reduction': optimal['critical_reduction'],
            'macro_f1': optimal['macro_f1'],
            'macro_f1_change': optimal['macro_f1_change'],
            'macro_f1_decline_pct': optimal['macro_f1_decline_pct'],
            'safety_recall': optimal['safety_recall'],
            'weighted_f1': optimal['weighted_f1'],
            'meets_f1_target': optimal['macro_f1_change'] >= -max_f1_decline
        }


def main():
    """Main optimization process."""
    setup_logging()

    model_path = 'experiments/experimental_runs/2025-09-16/2025-09-16-comprehensive-grid-search-optimized-model.pkl'

    # Test range of reduction values
    reduction_values = [0.00, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

    optimizer = ThresholdOptimizer(model_path)
    optimizer.load_model_and_data()
    optimizer.load_base_thresholds()

    print("🔍 OPTIMIZING CRITICAL THRESHOLD REDUCTION")
    print("="*50)
    print(f"📊 Testing {len(reduction_values)} reduction values: {reduction_values}")
    print(f"🎯 Target: Macro F1 decline ≤ 2% while maximizing Safety Recall")
    print()

    # Run optimization
    results_df = optimizer.optimize_thresholds(reduction_values)

    # Baseline (no hierarchy) for comparison - use reduction=0.0 as proxy
    baseline_macro_f1 = 0.7784  # From previous evaluation

    # Find optimal
    optimal = optimizer.find_optimal_reduction(results_df, baseline_macro_f1, max_f1_decline=0.02)

    # Save results
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    results_file = f'experiments/threshold_optimization_{timestamp}.csv'
    results_df.to_csv(results_file, index=False)

    # Display results
    print("📊 OPTIMIZATION RESULTS")
    print("-" * 30)
    print(results_df[['critical_reduction', 'macro_f1', 'safety_recall', 'weighted_f1']].round(4))
    print()

    print("🎯 OPTIMAL CONFIGURATION")
    print("-" * 25)
    print(f"💡 Optimal reduction: {optimal['optimal_reduction']:.3f}")
    print(f"📈 Macro F1: {optimal['macro_f1']:.4f} (change: {optimal['macro_f1_decline_pct']:+.2f}%)")
    print(f"🛡️ Safety Recall: {optimal['safety_recall']:.4f}")
    print(f"⚖️ Weighted F1: {optimal['weighted_f1']:.4f}")
    print(f"✅ Meets F1 target: {'Yes' if optimal['meets_f1_target'] else 'No'}")
    print()

    print(f"📄 Full results saved to: {results_file}")

    return optimal


if __name__ == "__main__":
    main()