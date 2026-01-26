#!/usr/bin/env python3
"""
Hierarchy Post-Processing Evaluation Script

Evaluates the experimental model with and without hierarchy post-processing to measure:
- Safety Recall improvements on critical labels
- Parent-child violation reductions
- Overall model performance impact

Usage:
    python scripts/04_evaluation/evaluate_hierarchy.py
    python scripts/04_evaluation/evaluate_hierarchy.py --model-path experiments/experimental_runs/2025-09-16/2025-09-16-comprehensive-grid-search-optimized-model.pkl
"""

import argparse
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import joblib

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from disasterproject.utils.config import (
    setup_logging, TARGET_COLUMNS, TAXONOMY, CRITICAL_LABELS,
    EXCLUDE_FROM_CONSTRAINTS, DEFAULT_TEST_SIZE, DEFAULT_RANDOM_SEED,
    HIERARCHY_CRITICAL_THRESHOLD_REDUCTION,
)
from disasterproject.data.loader import load_data
from disasterproject.hierarchy import apply_hierarchy, count_violations
from disasterproject.evaluation.metrics import evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class HierarchyEvaluator:
    """Evaluates model performance with and without hierarchy post-processing."""

    def __init__(self, model_path: str, output_dir: str):
        self.model_path = model_path
        self.output_dir = output_dir
        self.model = None
        self.X_test = None
        self.y_test = None
        self.label_names = None
        self.thresholds = {}
        self.effective_thresholds = {}

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def load_model_and_data(self):
        """Load the model and prepare test data."""
        logger.info(f"Loading model from {self.model_path}")
        self.model = joblib.load(self.model_path)

        # Load data and split (same way as training)
        logger.info("Loading and splitting data...")
        X, y = load_data('data/02_stg/stg_disaster_response.db')

        if X is None or y is None:
            raise ValueError("Failed to load data from database")

        self.label_names = TARGET_COLUMNS

        # Split data (consistent with training)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_SEED
        )

        self.X_test = X_test
        self.y_test = y_test  # Already numpy array

        logger.info(f"Test set size: {len(X_test)} samples")
        logger.info(f"Number of labels: {len(self.label_names)}")

    def load_thresholds(self) -> Dict[str, float]:
        """Load thresholds from experimental run or use defaults."""
        # Try to load thresholds from experimental run directory
        exp_dir = Path(self.model_path).parent
        threshold_files = [
            exp_dir / "2025-09-16_thresholds.json",
            exp_dir / "thresholds.json"
        ]

        for threshold_file in threshold_files:
            if threshold_file.exists():
                logger.info(f"Loading thresholds from {threshold_file}")
                with open(threshold_file, 'r') as f:
                    thresholds = json.load(f)
                # Convert to label name keys if needed
                if thresholds and isinstance(list(thresholds.keys())[0], int):
                    # Convert index-based to name-based
                    thresholds = {self.label_names[int(k)]: v for k, v in thresholds.items()}
                # Fill in missing labels with default threshold
                full_thresholds = {label: 0.5 for label in self.label_names}
                full_thresholds.update(thresholds)
                return full_thresholds

        # Default thresholds
        logger.info("Using default thresholds (0.5)")
        return {label: 0.5 for label in self.label_names}

    def _compute_effective_thresholds(self) -> Dict[str, float]:
        """Apply configured critical reduction to base thresholds."""
        effective = dict(self.thresholds)
        if HIERARCHY_CRITICAL_THRESHOLD_REDUCTION > 0:
            for lbl in CRITICAL_LABELS:
                if lbl in effective:
                    effective[lbl] = max(0.0, effective[lbl] - HIERARCHY_CRITICAL_THRESHOLD_REDUCTION)
        return effective

    def get_predictions(self, apply_hierarchy_processing: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Get model predictions with optional hierarchy processing."""
        logger.info(f"Generating predictions (hierarchy={'enabled' if apply_hierarchy_processing else 'disabled'})")

        # Get raw probabilities from model
        if hasattr(self.model, 'predict_proba'):
            # MultiOutputClassifier returns list of arrays
            proba_list = self.model.predict_proba(self.X_test)
            
            # Extract positive class probabilities
            # Access the underlying classifier to get class information
            clf = self.model.named_steps['clf']
            raw_probs_list = []
            
            for i, proba in enumerate(proba_list):
                if proba.ndim == 2 and proba.shape[1] == 2:
                    # Normal binary classifier with both classes
                    raw_probs_list.append(proba[:, 1])
                elif proba.ndim == 2 and proba.shape[1] == 1:
                    # Single class present - check which class it is
                    if hasattr(clf, 'classes_') and i < len(clf.classes_):
                        classes = clf.classes_[i]
                        if len(classes) == 1 and classes[0] == 0:
                            # Only class 0 present, probability of class 1 is 0
                            raw_probs_list.append(np.zeros(proba.shape[0]))
                        elif len(classes) == 1 and classes[0] == 1:
                            # Only class 1 present, probability of class 1 is 1
                            raw_probs_list.append(np.ones(proba.shape[0]))
                        else:
                            # Fallback (shouldn't happen)
                            raw_probs_list.append(proba.ravel())
                    else:
                        # Fallback if class info not available
                        raw_probs_list.append(proba.ravel())
                else:
                    # Fallback for unexpected shapes
                    raw_probs_list.append(proba.ravel())
            
            raw_probs = np.column_stack(raw_probs_list)
        else:
            # Fallback to decision_function if available
            raw_probs = self.model.decision_function(self.X_test)
            # Convert to probabilities using sigmoid
            raw_probs = 1 / (1 + np.exp(-raw_probs))

        if apply_hierarchy_processing:
            return self._apply_hierarchy_to_predictions(raw_probs)
        else:
            return self._apply_thresholds_only(raw_probs)

    def _apply_thresholds_only(self, raw_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply thresholds without hierarchy processing."""
        predictions = np.zeros_like(raw_probs, dtype=int)

        for i, label in enumerate(self.label_names):
            threshold = self.thresholds.get(label, 0.5)
            predictions[:, i] = (raw_probs[:, i] >= threshold).astype(int)

        return raw_probs, predictions

    def _apply_hierarchy_to_predictions(self, raw_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply hierarchy processing to each sample."""
        adjusted_probs = np.zeros_like(raw_probs)
        predictions = np.zeros_like(raw_probs, dtype=int)

        for i in range(len(raw_probs)):
            # Convert to dictionary format for hierarchy function
            prob_dict = {label: raw_probs[i, j] for j, label in enumerate(self.label_names)}

            # Apply hierarchy processing
            adj_probs, binary_preds = apply_hierarchy(
                probs=prob_dict,
                thresholds=self.thresholds,
                taxonomy=TAXONOMY,
                critical_labels=CRITICAL_LABELS,
                exclude=EXCLUDE_FROM_CONSTRAINTS,
                critical_threshold_reduction=HIERARCHY_CRITICAL_THRESHOLD_REDUCTION,
            )

            # Convert back to arrays
            for j, label in enumerate(self.label_names):
                adjusted_probs[i, j] = adj_probs[label]
                predictions[i, j] = binary_preds[label]

        return adjusted_probs, predictions

    def calculate_safety_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Safety Recall (mean recall over critical labels)."""
        safety_recalls = []

        for label in CRITICAL_LABELS:
            if label in self.label_names:
                idx = self.label_names.index(label)
                y_true_label = y_true[:, idx]
                y_pred_label = y_pred[:, idx]

                # Calculate recall for this label
                if y_true_label.sum() > 0:  # Avoid division by zero
                    recall = np.sum((y_true_label == 1) & (y_pred_label == 1)) / y_true_label.sum()
                    safety_recalls.append(recall)

        return np.mean(safety_recalls) if safety_recalls else 0.0

    def _count_edges(self, prob_dict: Dict[str, float]) -> int:
        """Count valid parent→child edges considered for a single sample."""
        total = 0
        for parent, children in TAXONOMY.items():
            if parent == 'related' or parent in EXCLUDE_FROM_CONSTRAINTS:
                continue
            if parent not in prob_dict:
                continue
            for child in children:
                if child in EXCLUDE_FROM_CONSTRAINTS or child not in prob_dict:
                    continue
                total += 1
        return total

    def count_hierarchy_violations_and_edges(self, probs: np.ndarray) -> tuple[int, int]:
        """Count hierarchy violations and total edges considered across all samples."""
        total_violations = 0
        total_edges = 0

        for i in range(len(probs)):
            prob_dict = {label: probs[i, j] for j, label in enumerate(self.label_names)}
            total_violations += count_violations(prob_dict, TAXONOMY, EXCLUDE_FROM_CONSTRAINTS)
            total_edges += self._count_edges(prob_dict)

        return total_violations, total_edges

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, evaluation_type: str) -> Dict:
        """Calculate comprehensive metrics."""
        # Overall metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )

        # Safety recall
        safety_recall = self.calculate_safety_recall(y_true, y_pred)

        # Hierarchy violations (per 1k edges)
        violations, total_edges = self.count_hierarchy_violations_and_edges(probs)
        violations_per_1k = (violations / total_edges * 1000) if total_edges > 0 else 0.0

        return {
            'evaluation_type': evaluation_type,
            'weighted_f1': f1,
            'weighted_precision': precision,
            'weighted_recall': recall,
            'macro_f1': macro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'safety_recall': safety_recall,
            'total_violations': violations,
            'violations_per_1k': violations_per_1k,
            'n_samples': len(y_true)
        }

    def run_evaluation(self) -> Dict:
        """Run complete evaluation with and without hierarchy processing."""
        try:
            logger.info("Loading model and data...")
            self.load_model_and_data()

            logger.info("Loading thresholds...")
            self.thresholds = self.load_thresholds()
            logger.info(f"Loaded {len(self.thresholds)} thresholds")
            # Compute and persist effective thresholds used
            self.effective_thresholds = self._compute_effective_thresholds()

            results = {}

            # Baseline evaluation (no hierarchy)
            logger.info("Running baseline evaluation...")
            baseline_probs, baseline_preds = self.get_predictions(apply_hierarchy_processing=False)
            baseline_metrics = self.calculate_metrics(
                self.y_test, baseline_preds, baseline_probs, 'baseline'
            )
            results['baseline'] = baseline_metrics

            # Hierarchy evaluation
            logger.info("Running hierarchy evaluation...")
            hierarchy_probs, hierarchy_preds = self.get_predictions(apply_hierarchy_processing=True)
            hierarchy_metrics = self.calculate_metrics(
                self.y_test, hierarchy_preds, hierarchy_probs, 'hierarchy'
            )
            results['hierarchy'] = hierarchy_metrics

            logger.info("Note: 'violations per 1k' is edge-normalized (per parent→child edge).")

            return results
        except Exception as e:
            logger.error(f"Error in run_evaluation: {e}", exc_info=True)
            raise

    def save_results(self, results: Dict):
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')

        # Save detailed metrics
        metrics_file = os.path.join(self.output_dir, f"hierarchy_evaluation_{timestamp}.json")
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Detailed results saved to {metrics_file}")

        # Persist thresholds used for hierarchy
        try:
            thresholds_file = os.path.join(self.output_dir, f"thresholds_used_hierarchy_{timestamp}.json")
            with open(thresholds_file, 'w') as f:
                json.dump(self.effective_thresholds or self.thresholds, f, indent=2)
            logger.info(f"Effective thresholds saved to {thresholds_file}")
        except Exception as e:
            logger.warning(f"Failed to save effective thresholds: {e}")

        # Save summary CSV
        summary_data = []
        for eval_type, metrics in results.items():
            summary_data.append(metrics)

        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(self.output_dir, f"hierarchy_summary_{timestamp}.csv")
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Summary saved to {summary_file}")

        return metrics_file, summary_file

    def print_comparison(self, results: Dict):
        """Print comparison results to console."""
        baseline = results['baseline']
        hierarchy = results['hierarchy']

        print("\n" + "="*60)
        print("🔍 HIERARCHY POST-PROCESSING EVALUATION RESULTS")
        print("="*60)

        print(f"\n📊 MODEL: {Path(self.model_path).name}")
        print(f"📅 DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔢 TEST SAMPLES: {baseline['n_samples']:,}")

        print(f"\n📈 PERFORMANCE METRICS:")
        print("-" * 40)
        print(f"{'Metric':<20} {'Baseline':<12} {'Hierarchy':<12} {'Change'}")
        print("-" * 40)

        # Key metrics comparison
        metrics_to_compare = [
            ('Weighted F1', 'weighted_f1'),
            ('Macro F1', 'macro_f1'),
            ('Safety Recall', 'safety_recall'),
            ('Violations/1k', 'violations_per_1k')
        ]

        for display_name, key in metrics_to_compare:
            baseline_val = baseline[key]
            hierarchy_val = hierarchy[key]

            if key == 'violations_per_1k':
                change = hierarchy_val - baseline_val  # Absolute change for violations
                print(f"{display_name:<20} {baseline_val:<12.2f} {hierarchy_val:<12.2f} {change:+.2f}")
            else:
                change_pct = ((hierarchy_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
                print(f"{display_name:<20} {baseline_val:<12.4f} {hierarchy_val:<12.4f} {change_pct:+.2f}%")

        print("\n🎯 KEY FINDINGS:")
        print("-" * 20)

        # Safety recall improvement
        safety_improvement = hierarchy['safety_recall'] - baseline['safety_recall']
        if safety_improvement > 0:
            baseline_sr = baseline['safety_recall']
            if baseline_sr > 0:
                pct = safety_improvement / baseline_sr * 100
                print(
                    f"✅ Safety Recall improved by {safety_improvement:.4f} (+{pct:.1f}%)"
                )
            else:
                # Avoid divide-by-zero when baseline safety recall is 0
                print(
                    f"✅ Safety Recall improved by {safety_improvement:.4f} (baseline was 0)"
                )
        else:
            print(f"❌ Safety Recall decreased by {abs(safety_improvement):.4f}")

        # Violations reduction
        violation_reduction = baseline['violations_per_1k'] - hierarchy['violations_per_1k']
        print(f"✅ Violations reduced by {violation_reduction:.2f} per 1k edges")

        # F1 impact
        f1_change = hierarchy['macro_f1'] - baseline['macro_f1']
        if abs(f1_change) <= 0.02:  # Within 2 points
            print(f"✅ Macro F1 impact within acceptable range ({f1_change:+.4f})")
        else:
            print(f"⚠️  Macro F1 impact outside target range ({f1_change:+.4f})")

        print("\n" + "="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Evaluate hierarchy post-processing')
    parser.add_argument(
        '--model-path',
        default='experiments/experimental_runs/2025-09-16/2025-09-16-comprehensive-grid-search-optimized-model.pkl',
        help='Path to model file'
    )
    parser.add_argument(
        '--output-dir',
        default='experiments/hierarchy_evaluation',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return 1

    # Run evaluation
    evaluator = HierarchyEvaluator(args.model_path, args.output_dir)

    try:
        results = evaluator.run_evaluation()
        metrics_file, summary_file = evaluator.save_results(results)
        evaluator.print_comparison(results)

        print(f"\n📄 Detailed results: {metrics_file}")
        print(f"📄 Summary CSV: {summary_file}")

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"❌ Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
