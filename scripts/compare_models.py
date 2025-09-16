#!/usr/bin/env python3
"""
Simple model comparison tool for disaster response classification.

Compares production model (model/) vs experimental model (experiments/results/)
by directly comparing performance metrics files.

Usage:
    python scripts/compare_models.py
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, TextIO

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class OutputWriter:
    """Helper class to write output to both console and file."""
    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path
        self.file_handle = None
        if file_path:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self.file_handle = open(file_path, 'w', encoding='utf-8')

    def print(self, *args, **kwargs):
        """Print to both console and file."""
        print(*args, **kwargs)
        if self.file_handle:
            print(*args, **kwargs, file=self.file_handle)
            self.file_handle.flush()  # Force write to disk immediately

    def close(self):
        """Close file handle."""
        if self.file_handle:
            self.file_handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def load_metrics(metrics_path: str) -> Optional[pd.DataFrame]:
    """Load performance metrics CSV file."""
    try:
        if os.path.exists(metrics_path):
            return pd.read_csv(metrics_path)
        else:
            print(f"❌ Metrics file not found: {metrics_path}")
            return None
    except Exception as e:
        print(f"❌ Error loading {metrics_path}: {e}")
        return None


def calculate_overall_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate overall metrics from performance DataFrame."""
    weighted_avg = df[df['output_class'] == 'weighted avg']
    return {
        'f1_score': weighted_avg['f1-score'].mean(),
        'precision': weighted_avg['precision'].mean(),
        'recall': weighted_avg['recall'].mean()
    }


def load_model_info(info_path: str) -> Dict:
    """Load model info JSON if it exists."""
    try:
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                return json.load(f)
        return {}
    except Exception:
        return {}


def compare_models(output_writer: OutputWriter):
    """Compare production vs experimental model performance."""
    output_writer.print("🔍 MODEL COMPARISON: Production vs Experimental")
    output_writer.print("=" * 60)
    output_writer.print(f"📅 Comparison Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_writer.print()

    # Define paths
    prod_metrics_path = "model/performance_metrics.csv"
    exp_metrics_path = "experiments/results/performance_metrics.csv"
    prod_info_path = "model/MODEL_INFO.json"
    exp_info_path = "experiments/results/MODEL_INFO.json"

    # Load metrics
    prod_metrics = load_metrics(prod_metrics_path)
    exp_metrics = load_metrics(exp_metrics_path)

    if prod_metrics is None and exp_metrics is None:
        output_writer.print("❌ No metrics files found to compare!")
        return

    output_writer.print("📊 PERFORMANCE COMPARISON:")
    output_writer.print("-" * 40)

    # Production model metrics
    if prod_metrics is not None:
        prod_overall = calculate_overall_metrics(prod_metrics)
        output_writer.print(f"🏭 PRODUCTION MODEL (model/):")
        output_writer.print(f"   F1-Score:   {prod_overall['f1_score']:.4f}")
        output_writer.print(f"   Precision:  {prod_overall['precision']:.4f}")
        output_writer.print(f"   Recall:     {prod_overall['recall']:.4f}")

        # Load additional info
        prod_info = load_model_info(prod_info_path)
        if prod_info.get('model_size_mb'):
            output_writer.print(f"   Model Size: {prod_info['model_size_mb']:.1f} MB")
    else:
        output_writer.print("🏭 PRODUCTION MODEL: No metrics found")

    output_writer.print()

    # Experimental model metrics
    if exp_metrics is not None:
        exp_overall = calculate_overall_metrics(exp_metrics)
        output_writer.print(f"🧪 EXPERIMENTAL MODEL (experiments/results/):")
        output_writer.print(f"   F1-Score:   {exp_overall['f1_score']:.4f}")
        output_writer.print(f"   Precision:  {exp_overall['precision']:.4f}")
        output_writer.print(f"   Recall:     {exp_overall['recall']:.4f}")

        # Load additional info
        exp_info = load_model_info(exp_info_path)
        if exp_info.get('model_size_mb'):
            output_writer.print(f"   Model Size: {exp_info['model_size_mb']:.1f} MB")
    else:
        output_writer.print("🧪 EXPERIMENTAL MODEL: No metrics found")

    # Calculate improvements if both models exist
    if prod_metrics is not None and exp_metrics is not None:
        output_writer.print(f"\n📈 PERFORMANCE CHANGES:")
        output_writer.print("-" * 25)

        f1_change = ((exp_overall['f1_score'] - prod_overall['f1_score']) / prod_overall['f1_score']) * 100
        precision_change = ((exp_overall['precision'] - prod_overall['precision']) / prod_overall['precision']) * 100
        recall_change = ((exp_overall['recall'] - prod_overall['recall']) / prod_overall['recall']) * 100

        output_writer.print(f"   F1-Score:   {f1_change:+.2f}%")
        output_writer.print(f"   Precision:  {precision_change:+.2f}%")
        output_writer.print(f"   Recall:     {recall_change:+.2f}%")

        # Overall verdict
        output_writer.print(f"\n🏆 VERDICT:")
        if f1_change > 0:
            output_writer.print("   ✅ Experimental model shows improvement!")
        elif f1_change < 0:
            output_writer.print("   ⚠️  Experimental model shows decline")
        else:
            output_writer.print("   ➖ Models show identical performance")

    output_writer.print("\n" + "=" * 60)


def show_detailed_category_comparison(output_writer: OutputWriter):
    """Show detailed per-category performance comparison."""
    output_writer.print("\n🔍 DETAILED CATEGORY COMPARISON")
    output_writer.print("=" * 50)

    prod_metrics = load_metrics("model/performance_metrics.csv")
    exp_metrics = load_metrics("experiments/results/performance_metrics.csv")

    if prod_metrics is None or exp_metrics is None:
        output_writer.print("❌ Need both model metrics files for detailed comparison")
        return

    # Focus on weighted avg for each category
    prod_weighted = prod_metrics[prod_metrics['output_class'] == 'weighted avg'][['category', 'f1-score', 'precision', 'recall']]
    exp_weighted = exp_metrics[exp_metrics['output_class'] == 'weighted avg'][['category', 'f1-score', 'precision', 'recall']]

    # Merge and calculate differences
    comparison = prod_weighted.merge(exp_weighted, on='category', suffixes=('_prod', '_exp'))
    comparison['f1_diff'] = ((comparison['f1-score_exp'] - comparison['f1-score_prod']) / comparison['f1-score_prod'] * 100)
    comparison = comparison.sort_values('f1_diff', ascending=False)

    output_writer.print("Categories with biggest improvements:")
    output_writer.print("-" * 40)
    top_improved = comparison.head(5)
    for _, row in top_improved.iterrows():
        if row['f1_diff'] > 0:
            output_writer.print(f"✅ {row['category']:<20} {row['f1_diff']:+.1f}% F1 improvement")

    output_writer.print("\nCategories with biggest declines:")
    output_writer.print("-" * 40)
    worst_declined = comparison.tail(5)
    for _, row in worst_declined.iterrows():
        if row['f1_diff'] < 0:
            output_writer.print(f"❌ {row['category']:<20} {row['f1_diff']:+.1f}% F1 decline")


def main():
    """Main comparison interface."""
    # Create output file with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    output_file = f"experiments/comparisons/{timestamp}_model_comparison.txt"

    # Create output writer for both console and file using context manager
    with OutputWriter(output_file) as output_writer:
        if len(sys.argv) > 1 and sys.argv[1] == '--detailed':
            show_detailed_category_comparison(output_writer)
        else:
            compare_models(output_writer)

            # Ask if user wants detailed view
            try:
                detailed = input("\nShow detailed per-category comparison? (y/N): ").strip().lower()
                if detailed in ['y', 'yes']:
                    show_detailed_category_comparison(output_writer)
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye! 👋")

        output_writer.print(f"\n📄 Full comparison report saved to: {output_file}")

    print(f"📄 Full comparison report saved to: {output_file}")


if __name__ == "__main__":
    main()