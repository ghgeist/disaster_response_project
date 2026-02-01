#!/usr/bin/env python3
"""
Prepare 15K vocabulary model for production promotion.

This script:
1. Creates a dedicated promotion directory with the 15K model
2. Copies optimized thresholds with standard naming
3. Ensures all required artifacts are in place
"""

# Standard library imports
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
exp_dir = project_root / "experiments" / "experimental_runs" / "2025-11-06"
promo_dir = project_root / "experiments" / "experimental_runs" / "2025-11-06-vocab15k-promotion"

# Model and threshold paths
model_file = exp_dir / "lr_vocab15k_model.pkl"
threshold_file = exp_dir / "vocab15k" / "optimized_critical_thresholds.json"
training_log = exp_dir / "training_log.json"
performance_metrics = exp_dir / "performance_metrics.csv"
label_order = exp_dir / "label_order.json"

print("="*70)
print("PREPARING 15K VOCAB MODEL FOR PRODUCTION PROMOTION")
print("="*70)

# Create promotion directory
promo_dir.mkdir(parents=True, exist_ok=True)
print(f"\n✓ Created promotion directory: {promo_dir}")

# Copy model file
if not model_file.exists():
    print(f"❌ Model file not found: {model_file}")
    sys.exit(1)

promo_model = promo_dir / "lr_vocab15k_model.pkl"
shutil.copy2(model_file, promo_model)
print(f"✓ Copied model: {promo_model.name}")

# Copy and rename thresholds with standard naming
if not threshold_file.exists():
    print(f"❌ Threshold file not found: {threshold_file}")
    sys.exit(1)

# Load threshold data to ensure it's valid
with open(threshold_file, 'r') as f:
    threshold_data = json.load(f)

# Save with standard naming: {model_stem}_thresholds.json
model_stem = "lr_vocab15k_model"
standard_thresholds = promo_dir / f"{model_stem}_thresholds.json"
with open(standard_thresholds, 'w') as f:
    json.dump(threshold_data, f, indent=2)
print(f"✓ Copied thresholds: {standard_thresholds.name}")

# Also save legacy name for compatibility
legacy_thresholds = promo_dir / "optimized_critical_thresholds.json"
shutil.copy2(threshold_file, legacy_thresholds)
print(f"  (Also saved as: {legacy_thresholds.name} for compatibility)")

# Copy training log if available
if training_log.exists():
    promo_training_log = promo_dir / "training_log.json"
    shutil.copy2(training_log, promo_training_log)
    print(f"✓ Copied training log: {promo_training_log.name}")

# Copy performance metrics if available
if performance_metrics.exists():
    promo_metrics = promo_dir / "performance_metrics.csv"
    shutil.copy2(performance_metrics, promo_metrics)
    print(f"✓ Copied performance metrics: {promo_metrics.name}")

# Copy label order if available
if label_order.exists():
    promo_labels = promo_dir / "label_order.json"
    shutil.copy2(label_order, promo_labels)
    print(f"✓ Copied label order: {promo_labels.name}")

# Create promotion info file
promo_info = {
    "model": "lr_vocab15k_model.pkl",
    "vocabulary": {
        "max_features": 15000,
        "min_df": 3,
        "max_df": 0.90
    },
    "performance": {
        "f1_weighted": 0.9276,
        "critical_recall": 0.6497,
        "model_size_mb": 4.53
    },
    "prepared_at": datetime.now().isoformat(),
    "source": {
        "experiment_dir": str(exp_dir),
        "thresholds_source": str(threshold_file)
    }
}

info_file = promo_dir / "PROMOTION_INFO.json"
with open(info_file, 'w') as f:
    json.dump(promo_info, f, indent=2)
print(f"✓ Created promotion info: {info_file.name}")

print("\n" + "="*70)
print("✅ PREPARATION COMPLETE")
print("="*70)
print(f"\nPromotion directory: {promo_dir}")
print("\nNext steps:")
print(f"1. Validate: python scripts/promote_model.py {promo_dir} --dry-run")
print(f"2. Promote: python scripts/promote_model.py {promo_dir} --print-new-path")
print("\n" + "="*70)

