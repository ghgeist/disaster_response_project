#!/usr/bin/env python3
"""
Model Promotion Script for Disaster Response System

Implements MLOps best practices for promoting experimental models to production:
- Validates candidate model performance
- Archives current production model metadata
- Promotes new model with proper versioning
- Maintains model registry and lineage
"""

import argparse
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
import sys
import os
from typing import Optional, Tuple

import pandas as pd

# Add src to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from disasterproject.utils.config import PERFORMANCE_THRESHOLDS


def compute_model_hash(model_path: Path) -> str:
    """Compute SHA256 hash of model file for integrity verification."""
    sha256_hash = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def _load_training_log(candidate_dir: Path) -> Optional[dict]:
    """Load training_log.json if present."""
    for name in ["training_log.json", f"{candidate_dir.name}_training_log.json"]:
        p = candidate_dir / name
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
    return None


def _parse_metrics_from_training_log(log_data: dict) -> Optional[Tuple[float, float]]:
    """Extract (f1_weighted, f1_micro) from training log structure."""
    try:
        perf = log_data.get("performance") or {}
        # Map to thresholds: use overall_f1 (weighted avg across categories) for f1_weighted
        f1_weighted = float(perf.get("overall_f1")) if perf.get("overall_f1") is not None else None
        # Prefer explicit micro metrics when available; otherwise fall back to weighted F1
        f1_micro = (
            perf.get("micro_f1")
            or perf.get("f1_micro")
            or perf.get("samples_f1")
            or None
        )
        f1_micro = float(f1_micro) if f1_micro is not None else f1_weighted
        if f1_weighted is None:
            return None
        return f1_weighted, f1_micro
    except Exception:
        return None


def _parse_metrics_from_csv(metrics_csv: Path) -> Optional[Tuple[float, float]]:
    """Compute (f1_weighted, f1_micro≈positive_class_f1) from performance_metrics.csv."""
    try:
        df = pd.read_csv(metrics_csv)
        if df.empty:
            return None
        # f1_weighted = mean of 'weighted avg' f1 across categories
        w = df[df["output_class"].astype(str).str.lower() == "weighted avg"]["f1-score"].astype(float)
        f1_weighted = float(w.mean()) if not w.empty else None
        # True micro-F1 cannot be reconstructed from per-label CSV; use weighted F1
        f1_micro = f1_weighted
        if f1_weighted is None:
            return None
        return f1_weighted, f1_micro
    except Exception:
        return None


def _discover_metrics(candidate_dir: Path) -> Tuple[float, float]:
    """Discover f1_weighted and f1_micro using multiple fallbacks."""
    # 1) training_log.json
    log_data = _load_training_log(candidate_dir)
    if log_data:
        parsed = _parse_metrics_from_training_log(log_data)
        if parsed:
            return parsed
    # 2) performance_metrics.csv (several naming patterns)
    candidates = [
        candidate_dir / "performance_metrics.csv",
        candidate_dir / f"{candidate_dir.name}_performance_metrics.csv",
        candidate_dir / f"{candidate_dir.name.replace('-', '_')}_performance_metrics.csv",
    ]
    for p in candidates:
        if p.exists():
            parsed = _parse_metrics_from_csv(p)
            if parsed:
                return parsed
    raise FileNotFoundError("Unable to discover metrics (training_log.json or performance_metrics.csv)")


def _discover_model_file(candidate_dir: Path) -> Path:
    """Find exactly one .pkl model file in candidate_dir."""
    model_files = list(candidate_dir.glob("*.pkl"))
    if not model_files:
        raise FileNotFoundError(f"No model file (.pkl) found in {candidate_dir}")
    # Prefer the most recent file
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return model_files[0]


def validate_candidate_model(candidate_dir: Path) -> dict:
    """Validate that candidate model meets promotion criteria (robust discovery)."""

    # Discover metrics using flexible inputs
    f1_weighted, f1_micro = _discover_metrics(candidate_dir)

    # Find model file and compute size/hash
    model_file = _discover_model_file(candidate_dir)
    model_size_mb = model_file.stat().st_size / (1024 * 1024)

    # Validation criteria
    min_f1_weighted = PERFORMANCE_THRESHOLDS.get('min_f1_weighted', 0.5)
    min_f1_micro = PERFORMANCE_THRESHOLDS.get('min_f1_micro', 0.6)
    max_model_size_mb = PERFORMANCE_THRESHOLDS.get('max_model_size_mb', 1000)

    validation_results = {
        'model_path': str(model_file),
        'model_size_mb': model_size_mb,
        'f1_weighted': f1_weighted,
        'f1_micro': f1_micro,
        'model_hash': compute_model_hash(model_file),
        'validation_passed': True,
        'validation_errors': []
    }

    # Check performance thresholds
    if f1_weighted < min_f1_weighted:
        validation_results['validation_errors'].append(
            f"F1 weighted {f1_weighted:.4f} below threshold {min_f1_weighted}")

    if f1_micro < min_f1_micro:
        validation_results['validation_errors'].append(
            f"F1 micro {f1_micro:.4f} below threshold {min_f1_micro}")

    if model_size_mb > max_model_size_mb:
        validation_results['validation_errors'].append(
            f"Model size {model_size_mb:.1f}MB exceeds limit {max_model_size_mb}MB")

    validation_results['validation_passed'] = len(validation_results['validation_errors']) == 0

    return validation_results


def archive_current_production_model(model_dir: Path, archive_dir: Path) -> dict:
    """Archive current production model metadata to model registry."""

    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Find current production model
    prod_models = list(model_dir.glob("*_prod_*.pkl"))
    if not prod_models:
        print("No current production model found to archive")
        return {}

    if len(prod_models) > 1:
        print(f"Warning: Multiple production models found: {prod_models}")
        print("Archiving the most recent one")
        prod_models.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    current_prod_model = prod_models[0]
    base_name = current_prod_model.stem

    # Archive metadata files (not the large .pkl file)
    archived_metadata = {}
    metadata_extensions = ['.json', '_labels.json', '_thresholds.json', '_training.json']

    for ext in metadata_extensions:
        source_file = model_dir / f"{base_name}{ext}"
        if source_file.exists():
            archive_file = archive_dir / f"{base_name}_{timestamp}{ext}"
            shutil.copy2(source_file, archive_file)
            archived_metadata[ext] = str(archive_file)

    # Archive MODEL_INFO.json if it exists
    model_info_file = model_dir / "MODEL_INFO.json"
    if model_info_file.exists():
        archive_info_file = archive_dir / f"MODEL_INFO_{base_name}_{timestamp}.json"
        shutil.copy2(model_info_file, archive_info_file)
        archived_metadata['model_info'] = str(archive_info_file)

    # Create archival record
    archive_record = {
        'archived_model': str(current_prod_model),
        'archive_timestamp': timestamp,
        'model_size_mb': current_prod_model.stat().st_size / (1024 * 1024),
        'model_hash': compute_model_hash(current_prod_model),
        'archived_metadata': archived_metadata,
        'status': 'archived'
    }

    # Save archive record
    record_file = archive_dir / f"archive_record_{base_name}_{timestamp}.json"
    with open(record_file, 'w') as f:
        json.dump(archive_record, f, indent=2)

    print(f"Archived production model metadata: {base_name}")
    print(f"Archive record: {record_file}")

    return archive_record


def promote_model(candidate_dir: Path, model_dir: Path, validation_results: dict) -> dict:
    """Promote validated candidate model to production."""

    candidate_model = Path(validation_results['model_path'])
    timestamp = datetime.now().strftime("%Y-%m-%d")

    # Generate production model name
    version_parts = candidate_dir.name.split('-')
    if len(version_parts) >= 3 and version_parts[0].isdigit():
        version = f"v{version_parts[0][-2:]}-{version_parts[1]}-{version_parts[2][:2]}"
    else:
        version = f"v{timestamp.replace('-', '')[:6]}"

    prod_model_name = f"disaster_rf_{version}_prod_{timestamp}.pkl"
    prod_model_path = model_dir / prod_model_name
    base_name = prod_model_path.stem

    # Copy model file
    shutil.copy2(candidate_model, prod_model_path)

    # Copy/create metadata files
    metadata_files = {}

    # Copy existing metadata from candidate
    for suffix in ['_labels.json', '_thresholds.json', '_training_log.json']:
        candidate_file = candidate_dir / f"{candidate_dir.name}{suffix}"
        if candidate_file.exists():
            prod_file = model_dir / f"{base_name}{suffix.replace('_training_log', '_training')}"
            shutil.copy2(candidate_file, prod_file)
            metadata_files[suffix] = str(prod_file)

    # Create new MODEL_INFO.json
    model_info = {
        'sha256': validation_results['model_hash'],
        'promoted_from': str(candidate_dir),
        'promotion_timestamp': datetime.now().isoformat(),
        'model_size_mb': validation_results['model_size_mb'],
        'validation_results': validation_results,
        'performance': {
            'f1_weighted': validation_results['f1_weighted'],
            'f1_micro': validation_results['f1_micro']
        },
        'version': version,
        'status': 'production'
    }

    model_info_path = model_dir / "MODEL_INFO.json"
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2)

    metadata_files['MODEL_INFO.json'] = str(model_info_path)

    promotion_record = {
        'promoted_model': str(prod_model_path),
        'source_candidate': str(candidate_dir),
        'promotion_timestamp': datetime.now().isoformat(),
        'version': version,
        'validation_results': validation_results,
        'metadata_files': metadata_files,
        'status': 'promoted'
    }

    print(f"✅ Model promoted to production: {prod_model_name}")
    print(f"📊 Performance: F1-weighted={validation_results['f1_weighted']:.4f}, F1-micro={validation_results['f1_micro']:.4f}")
    print(f"💾 Size: {validation_results['model_size_mb']:.1f}MB")

    return promotion_record


def _update_app_config_model_filename(config_path: Path, new_filename: str, backup: bool = True) -> bool:
    """Safely update app/config.py MODEL_FILENAME to new_filename."""
    try:
        text = config_path.read_text(encoding="utf-8")
        if "MODEL_FILENAME" not in text:
            print("Warning: MODEL_FILENAME not found in config; skipping auto-update")
            return False
        import re
        # Only match MODEL_FILENAME in the Config class, not TestConfig class
        # Use a more specific pattern that looks for MODEL_FILENAME within the Config class
        pattern = r"^class Config\b.*?^(\s*MODEL_FILENAME\s*=\s*)(['\"])(.+?)\2"
        repl = r"\1'" + new_filename + r"'"
        new_text, n = re.subn(pattern, repl, text, flags=re.MULTILINE | re.DOTALL)
        if n == 0:
            print("Warning: Could not update MODEL_FILENAME line; skipping auto-update")
            return False
        if backup:
            bak = config_path.with_suffix(config_path.suffix + ".bak")
            bak.write_text(text, encoding="utf-8")
        config_path.write_text(new_text, encoding="utf-8")
        return True
    except Exception as e:
        print(f"Warning: Failed to update app config: {e}")
        return False


def cleanup_old_production_models(model_dir: Path, keep_count: int = 2):
    """Remove old production model files, keeping only metadata."""

    prod_models = sorted(
        model_dir.glob("*_prod_*.pkl"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    models_to_remove = prod_models[keep_count:]

    for old_model in models_to_remove:
        size_mb = old_model.stat().st_size / (1024 * 1024)
        print(f"🗑️  Removing old production model: {old_model.name} ({size_mb:.1f}MB)")
        old_model.unlink()


def main():
    parser = argparse.ArgumentParser(description="Promote experimental model to production")
    parser.add_argument("candidate_dir", help="Path to candidate model directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate but don't promote")
    parser.add_argument("--force", action="store_true", help="Skip validation checks")
    parser.add_argument("--keep-old", type=int, default=1, help="Number of old production models to keep")
    parser.add_argument("--no-update-config", action="store_true", help="Do not update app/config.py MODEL_FILENAME")
    parser.add_argument("--print-new-path", action="store_true", help="Print promoted model filename for CI logs")

    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    candidate_dir = Path(args.candidate_dir)
    model_dir = project_root / "model"
    archive_dir = project_root / "experiments" / "model_archive"

    if not candidate_dir.exists():
        print(f"❌ Candidate directory not found: {candidate_dir}")
        return 1

    try:
        # Validate candidate model
        print(f"🔍 Validating candidate model: {candidate_dir.name}")
        validation_results = validate_candidate_model(candidate_dir)

        if not validation_results['validation_passed'] and not args.force:
            print("❌ Validation failed:")
            for error in validation_results['validation_errors']:
                print(f"  - {error}")
            return 1

        if validation_results['validation_errors'] and args.force:
            print("⚠️  Validation warnings (proceeding with --force):")
            for error in validation_results['validation_errors']:
                print(f"  - {error}")

        print("✅ Validation passed")
        print(f"📊 F1-weighted: {validation_results['f1_weighted']:.4f}")
        print(f"📊 F1-micro: {validation_results['f1_micro']:.4f}")
        print(f"💾 Model size: {validation_results['model_size_mb']:.1f}MB")

        if args.dry_run:
            print("🔍 Dry run complete - no changes made")
            return 0

        # Archive current production model
        print("\n📦 Archiving current production model...")
        archive_record = archive_current_production_model(model_dir, archive_dir)

        # Promote new model
        print("\n🚀 Promoting candidate model to production...")
        promotion_record = promote_model(candidate_dir, model_dir, validation_results)

        # Optionally update app/config.py to point at the new model filename
        prod_model_path = Path(promotion_record['promoted_model'])
        new_filename = prod_model_path.name
        if args.print_new_path:
            print(f"NEW_PRODUCTION_MODEL={new_filename}")

        if not args.no_update_config:
            app_config_path = project_root / "app" / "config.py"
            updated = _update_app_config_model_filename(app_config_path, new_filename, backup=True)
            if updated:
                print(f"🛠  Updated app/config.py MODEL_FILENAME -> {new_filename}")
            else:
                print("⚠️  Skipped updating app/config.py (see warnings above)")

        # Cleanup old models
        print(f"\n🧹 Cleaning up old production models (keeping {args.keep_old})...")
        cleanup_old_production_models(model_dir, keep_count=args.keep_old)

        # Save promotion record
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        record_file = archive_dir / f"promotion_record_{timestamp}.json"
        record_file.parent.mkdir(parents=True, exist_ok=True)

        complete_record = {
            'promotion': promotion_record,
            'archive': archive_record,
            'timestamp': timestamp
        }

        with open(record_file, 'w') as f:
            json.dump(complete_record, f, indent=2)

        print(f"\n✅ Promotion complete!")
        print(f"📝 Promotion record: {record_file}")

        return 0

    except Exception as e:
        print(f"❌ Promotion failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
