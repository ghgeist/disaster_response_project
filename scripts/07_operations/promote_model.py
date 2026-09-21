#!/usr/bin/env python3
"""
Model Promotion Script for Disaster Response System

Implements MLOps best practices for promoting experimental models to production:
- Validates candidate model against the train/cal/eval evidence contract
- Archives current production model metadata
- Promotes new model with proper versioning
- Deploys the exact threshold artifact that validation inspected
- Maintains model registry and lineage
"""

# Standard library imports
import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Third-party imports
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

# Add project src to path for package imports (script lives in scripts/07_operations/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# Local imports
from disasterproject.models.pipeline import WeightedMultiOutputClassifier
from disasterproject.utils.config import PERFORMANCE_THRESHOLDS

REQUIRED_OPTIMIZATION_SPLIT = "calibration"
REQUIRED_REPORTING_SPLIT = "frozen_eval"


def compute_model_hash(model_path: Path) -> str:
    """Compute SHA256 hash of a file for integrity verification."""
    sha256_hash = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def weighted_f1_relative_drop(baseline_weighted_f1: float, optimized_weighted_f1: float) -> float:
    """Relative weighted-F1 damage from applying thresholds.

    Defined as (baseline - optimized) / baseline. A max of 0.05 means a 5%
    relative drop, not 5 percentage points.
    """
    if baseline_weighted_f1 <= 0:
        raise ValueError(
            f"baseline_weighted_f1 must be > 0 to compute relative drop, got {baseline_weighted_f1}"
        )
    return (baseline_weighted_f1 - optimized_weighted_f1) / baseline_weighted_f1


def detect_algorithm_type(model_path: Path) -> str:
    """
    Detect the algorithm type from a model file.

    Returns:
        str: Algorithm code ('rf' for RandomForest, 'lr' for LogisticRegression, 'unknown' otherwise)
    """
    try:
        model = joblib.load(model_path)

        if hasattr(model, 'named_steps'):
            clf_step = model.named_steps.get('clf')
            if clf_step is None:
                for step_name, step_obj in model.named_steps.items():
                    if 'clf' in step_name.lower() or 'classifier' in step_name.lower():
                        clf_step = step_obj
                        break

            if clf_step is not None:
                if isinstance(clf_step, (MultiOutputClassifier, WeightedMultiOutputClassifier)):
                    estimator = clf_step.estimator
                elif hasattr(clf_step, 'estimator'):
                    estimator = clf_step.estimator
                elif hasattr(clf_step, 'estimators_') and len(clf_step.estimators_) > 0:
                    estimator = clf_step.estimators_[0]
                else:
                    estimator = clf_step

                if isinstance(estimator, RandomForestClassifier):
                    return 'rf'
                if isinstance(estimator, LogisticRegression):
                    return 'lr'

        if isinstance(model, RandomForestClassifier):
            return 'rf'
        if isinstance(model, LogisticRegression):
            return 'lr'

        return 'unknown'
    except Exception as e:
        print(f"Warning: Could not detect algorithm type: {e}")
        return 'unknown'


def discover_production_metrics_file(model_dir: Path) -> Optional[Path]:
    """
    Discover the production performance_metrics.csv file based on the current production model.

    Uses the same discovery logic as the app: finds the latest production model file,
    then looks for a matching metrics file with model-specific naming.
    """
    if not model_dir.exists():
        return None

    pattern = 'disaster_*_prod_*.pkl'
    model_files = list(model_dir.glob(pattern))

    if not model_files:
        return None

    model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest_model = model_files[0]

    base_name = latest_model.stem
    metrics_file = model_dir / f"{base_name}_performance_metrics.csv"

    if metrics_file.exists():
        return metrics_file

    legacy_metrics = model_dir / "performance_metrics.csv"
    if legacy_metrics.exists():
        return legacy_metrics

    return None


def _load_training_log(candidate_dir: Path) -> Optional[dict]:
    """Load training_log.json if present."""
    for name in ["training_log.json", f"{candidate_dir.name}_training_log.json"]:
        p = candidate_dir / name
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError, ValueError):
                return None
    return None


def _parse_baseline_micro_f1(log_data: dict) -> Optional[float]:
    """Extract explicit frozen-eval baseline micro F1 from training log.

    Does not substitute weighted F1 or samples_f1.
    """
    perf = log_data.get("performance") or {}
    raw = perf.get("micro_f1")
    if raw is None:
        raw = perf.get("f1_micro")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _discover_model_file(candidate_dir: Path) -> Optional[Path]:
    """Find the newest .pkl model file in candidate_dir, or None."""
    model_files = list(candidate_dir.glob("*.pkl"))
    if not model_files:
        return None
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return model_files[0]


def discover_candidate_thresholds(
    candidate_dir: Path, model_stem: Optional[str] = None
) -> Tuple[Optional[Path], Optional[dict], list]:
    """Discover `{model_stem}_thresholds.json` for validation and promotion.

    Prefers the non-f2 model-stem thresholds file. Returns
    (path, payload, errors). Missing/invalid evidence becomes errors, not raises.
    """
    errors: list = []
    if model_stem is None:
        model_file = _discover_model_file(candidate_dir)
        if model_file is None:
            errors.append(f"No model file (.pkl) found in {candidate_dir}")
            return None, None, errors
        model_stem = model_file.stem

    thresholds_path = candidate_dir / f"{model_stem}_thresholds.json"
    if not thresholds_path.exists():
        errors.append(
            f"Required thresholds artifact not found: {thresholds_path.name} "
            f"(expected {{model_stem}}_thresholds.json for stem '{model_stem}')"
        )
        return None, None, errors

    try:
        with open(thresholds_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"Unable to parse thresholds artifact {thresholds_path.name}: {exc}")
        return None, None, errors

    if not isinstance(payload, dict):
        errors.append(f"Thresholds artifact {thresholds_path.name} must be a JSON object")
        return None, None, errors

    return thresholds_path, payload, errors


def _validate_thresholds_provenance(payload: dict) -> list:
    """Require cal-optimize / frozen-eval-report provenance on thresholds metadata."""
    errors: list = []
    metadata = payload.get("metadata") or {}
    optimization_split = metadata.get("optimization_split")
    reporting_split = metadata.get("reporting_split")

    if optimization_split != REQUIRED_OPTIMIZATION_SPLIT:
        errors.append(
            f"Thresholds optimization_split must be '{REQUIRED_OPTIMIZATION_SPLIT}', "
            f"got {optimization_split!r}"
        )
    if reporting_split != REQUIRED_REPORTING_SPLIT:
        errors.append(
            f"Thresholds reporting_split must be '{REQUIRED_REPORTING_SPLIT}', "
            f"got {reporting_split!r}"
        )
    return errors


def _extract_threshold_operating_metrics(
    payload: dict,
) -> Tuple[Optional[float], Optional[float], Optional[float], list]:
    """Read eval critical recall and weighted F1 baseline/optimized from thresholds.

    Returns (eval_critical_recall, baseline_weighted_f1, optimized_weighted_f1, errors).
    """
    errors: list = []
    performance = payload.get("performance") or {}
    baseline = performance.get("baseline") or {}
    optimized = performance.get("optimized") or {}

    eval_critical_recall = optimized.get("critical_recall")
    if eval_critical_recall is None:
        errors.append(
            "Thresholds artifact missing performance.optimized.critical_recall "
            "(frozen-eval operating point)"
        )
        eval_critical_recall_f = None
    else:
        try:
            eval_critical_recall_f = float(eval_critical_recall)
        except (TypeError, ValueError):
            errors.append(
                f"Invalid performance.optimized.critical_recall: {eval_critical_recall!r}"
            )
            eval_critical_recall_f = None

    baseline_w = baseline.get("f1_weighted")
    optimized_w = optimized.get("f1_weighted")
    baseline_w_f: Optional[float] = None
    optimized_w_f: Optional[float] = None

    if baseline_w is None or optimized_w is None:
        errors.append(
            "Thresholds artifact missing performance.baseline.f1_weighted and/or "
            "performance.optimized.f1_weighted (required for damage guardrail)"
        )
    else:
        try:
            baseline_w_f = float(baseline_w)
            optimized_w_f = float(optimized_w)
        except (TypeError, ValueError):
            errors.append(
                f"Invalid weighted F1 values baseline={baseline_w!r} optimized={optimized_w!r}"
            )
            baseline_w_f = None
            optimized_w_f = None

    return eval_critical_recall_f, baseline_w_f, optimized_w_f, errors


def validate_candidate_model(candidate_dir: Path) -> dict:
    """Validate candidate against the train/cal/eval promotion contract.

    Missing or invalid evidence is recorded in validation_errors (fail closed)
    so --force can override without discovery exceptions escaping validation.
    """
    errors: list = []
    min_baseline_micro = PERFORMANCE_THRESHOLDS.get('min_baseline_f1_micro', 0.60)
    min_eval_critical_recall = PERFORMANCE_THRESHOLDS.get('min_eval_critical_recall', 0.55)
    max_model_size_mb = PERFORMANCE_THRESHOLDS.get('max_model_size_mb', 50)
    max_weighted_drop = PERFORMANCE_THRESHOLDS.get('max_weighted_f1_relative_drop', 0.05)

    validation_results = {
        'model_path': None,
        'model_size_mb': None,
        'model_hash': None,
        'baseline_f1_micro': None,
        'eval_critical_recall': None,
        'baseline_f1_weighted': None,
        'optimized_f1_weighted': None,
        'weighted_f1_relative_drop': None,
        'thresholds_path': None,
        'thresholds_sha256': None,
        'optimization_split': None,
        'reporting_split': None,
        # Legacy aliases kept for older promotion-record consumers
        'f1_weighted': None,
        'f1_micro': None,
        'validation_passed': False,
        'validation_errors': errors,
    }

    model_file = _discover_model_file(candidate_dir)
    if model_file is None:
        errors.append(f"No model file (.pkl) found in {candidate_dir}")
        model_stem = None
    else:
        validation_results['model_path'] = str(model_file)
        validation_results['model_size_mb'] = model_file.stat().st_size / (1024 * 1024)
        validation_results['model_hash'] = compute_model_hash(model_file)
        model_stem = model_file.stem
        if validation_results['model_size_mb'] > max_model_size_mb:
            errors.append(
                f"Model size {validation_results['model_size_mb']:.1f}MB exceeds "
                f"limit {max_model_size_mb}MB"
            )

    log_data = _load_training_log(candidate_dir)
    if log_data is None:
        errors.append(
            "training_log.json not found or unreadable "
            "(required for baseline frozen-eval micro F1)"
        )
    else:
        baseline_micro = _parse_baseline_micro_f1(log_data)
        if baseline_micro is None:
            errors.append(
                "training_log.json missing explicit performance.micro_f1 / f1_micro "
                "(no weighted/samples substitution allowed)"
            )
        else:
            validation_results['baseline_f1_micro'] = baseline_micro
            validation_results['f1_micro'] = baseline_micro
            if baseline_micro < min_baseline_micro:
                errors.append(
                    f"Baseline frozen-eval micro F1 {baseline_micro:.4f} below "
                    f"threshold {min_baseline_micro}"
                )

    thresholds_path = None
    thresholds_payload = None
    if model_stem is not None:
        thresholds_path, thresholds_payload, threshold_errors = discover_candidate_thresholds(
            candidate_dir, model_stem=model_stem
        )
        errors.extend(threshold_errors)

    if thresholds_path is not None and thresholds_payload is not None:
        validation_results['thresholds_path'] = str(thresholds_path)
        validation_results['thresholds_sha256'] = compute_model_hash(thresholds_path)
        metadata = thresholds_payload.get("metadata") or {}
        validation_results['optimization_split'] = metadata.get("optimization_split")
        validation_results['reporting_split'] = metadata.get("reporting_split")
        errors.extend(_validate_thresholds_provenance(thresholds_payload))

        eval_cr, baseline_w, optimized_w, metric_errors = _extract_threshold_operating_metrics(
            thresholds_payload
        )
        errors.extend(metric_errors)

        if eval_cr is not None:
            validation_results['eval_critical_recall'] = eval_cr
            if eval_cr < min_eval_critical_recall:
                errors.append(
                    f"Frozen-eval critical recall {eval_cr:.4f} below "
                    f"threshold {min_eval_critical_recall}"
                )

        if baseline_w is not None and optimized_w is not None:
            validation_results['baseline_f1_weighted'] = baseline_w
            validation_results['optimized_f1_weighted'] = optimized_w
            validation_results['f1_weighted'] = optimized_w
            try:
                relative_drop = weighted_f1_relative_drop(baseline_w, optimized_w)
            except ValueError as exc:
                errors.append(str(exc))
            else:
                validation_results['weighted_f1_relative_drop'] = relative_drop
                # Float-safe boundary: treat values within 1e-12 of the max as passing
                # so (1.0 - 0.95) / 1.0 satisfies <= 0.05 despite binary float noise.
                if relative_drop - max_weighted_drop > 1e-12:
                    errors.append(
                        f"Weighted F1 relative drop {relative_drop:.4f} exceeds "
                        f"max {max_weighted_drop} "
                        f"((baseline - optimized) / baseline)"
                    )

    validation_results['validation_passed'] = len(errors) == 0
    return validation_results


def archive_current_production_model(model_dir: Path, archive_dir: Path) -> dict:
    """Archive current production model metadata to model registry."""

    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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

    archived_metadata = {}
    metadata_extensions = ['.json', '_labels.json', '_thresholds.json', '_training.json']

    for ext in metadata_extensions:
        source_file = model_dir / f"{base_name}{ext}"
        if source_file.exists():
            archive_file = archive_dir / f"{base_name}_{timestamp}{ext}"
            shutil.copy2(source_file, archive_file)
            archived_metadata[ext] = str(archive_file)

    model_info_file = model_dir / "MODEL_INFO.json"
    if model_info_file.exists():
        archive_info_file = archive_dir / f"MODEL_INFO_{base_name}_{timestamp}.json"
        shutil.copy2(model_info_file, archive_info_file)
        archived_metadata['model_info'] = str(archive_info_file)

    archive_record = {
        'archived_model': str(current_prod_model),
        'archive_timestamp': timestamp,
        'model_size_mb': current_prod_model.stat().st_size / (1024 * 1024),
        'model_hash': compute_model_hash(current_prod_model),
        'archived_metadata': archived_metadata,
        'status': 'archived'
    }

    record_file = archive_dir / f"archive_record_{base_name}_{timestamp}.json"
    with open(record_file, 'w') as f:
        json.dump(archive_record, f, indent=2)

    print(f"Archived production model metadata: {base_name}")
    print(f"Archive record: {record_file}")

    return archive_record


def _resolve_training_date_and_version(candidate_dir: Path) -> Tuple[str, str]:
    """Derive training date and version code from candidate directory name."""
    version_parts = candidate_dir.name.split('-')
    if len(version_parts) >= 3 and version_parts[0].isdigit() and len(version_parts[0]) == 4:
        training_date = f"{version_parts[0]}-{version_parts[1]}-{version_parts[2]}"
        version = f"v{version_parts[0][-2:]}-{version_parts[1]}-{version_parts[2]}"
        return training_date, version

    training_log_path = candidate_dir / "training_log.json"
    if training_log_path.exists():
        try:
            with open(training_log_path, "r", encoding="utf-8") as f:
                log_data = json.load(f)
            timestamp_str = log_data.get("timestamp", "")
            if timestamp_str:
                training_date_obj = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                training_date = training_date_obj.strftime("%Y-%m-%d")
                version = f"v{training_date[2:4]}-{training_date[5:7]}-{training_date[8:10]}"
                return training_date, version
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    training_date = datetime.now().strftime("%Y-%m-%d")
    version = f"v{training_date[2:4]}-{training_date[5:7]}-{training_date[8:10]}"
    return training_date, version


def _copy_validated_thresholds(
    validation_results: dict, model_dir: Path, base_name: str, metadata_files: dict
) -> None:
    """Copy the exact thresholds artifact validation inspected into production naming."""
    thresholds_path_str = validation_results.get('thresholds_path')
    expected_sha = validation_results.get('thresholds_sha256')
    if not thresholds_path_str:
        print("⚠️  No validated thresholds_path in validation_results; skipping thresholds copy")
        return

    candidate_thresholds = Path(thresholds_path_str)
    if not candidate_thresholds.exists():
        raise FileNotFoundError(
            f"Validated thresholds artifact missing at promotion time: {candidate_thresholds}"
        )

    prod_thresholds = model_dir / f"{base_name}_thresholds.json"
    shutil.copy2(candidate_thresholds, prod_thresholds)
    copied_sha = compute_model_hash(prod_thresholds)
    if expected_sha and copied_sha != expected_sha:
        try:
            prod_thresholds.unlink()
        except OSError as cleanup_error:
            print(f"⚠️  Warning: Failed to remove mismatched thresholds file: {cleanup_error}")
        raise ValueError(
            f"Thresholds file integrity check failed!\n"
            f"  Expected hash: {expected_sha}\n"
            f"  Copied hash:   {copied_sha}\n"
            f"Deployed thresholds must match the artifact validation scored."
        )

    metadata_files['_thresholds.json'] = str(prod_thresholds)
    print(f"✅ Thresholds deployed: {prod_thresholds.name} (hash: {copied_sha[:16]}...)")


def promote_model(candidate_dir: Path, model_dir: Path, validation_results: dict) -> dict:
    """Promote validated candidate model to production."""

    candidate_model_str = validation_results.get('model_path')
    if not candidate_model_str:
        raise FileNotFoundError(
            "validation_results['model_path'] is missing; cannot promote without a model file"
        )

    candidate_model = Path(candidate_model_str)

    if not candidate_model.is_absolute():
        candidate_model = candidate_model.resolve()
        if not candidate_model.exists():
            candidate_model = (candidate_dir / Path(candidate_model_str).name).resolve()
    else:
        candidate_model = candidate_model.resolve()

    if not candidate_model.exists():
        raise FileNotFoundError(
            f"Candidate model file not found: {candidate_model}\n"
            f"  Original path: {candidate_model_str}\n"
            f"  Candidate dir: {candidate_dir}"
        )

    algorithm_code = detect_algorithm_type(candidate_model)
    if algorithm_code == 'unknown':
        print("⚠️  Warning: Could not detect algorithm type, defaulting to 'rf'")
        algorithm_code = 'rf'

    algorithm_names = {'rf': 'RandomForest', 'lr': 'LogisticRegression'}
    print(f"🔍 Detected algorithm: {algorithm_names.get(algorithm_code, algorithm_code)}")

    training_date, version = _resolve_training_date_and_version(candidate_dir)

    prod_model_name = f"disaster_{algorithm_code}_{version}_prod_{training_date}.pkl"
    prod_model_path = model_dir / prod_model_name
    base_name = prod_model_path.stem

    print(f"📋 Copying model from {candidate_model.name} to {prod_model_name}...")
    print(f"   Source: {candidate_model}")
    print(f"   Destination: {prod_model_path}")
    try:
        shutil.copy2(candidate_model, prod_model_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to copy model file:\n"
            f"  Source: {candidate_model}\n"
            f"  Source exists: {candidate_model.exists()}\n"
            f"  Destination: {prod_model_path}\n"
            f"  Destination parent exists: {prod_model_path.parent.exists()}\n"
            f"  Error: {e}"
        ) from e

    copied_hash = compute_model_hash(prod_model_path)
    expected_hash = validation_results['model_hash']
    if copied_hash != expected_hash:
        try:
            prod_model_path.unlink()
            print(f"🗑️  Removed corrupted model file: {prod_model_path}")
        except Exception as cleanup_error:
            print(f"⚠️  Warning: Failed to remove corrupted file: {cleanup_error}")
        raise ValueError(
            f"Model file integrity check failed!\n"
            f"  Expected hash: {expected_hash}\n"
            f"  Copied hash:   {copied_hash}\n"
            f"The copied model file does not match the validated candidate."
        )
    print(f"✅ Model file integrity verified (hash: {copied_hash[:16]}...)")

    metadata_files = {}

    # Copy optional label metadata if present under either naming convention
    for candidate_labels in [
        candidate_dir / f"{candidate_dir.name}_labels.json",
        candidate_dir / "label_order.json",
        candidate_dir / f"{candidate_model.stem}_labels.json",
    ]:
        if candidate_labels.exists():
            prod_labels = model_dir / f"{base_name}_labels.json"
            shutil.copy2(candidate_labels, prod_labels)
            metadata_files['_labels.json'] = str(prod_labels)
            break

    _copy_validated_thresholds(validation_results, model_dir, base_name, metadata_files)

    training_log = candidate_dir / "training_log.json"
    if training_log.exists():
        prod_training = model_dir / f"{base_name}_training.json"
        shutil.copy2(training_log, prod_training)
        metadata_files['_training.json'] = str(prod_training)

    metrics_csv = candidate_dir / "performance_metrics.csv"
    if metrics_csv.exists():
        prod_metrics_csv = model_dir / f"{base_name}_performance_metrics.csv"
        shutil.copy2(metrics_csv, prod_metrics_csv)
        metadata_files['performance_metrics.csv'] = str(prod_metrics_csv)
        print(f"📊 Copied performance metrics: {prod_metrics_csv.name}")

    model_info = {
        'sha256': validation_results['model_hash'],
        'promoted_from': str(candidate_dir),
        'promotion_timestamp': datetime.now().isoformat(),
        'model_size_mb': validation_results['model_size_mb'],
        'algorithm': algorithm_code,
        'algorithm_name': algorithm_names.get(algorithm_code, algorithm_code),
        'validation_results': validation_results,
        'performance': {
            'baseline_f1_micro': validation_results.get('baseline_f1_micro'),
            'eval_critical_recall': validation_results.get('eval_critical_recall'),
            'baseline_f1_weighted': validation_results.get('baseline_f1_weighted'),
            'optimized_f1_weighted': validation_results.get('optimized_f1_weighted'),
            'weighted_f1_relative_drop': validation_results.get('weighted_f1_relative_drop'),
            'f1_weighted': validation_results.get('f1_weighted'),
            'f1_micro': validation_results.get('f1_micro'),
        },
        'thresholds_sha256': validation_results.get('thresholds_sha256'),
        'optimization_split': validation_results.get('optimization_split'),
        'reporting_split': validation_results.get('reporting_split'),
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
    print(f"📊 Algorithm: {algorithm_names.get(algorithm_code, algorithm_code)}")
    baseline_micro = validation_results.get('baseline_f1_micro')
    eval_cr = validation_results.get('eval_critical_recall')
    drop = validation_results.get('weighted_f1_relative_drop')
    if baseline_micro is not None:
        print(f"📊 Baseline micro F1: {baseline_micro:.4f}")
    if eval_cr is not None:
        print(f"📊 Eval critical recall: {eval_cr:.4f}")
    if drop is not None:
        print(f"📊 Weighted F1 relative drop: {drop:.4f}")
    if validation_results.get('model_size_mb') is not None:
        print(f"💾 Size: {validation_results['model_size_mb']:.1f}MB")

    return promotion_record


def _update_app_config_model_filename(config_path: Path, new_filename: str, backup: bool = True) -> bool:
    """Safely update app/config.py MODEL_FILENAME to new_filename."""
    try:
        text = config_path.read_text(encoding="utf-8")
        if "MODEL_FILENAME" not in text:
            print("Warning: MODEL_FILENAME not found in config; skipping auto-update")
            return False
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


def _format_optional_float(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description="Promote experimental model to production")
    parser.add_argument("candidate_dir", help="Path to candidate model directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate but don't promote")
    parser.add_argument("--force", action="store_true", help="Skip validation checks")
    parser.add_argument("--keep-old", type=int, default=1, help="Number of old production models to keep")
    parser.add_argument("--no-update-config", action="store_true", help="Do not update app/config.py MODEL_FILENAME")
    parser.add_argument("--print-new-path", action="store_true", help="Print promoted model filename for CI logs")

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    candidate_dir = Path(args.candidate_dir)
    if not candidate_dir.is_absolute():
        candidate_dir = (project_root / candidate_dir).resolve()
    else:
        candidate_dir = candidate_dir.resolve()

    model_dir = (project_root / "model").resolve()
    archive_dir = (project_root / "experiments" / "model_archive").resolve()

    if not candidate_dir.exists():
        print(f"❌ Candidate directory not found: {candidate_dir}")
        return 1

    try:
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

        print("✅ Validation passed" if validation_results['validation_passed'] else "✅ Proceeding with --force")
        print(
            f"📊 Baseline micro F1: "
            f"{_format_optional_float(validation_results.get('baseline_f1_micro'))}"
        )
        print(
            f"📊 Eval critical recall: "
            f"{_format_optional_float(validation_results.get('eval_critical_recall'))}"
        )
        print(
            f"📊 Weighted F1 relative drop: "
            f"{_format_optional_float(validation_results.get('weighted_f1_relative_drop'))}"
        )
        size_mb = validation_results.get('model_size_mb')
        print(f"💾 Model size: {_format_optional_float(size_mb, digits=1)}MB")

        if args.dry_run:
            print("🔍 Dry run complete - no changes made")
            return 0

        print("\n📦 Archiving current production model...")
        archive_record = archive_current_production_model(model_dir, archive_dir)

        print("\n🚀 Promoting candidate model to production...")
        promotion_record = promote_model(candidate_dir, model_dir, validation_results)

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

        print(f"\n🧹 Cleaning up old production models (keeping {args.keep_old})...")
        cleanup_old_production_models(model_dir, keep_count=args.keep_old)

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

        print("\n✅ Promotion complete!")
        print(f"📝 Promotion record: {record_file}")

        return 0

    except Exception as e:
        print(f"❌ Promotion failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
