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
import math
import os
import re
import shutil
import sys
import uuid
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
from disasterproject.utils.config import PERFORMANCE_THRESHOLDS, TARGET_COLUMNS

REQUIRED_OPTIMIZATION_SPLIT = "calibration"
REQUIRED_REPORTING_SPLIT = "frozen_eval"
SUPPORTED_ALGORITHMS = frozenset({"rf", "lr"})
# Agreement tolerance for duplicated metrics across training_log vs thresholds artifacts
METRIC_CONSISTENCY_ABS_TOL = 1e-6
FORCE_REQUIRED_FIELDS = (
    "model_path",
    "model_hash",
    "thresholds_path",
    "thresholds_sha256",
    "labels_path",
    "labels_sha256",
    "algorithm",
)


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


def _discover_model_file(candidate_dir: Path) -> Tuple[Optional[Path], list]:
    """Find exactly one .pkl model file in candidate_dir.

    Returns (path, errors). Multiple or zero models fail closed — no mtime heuristic.
    """
    model_files = list(candidate_dir.glob("*.pkl"))
    if not model_files:
        return None, [f"No model file (.pkl) found in {candidate_dir}"]
    if len(model_files) > 1:
        names = sorted(path.name for path in model_files)
        return None, [
            f"Multiple model files (.pkl) found in {candidate_dir}: {names}. "
            "Promotion requires exactly one candidate model."
        ]
    return model_files[0], []


def discover_candidate_thresholds(
    candidate_dir: Path, model_stem: Optional[str] = None
) -> Tuple[Optional[Path], Optional[dict], list]:
    """Discover `{model_stem}_thresholds.json` for validation and promotion.

    Prefers the non-f2 model-stem thresholds file. Returns
    (path, payload, errors). Missing/invalid evidence becomes errors, not raises.
    """
    errors: list = []
    if model_stem is None:
        model_file, model_errors = _discover_model_file(candidate_dir)
        if model_errors:
            return None, None, model_errors
        model_stem = model_file.stem

    thresholds_path = (candidate_dir / f"{model_stem}_thresholds.json").resolve()
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


def discover_candidate_labels(
    candidate_dir: Path, model_stem: Optional[str] = None
) -> Tuple[Optional[Path], Optional[list], list]:
    """Discover required `{model_stem}_labels.json` for validation and promotion.

    Returns (path, payload, errors). Legacy ``label_order.json`` is not accepted.
    """
    errors: list = []
    if model_stem is None:
        model_file, model_errors = _discover_model_file(candidate_dir)
        if model_errors:
            return None, None, model_errors
        model_stem = model_file.stem

    labels_path = (candidate_dir / f"{model_stem}_labels.json").resolve()
    if not labels_path.exists():
        errors.append(
            f"Required labels artifact not found: {labels_path.name} "
            f"(expected {{model_stem}}_labels.json for stem '{model_stem}')"
        )
        return None, None, errors

    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"Unable to parse labels artifact {labels_path.name}: {exc}")
        return None, None, errors

    if not isinstance(payload, list):
        errors.append(f"Labels artifact {labels_path.name} must be a JSON array")
        return None, None, errors

    return labels_path, payload, errors


def _validate_labels_order(payload: list) -> list:
    """Require exact TARGET_COLUMNS order and coverage."""
    errors: list = []
    labels = [str(item) for item in payload]
    expected = list(TARGET_COLUMNS)
    if labels == expected:
        return errors

    if set(labels) != set(expected):
        missing = [label for label in expected if label not in labels]
        extra = [label for label in labels if label not in expected]
        errors.append(
            "Labels artifact coverage mismatch with TARGET_COLUMNS: "
            f"missing={missing[:8]!r} extra={extra[:8]!r}"
        )
    else:
        errors.append(
            "Labels artifact order mismatch with TARGET_COLUMNS "
            f"(expected {len(expected)} labels in contract order)"
        )
    return errors


def assert_force_promotion_prerequisites(validation_results: dict) -> None:
    """Require structural deploy artifacts; --force cannot bypass these.

    --force may override metric/provenance gate failures only. Promoting still
    requires a loadable supported model plus the thresholds and labels artifacts
    validation would deploy.
    """
    missing = [key for key in FORCE_REQUIRED_FIELDS if not validation_results.get(key)]
    algorithm = validation_results.get("algorithm")
    if algorithm not in SUPPORTED_ALGORITHMS:
        missing.append("algorithm(rf|lr)")
    # De-dupe while preserving order
    deduped = list(dict.fromkeys(missing))
    if deduped:
        raise ValueError(
            "--force cannot override missing structural promotion prerequisites: "
            f"{deduped}. A loadable supported model, thresholds artifact, and "
            "labels artifact (with hashes) are required so the deployed "
            "operating point matches validated evidence."
        )


def _metrics_agree(left: float, right: float, abs_tol: float = METRIC_CONSISTENCY_ABS_TOL) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=abs_tol)


def _validate_thresholds_map(payload: dict) -> list:
    """Require a complete deployable per-label threshold map.

    Every TARGET_COLUMNS label must be present with a finite numeric value in [0, 1].
    Missing labels would silently fall back to 0.5 at inference and break the
    validated operating-point invariant.
    """
    errors: list = []
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict):
        errors.append(
            "Thresholds artifact missing 'thresholds' object "
            "(required deployable per-label operating point)"
        )
        return errors

    missing = [label for label in TARGET_COLUMNS if label not in thresholds]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        errors.append(
            f"Thresholds map missing {len(missing)} TARGET_COLUMNS label(s): "
            f"{preview}{suffix}"
        )

    invalid: list = []
    for label in TARGET_COLUMNS:
        if label not in thresholds:
            continue
        raw = thresholds[label]
        try:
            value = float(raw)
        except (TypeError, ValueError):
            invalid.append(f"{label}={raw!r} (not numeric)")
            continue
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            invalid.append(f"{label}={raw!r} (need finite value in [0, 1])")

    if invalid:
        preview = ", ".join(invalid[:8])
        suffix = f" (+{len(invalid) - 8} more)" if len(invalid) > 8 else ""
        errors.append(
            f"Thresholds map has {len(invalid)} invalid value(s): {preview}{suffix}"
        )

    return errors

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
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], list]:
    """Read eval critical recall, baseline micro, and weighted F1 values from thresholds.

    Returns (
        eval_critical_recall,
        baseline_f1_micro,
        baseline_weighted_f1,
        optimized_weighted_f1,
        errors,
    ).
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

    baseline_micro_raw = baseline.get("f1_micro")
    if baseline_micro_raw is None:
        baseline_micro_raw = baseline.get("micro_f1")
    baseline_micro_f: Optional[float] = None
    if baseline_micro_raw is None:
        errors.append(
            "Thresholds artifact missing performance.baseline.f1_micro "
            "(required for consistency with training_log)"
        )
    else:
        try:
            baseline_micro_f = float(baseline_micro_raw)
        except (TypeError, ValueError):
            errors.append(
                f"Invalid performance.baseline.f1_micro: {baseline_micro_raw!r}"
            )

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

    return (
        eval_critical_recall_f,
        baseline_micro_f,
        baseline_w_f,
        optimized_w_f,
        errors,
    )


def validate_candidate_model(candidate_dir: Path) -> dict:
    """Validate candidate against the train/cal/eval promotion contract.

    Missing or invalid evidence is recorded in validation_errors (fail closed)
    so --force can override metric/provenance failures without discovery exceptions.
    Structural deploy fields (model + thresholds paths/hashes) must still be present
    for promotion; see assert_force_promotion_prerequisites.
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
        'algorithm': None,
        'baseline_f1_micro': None,
        'eval_critical_recall': None,
        'baseline_f1_weighted': None,
        'optimized_f1_weighted': None,
        'weighted_f1_relative_drop': None,
        'thresholds_path': None,
        'thresholds_sha256': None,
        'labels_path': None,
        'labels_sha256': None,
        'optimization_split': None,
        'reporting_split': None,
        # Legacy aliases kept for older promotion-record consumers
        'f1_weighted': None,
        'f1_micro': None,
        'validation_passed': False,
        'validation_errors': errors,
    }

    model_file, model_errors = _discover_model_file(candidate_dir)
    errors.extend(model_errors)
    model_stem = None
    if model_file is not None:
        model_file = model_file.resolve()
        validation_results['model_path'] = str(model_file)
        validation_results['model_size_mb'] = model_file.stat().st_size / (1024 * 1024)
        validation_results['model_hash'] = compute_model_hash(model_file)
        model_stem = model_file.stem
        algorithm = detect_algorithm_type(model_file)
        validation_results['algorithm'] = algorithm
        if algorithm not in SUPPORTED_ALGORITHMS:
            errors.append(
                f"Unsupported or unloadable model (algorithm={algorithm!r}); "
                "promotion requires a loadable LogisticRegression or RandomForest pipeline"
            )
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
        labels_path, labels_payload, label_errors = discover_candidate_labels(
            candidate_dir, model_stem=model_stem
        )
        errors.extend(label_errors)
        if labels_path is not None and labels_payload is not None:
            labels_path = labels_path.resolve()
            validation_results['labels_path'] = str(labels_path)
            validation_results['labels_sha256'] = compute_model_hash(labels_path)
            errors.extend(_validate_labels_order(labels_payload))

    if thresholds_path is not None and thresholds_payload is not None:
        thresholds_path = thresholds_path.resolve()
        validation_results['thresholds_path'] = str(thresholds_path)
        validation_results['thresholds_sha256'] = compute_model_hash(thresholds_path)
        metadata = thresholds_payload.get("metadata") or {}
        validation_results['optimization_split'] = metadata.get("optimization_split")
        validation_results['reporting_split'] = metadata.get("reporting_split")
        errors.extend(_validate_thresholds_provenance(thresholds_payload))
        errors.extend(_validate_thresholds_map(thresholds_payload))

        (
            eval_cr,
            thresholds_baseline_micro,
            baseline_w,
            optimized_w,
            metric_errors,
        ) = _extract_threshold_operating_metrics(thresholds_payload)
        errors.extend(metric_errors)

        if eval_cr is not None:
            validation_results['eval_critical_recall'] = eval_cr
            if eval_cr < min_eval_critical_recall:
                errors.append(
                    f"Frozen-eval critical recall {eval_cr:.4f} below "
                    f"threshold {min_eval_critical_recall}"
                )

            metadata_eval_cr = metadata.get("eval_critical_recall")
            if metadata_eval_cr is None:
                errors.append(
                    "Thresholds metadata missing eval_critical_recall "
                    "(must match performance.optimized.critical_recall)"
                )
            else:
                try:
                    metadata_eval_cr_f = float(metadata_eval_cr)
                except (TypeError, ValueError):
                    errors.append(
                        f"Invalid metadata.eval_critical_recall: {metadata_eval_cr!r}"
                    )
                else:
                    if not _metrics_agree(metadata_eval_cr_f, eval_cr):
                        errors.append(
                            "Inconsistent critical recall across thresholds artifact: "
                            f"metadata.eval_critical_recall={metadata_eval_cr_f:.6f} vs "
                            f"performance.optimized.critical_recall={eval_cr:.6f}"
                        )

        log_micro = validation_results.get('baseline_f1_micro')
        if log_micro is not None and thresholds_baseline_micro is not None:
            if not _metrics_agree(log_micro, thresholds_baseline_micro):
                errors.append(
                    "Inconsistent baseline micro F1 across artifacts: "
                    f"training_log={log_micro:.6f} vs "
                    f"thresholds.performance.baseline.f1_micro={thresholds_baseline_micro:.6f}"
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
    """Archive current production *metadata* (not the .pkl binary).

    Production ``*_prod_*.pkl`` binaries stay under ``model/`` until
    ``cleanup_old_production_models`` removes extras per ``--keep-old``.
    ``experiments/model_archive/`` stores companion metadata and a record with
    the prior binary's SHA256 so rollback can restore from Git history
    (tracked ``model/*_prod_*.pkl``) and verify the hash.
    """

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
    model_sha256 = compute_model_hash(current_prod_model)

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
        'prior_production_model': current_prod_model.name,
        'prior_production_path_at_archive': str(current_prod_model),
        # Legacy key retained for older readers; path may be deleted after cleanup.
        'archived_model': str(current_prod_model),
        'archive_timestamp': timestamp,
        'model_size_mb': current_prod_model.stat().st_size / (1024 * 1024),
        'model_hash': model_sha256,
        'model_sha256': model_sha256,
        'binary_archived': False,
        'binary_retention': 'not_copied',
        'rollback': {
            'method': 'git_history',
            'artifact_path': f'model/{current_prod_model.name}',
            'verify_sha256': model_sha256,
            'note': (
                'The production .pkl binary is not copied into '
                'experiments/model_archive/. After cleanup_old_production_models '
                'removes superseded binaries from model/, restore from Git '
                '(tracked model/*_prod_*.pkl) and verify model_sha256.'
            ),
        },
        'archived_metadata': archived_metadata,
        'status': 'metadata_archived',
    }

    record_file = archive_dir / f"archive_record_{base_name}_{timestamp}.json"
    archive_record['archive_record_path'] = str(record_file)
    with open(record_file, 'w') as f:
        json.dump(archive_record, f, indent=2)

    print(f"Archived production model metadata: {base_name}")
    print(
        "Note: .pkl binary was not copied; rollback is via Git history "
        f"(sha256={model_sha256[:16]}...)"
    )
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


def _cleanup_path(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError as cleanup_error:
        print(f"⚠️  Warning: Failed to remove {path}: {cleanup_error}")


def _deploy_production_labels(
    *,
    candidate_labels: Path,
    expected_labels_hash: str,
    prod_labels_path: Path,
) -> None:
    """Copy/verify the required stem-bound labels artifact for production.

    Existing matching labels are treated as an idempotent retry. Missing or
    colliding content fails closed.
    """
    if not candidate_labels.exists():
        raise FileNotFoundError(
            f"Validated labels artifact missing at promotion time: {candidate_labels}"
        )

    if prod_labels_path.exists():
        existing_hash = compute_model_hash(prod_labels_path)
        if existing_hash == expected_labels_hash:
            print(
                f"✅ Idempotent retry: existing production labels already match "
                f"{prod_labels_path.name}"
            )
            return
        raise ValueError(
            "Production labels collision: destination exists with different content. "
            f"existing={existing_hash[:16]}... expected={expected_labels_hash[:16]}..."
        )

    staging = prod_labels_path.with_suffix(prod_labels_path.suffix + ".staging")
    try:
        shutil.copy2(candidate_labels, staging)
        staged_hash = compute_model_hash(staging)
        if staged_hash != expected_labels_hash:
            raise ValueError(
                f"Staged labels integrity check failed!\n"
                f"  Expected hash: {expected_labels_hash}\n"
                f"  Staged hash:   {staged_hash}\n"
                f"Deployed labels must match the artifact validation scored."
            )
        os.replace(staging, prod_labels_path)
        final_hash = compute_model_hash(prod_labels_path)
        if final_hash != expected_labels_hash:
            raise ValueError(
                f"Final labels integrity check failed!\n"
                f"  Expected hash: {expected_labels_hash}\n"
                f"  Final hash:    {final_hash}"
            )
        print(
            f"✅ Labels deployed: {prod_labels_path.name} "
            f"(hash: {expected_labels_hash[:16]}...)"
        )
    except Exception:
        _cleanup_path(staging)
        if prod_labels_path.exists() and compute_model_hash(prod_labels_path) != expected_labels_hash:
            _cleanup_path(prod_labels_path)
        raise


def _resolve_production_destination_action(
    *,
    prod_model_path: Path,
    prod_thresholds_path: Path,
    expected_model_hash: str,
    expected_thresholds_hash: str,
) -> str:
    """Decide how to treat existing production artifact names.

    Production filenames are immutable once created:
    - create: neither destination exists
    - idempotent: both exist with matching SHA-256 hashes
    - otherwise fail closed (incomplete pair or content collision)

    Returns:
        "create" or "idempotent"
    """
    model_exists = prod_model_path.exists()
    thresholds_exist = prod_thresholds_path.exists()

    if not model_exists and not thresholds_exist:
        return "create"

    if model_exists ^ thresholds_exist:
        present = prod_model_path.name if model_exists else prod_thresholds_path.name
        missing = prod_thresholds_path.name if model_exists else prod_model_path.name
        raise ValueError(
            "Incomplete production artifact pair for immutable destination names:\n"
            f"  Present: {present}\n"
            f"  Missing: {missing}\n"
            "Refusing to overwrite or complete an incomplete production pair in place."
        )

    existing_model_hash = compute_model_hash(prod_model_path)
    existing_thresholds_hash = compute_model_hash(prod_thresholds_path)
    model_matches = existing_model_hash == expected_model_hash
    thresholds_match = existing_thresholds_hash == expected_thresholds_hash

    if model_matches and thresholds_match:
        return "idempotent"

    details = []
    if not model_matches:
        details.append(
            f"model {prod_model_path.name}: existing={existing_model_hash[:16]}... "
            f"expected={expected_model_hash[:16]}..."
        )
    if not thresholds_match:
        details.append(
            f"thresholds {prod_thresholds_path.name}: "
            f"existing={existing_thresholds_hash[:16]}... "
            f"expected={expected_thresholds_hash[:16]}..."
        )
    raise ValueError(
        "Production artifact collision: destination names already exist with "
        "different content. Production filenames are immutable once created.\n"
        + "\n".join(f"  - {item}" for item in details)
    )


def _atomic_deploy_model_and_thresholds(
    *,
    candidate_model: Path,
    expected_model_hash: str,
    candidate_thresholds: Path,
    expected_thresholds_hash: str,
    model_dir: Path,
    prod_model_path: Path,
    prod_thresholds_path: Path,
) -> None:
    """Stage model + thresholds, verify hashes, then finalize with model last.

    Existing production destinations are never overwritten. Matching pairs are
    treated as an idempotent retry; collisions and incomplete pairs fail closed.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    action = _resolve_production_destination_action(
        prod_model_path=prod_model_path,
        prod_thresholds_path=prod_thresholds_path,
        expected_model_hash=expected_model_hash,
        expected_thresholds_hash=expected_thresholds_hash,
    )
    if action == "idempotent":
        print(
            f"✅ Idempotent retry: existing production artifacts already match "
            f"{prod_model_path.name} and {prod_thresholds_path.name}"
        )
        return

    staging_dir = model_dir / f".promotion_staging_{os.getpid()}_{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=True, exist_ok=False)
    staged_model = staging_dir / "model.pkl.staging"
    staged_thresholds = staging_dir / "thresholds.json.staging"
    # Only clean up destinations we create in this call (new pair), never
    # pre-existing immutable production artifacts.
    thresholds_finalized = False
    model_finalized = False

    try:
        print(f"📋 Staging model from {candidate_model.name}...")
        try:
            shutil.copy2(candidate_model, staged_model)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to stage model file:\n"
                f"  Source: {candidate_model}\n"
                f"  Destination: {staged_model}\n"
                f"  Error: {exc}"
            ) from exc

        staged_model_hash = compute_model_hash(staged_model)
        if staged_model_hash != expected_model_hash:
            raise ValueError(
                f"Staged model integrity check failed!\n"
                f"  Expected hash: {expected_model_hash}\n"
                f"  Staged hash:   {staged_model_hash}\n"
                f"The staged model file does not match the validated candidate."
            )

        if not candidate_thresholds.exists():
            raise FileNotFoundError(
                f"Validated thresholds artifact missing at promotion time: "
                f"{candidate_thresholds}"
            )

        print(f"📋 Staging thresholds from {candidate_thresholds.name}...")
        try:
            shutil.copy2(candidate_thresholds, staged_thresholds)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to stage thresholds file:\n"
                f"  Source: {candidate_thresholds}\n"
                f"  Destination: {staged_thresholds}\n"
                f"  Error: {exc}"
            ) from exc

        staged_thresholds_hash = compute_model_hash(staged_thresholds)
        if staged_thresholds_hash != expected_thresholds_hash:
            raise ValueError(
                f"Staged thresholds integrity check failed!\n"
                f"  Expected hash: {expected_thresholds_hash}\n"
                f"  Staged hash:   {staged_thresholds_hash}\n"
                f"Deployed thresholds must match the artifact validation scored."
            )

        # Re-check immutability immediately before finalize in case another
        # process created the destinations while we were staging.
        action = _resolve_production_destination_action(
            prod_model_path=prod_model_path,
            prod_thresholds_path=prod_thresholds_path,
            expected_model_hash=expected_model_hash,
            expected_thresholds_hash=expected_thresholds_hash,
        )
        if action == "idempotent":
            print(
                f"✅ Idempotent retry after staging: existing production artifacts "
                f"already match {prod_model_path.name}"
            )
            return

        # Finalize thresholds first; production model last so discovery never sees
        # a new prod model without its validated operating-point file.
        # Destinations are guaranteed absent here (create path only).
        os.replace(staged_thresholds, prod_thresholds_path)
        thresholds_finalized = True
        if compute_model_hash(prod_thresholds_path) != expected_thresholds_hash:
            raise ValueError(
                f"Final thresholds integrity check failed!\n"
                f"  Expected hash: {expected_thresholds_hash}\n"
                f"  Final hash:    {compute_model_hash(prod_thresholds_path)}"
            )
        print(
            f"✅ Thresholds deployed: {prod_thresholds_path.name} "
            f"(hash: {expected_thresholds_hash[:16]}...)"
        )

        os.replace(staged_model, prod_model_path)
        model_finalized = True
        if compute_model_hash(prod_model_path) != expected_model_hash:
            raise ValueError(
                f"Final model integrity check failed!\n"
                f"  Expected hash: {expected_model_hash}\n"
                f"  Final hash:    {compute_model_hash(prod_model_path)}"
            )
        print(
            f"✅ Model deployed: {prod_model_path.name} "
            f"(hash: {expected_model_hash[:16]}...)"
        )
    except Exception:
        # Roll back only artifacts created in this create attempt.
        if model_finalized:
            _cleanup_path(prod_model_path)
        if thresholds_finalized:
            _cleanup_path(prod_thresholds_path)
        raise
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def promote_model(candidate_dir: Path, model_dir: Path, validation_results: dict) -> dict:
    """Promote validated candidate model to production atomically."""

    assert_force_promotion_prerequisites(validation_results)

    candidate_model_str = validation_results['model_path']
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

    algorithm_code = validation_results.get('algorithm') or detect_algorithm_type(candidate_model)
    if algorithm_code not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Cannot promote unsupported or unloadable model "
            f"(algorithm={algorithm_code!r}); expected one of {sorted(SUPPORTED_ALGORITHMS)}"
        )

    # Re-verify loadability at promote time (non-bypassable structural check)
    live_algorithm = detect_algorithm_type(candidate_model)
    if live_algorithm != algorithm_code:
        raise ValueError(
            f"Model algorithm changed between validation and promotion: "
            f"validated={algorithm_code!r}, live={live_algorithm!r}"
        )

    algorithm_names = {'rf': 'RandomForest', 'lr': 'LogisticRegression'}
    print(f"🔍 Detected algorithm: {algorithm_names.get(algorithm_code, algorithm_code)}")

    training_date, version = _resolve_training_date_and_version(candidate_dir)

    prod_model_name = f"disaster_{algorithm_code}_{version}_prod_{training_date}.pkl"
    prod_model_path = model_dir / prod_model_name
    base_name = prod_model_path.stem
    prod_thresholds_path = model_dir / f"{base_name}_thresholds.json"
    prod_labels_path = model_dir / f"{base_name}_labels.json"
    candidate_thresholds = Path(validation_results['thresholds_path'])
    candidate_labels = Path(validation_results['labels_path'])

    _atomic_deploy_model_and_thresholds(
        candidate_model=candidate_model,
        expected_model_hash=validation_results['model_hash'],
        candidate_thresholds=candidate_thresholds,
        expected_thresholds_hash=validation_results['thresholds_sha256'],
        model_dir=model_dir,
        prod_model_path=prod_model_path,
        prod_thresholds_path=prod_thresholds_path,
    )

    _deploy_production_labels(
        candidate_labels=candidate_labels,
        expected_labels_hash=validation_results['labels_sha256'],
        prod_labels_path=prod_labels_path,
    )

    metadata_files = {
        '_thresholds.json': str(prod_thresholds_path),
        '_labels.json': str(prod_labels_path),
    }

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
        'labels_sha256': validation_results.get('labels_sha256'),
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
    """Safely update a single-line production MODEL_FILENAME string assignment.

    Only rewrites lines like ``MODEL_FILENAME = 'disaster_*_prod_*.pkl'``.
    Skips when config uses env override / auto-discovery (no disaster_* literal),
    and never uses DOTALL matching that could erase the Config class body.
    """
    try:
        text = config_path.read_text(encoding="utf-8")
        if "MODEL_FILENAME" not in text:
            print("Warning: MODEL_FILENAME not found in config; skipping auto-update")
            return False
        # Single-line only: require a disaster_* production literal on the same line.
        pattern = (
            r"^(\s*MODEL_FILENAME\s*=\s*)(['\"])"
            r"(disaster_[^'\"]+_prod_[^'\"]+\.pkl)\2"
            r"(\s*(?:#.*)?)?$"
        )
        repl = r"\1'" + new_filename + r"'\4"
        new_text, n = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
        if n == 0:
            print(
                "Warning: No disaster_* production MODEL_FILENAME literal in "
                "app/config.py (likely auto-discovery); skipping auto-update"
            )
            return False
        if backup:
            bak = config_path.with_suffix(config_path.suffix + ".bak")
            bak.write_text(text, encoding="utf-8")
        config_path.write_text(new_text, encoding="utf-8")
        return True
    except Exception as e:
        print(f"Warning: Failed to update app config: {e}")
        return False


def cleanup_old_production_models(model_dir: Path, keep_count: int = 2) -> list[str]:
    """Remove old production model .pkl files, keeping companion metadata.

    Returns basenames of removed binaries. Does not delete thresholds/metrics JSON.
    """

    prod_models = sorted(
        model_dir.glob("*_prod_*.pkl"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    models_to_remove = prod_models[keep_count:]
    removed: list[str] = []

    for old_model in models_to_remove:
        size_mb = old_model.stat().st_size / (1024 * 1024)
        print(
            f"🗑️  Removing old production model binary: {old_model.name} "
            f"({size_mb:.1f}MB); metadata retained; restore via Git + archive sha256"
        )
        old_model.unlink()
        removed.append(old_model.name)

    return removed


def _format_optional_float(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description="Promote experimental model to production")
    parser.add_argument("candidate_dir", help="Path to candidate model directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate but don't promote")
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Override metric/provenance gate failures only; "
            "still requires model + thresholds + labels artifacts with hashes"
        ),
    )
    parser.add_argument("--keep-old", type=int, default=1, help="Number of old production models to keep")
    parser.add_argument(
        "--update-config",
        action="store_true",
        help=(
            "Rewrite a disaster_* MODEL_FILENAME string literal in app/config.py. "
            "Off by default: Config auto-discovers the newest model/*_prod_*.pkl "
            "(or MODEL_FILENAME env override)."
        ),
    )
    parser.add_argument(
        "--no-update-config",
        action="store_true",
        help=argparse.SUPPRESS,  # deprecated; skipping config update is now the default
    )
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
            # Structural deploy prerequisites are never bypassable by --force.
            # Check before dry-run success or archiving production.
            try:
                assert_force_promotion_prerequisites(validation_results)
            except ValueError as exc:
                print(f"❌ {exc}")
                return 1

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

        if args.force:
            # Re-check immediately before archive so a force path cannot mutate prod
            # without a deployable operating-point artifact.
            assert_force_promotion_prerequisites(validation_results)

        print("\n📦 Archiving current production model...")
        archive_record = archive_current_production_model(model_dir, archive_dir)

        print("\n🚀 Promoting candidate model to production...")
        promotion_record = promote_model(candidate_dir, model_dir, validation_results)

        prod_model_path = Path(promotion_record['promoted_model'])
        new_filename = prod_model_path.name
        if args.print_new_path:
            print(f"NEW_PRODUCTION_MODEL={new_filename}")

        if args.update_config:
            app_config_path = project_root / "app" / "config.py"
            updated = _update_app_config_model_filename(app_config_path, new_filename, backup=True)
            if updated:
                print(f"🛠  Updated app/config.py MODEL_FILENAME -> {new_filename}")
            else:
                print("⚠️  Skipped updating app/config.py (see warnings above)")
        elif args.no_update_config:
            print("ℹ️  --no-update-config is the default; app/config.py left unchanged")
        else:
            print(
                "ℹ️  Left app/config.py unchanged "
                "(auto-discovers model/*_prod_*.pkl; pass --update-config to rewrite a literal)"
            )

        print(f"\n🧹 Cleaning up old production models (keeping {args.keep_old})...")
        removed_binaries = cleanup_old_production_models(model_dir, keep_count=args.keep_old)
        if archive_record:
            prior_name = archive_record.get('prior_production_model') or Path(
                archive_record.get('archived_model', '')
            ).name
            archive_record = {
                **archive_record,
                'binary_removed_from_model_dir': prior_name in removed_binaries,
                'binaries_removed_by_cleanup': removed_binaries,
            }
            record_path = archive_record.get('archive_record_path')
            if record_path:
                try:
                    with open(record_path, 'w', encoding='utf-8') as f:
                        json.dump(archive_record, f, indent=2)
                except OSError as exc:
                    print(f"⚠️  Warning: Failed to refresh archive record: {exc}")

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
