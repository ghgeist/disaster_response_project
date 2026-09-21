"""Tests for model promotion evaluation-contract gates.

These tests ensure:
1. Algorithm type is correctly detected from model files
2. Correct filename is generated based on algorithm type
3. Model and thresholds file integrity is verified after copying
4. MODEL_INFO.json includes algorithm metadata
5. Promotion enforces baseline micro F1, eval critical recall, size,
   weighted-F1 relative-drop guardrail, and cal/frozen_eval provenance
6. Deployed thresholds SHA matches the validated candidate artifact
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_PATH = PROJECT_ROOT / 'scripts' / '07_operations'
sys.path.insert(0, str(SCRIPTS_PATH))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# pylint: disable=import-error
from promote_model import (  # noqa: E402
    assert_force_promotion_prerequisites,
    compute_model_hash,
    detect_algorithm_type,
    discover_production_metrics_file,
    promote_model,
    validate_candidate_model,
    weighted_f1_relative_drop,
    _update_app_config_model_filename,
)

from disasterproject.utils.config import TARGET_COLUMNS  # noqa: E402


def _full_thresholds_map(default: float = 0.5) -> dict:
    return {label: default for label in TARGET_COLUMNS}


def _write_contract_thresholds(
    path: Path,
    *,
    baseline_micro: float = 0.6458,
    baseline_weighted: float = 0.9370,
    optimized_weighted: float = 0.8966,
    eval_critical_recall: float = 0.6148,
    optimization_split: str = "calibration",
    reporting_split: str = "frozen_eval",
    metadata_eval_critical_recall: float | None = None,
    thresholds_map: dict | None = None,
) -> None:
    if metadata_eval_critical_recall is None:
        metadata_eval_critical_recall = eval_critical_recall
    payload = {
        "metadata": {
            "optimization_split": optimization_split,
            "reporting_split": reporting_split,
            "eval_critical_recall": metadata_eval_critical_recall,
        },
        "thresholds": thresholds_map if thresholds_map is not None else _full_thresholds_map(),
        "performance": {
            "baseline": {
                "f1_weighted": baseline_weighted,
                "f1_micro": baseline_micro,
            },
            "optimized": {
                "f1_weighted": optimized_weighted,
                "f1_micro": 0.4534,
                "critical_recall": eval_critical_recall,
            },
            "delta": {
                "f1_weighted": optimized_weighted - baseline_weighted,
                "f1_weighted_pct": (
                    ((optimized_weighted - baseline_weighted) / baseline_weighted) * 100.0
                ),
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_training_log(path: Path, micro_f1: float = 0.6458, overall_f1: float = 0.9370) -> None:
    path.write_text(
        json.dumps(
            {
                "performance": {
                    "overall_f1": overall_f1,
                    "micro_f1": micro_f1,
                }
            }
        ),
        encoding="utf-8",
    )


def _build_contract_candidate(
    candidate_dir: Path,
    model_path: Path,
    model_name: str,
    **threshold_kwargs,
) -> Path:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    dest_model = candidate_dir / model_name
    shutil.copy2(model_path, dest_model)
    _write_training_log(candidate_dir / "training_log.json")
    _write_contract_thresholds(
        candidate_dir / f"{dest_model.stem}_thresholds.json",
        **threshold_kwargs,
    )
    return candidate_dir


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def rf_model_path(temp_dir):
    """Create a temporary RandomForest model file."""
    model_path = temp_dir / "rf_model.pkl"
    pipeline = Pipeline([
        ('vect', TfidfVectorizer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=1, random_state=42)))
    ])
    X = ["test message", "another message"]
    y = [[1, 0], [0, 1]]
    pipeline.fit(X, y)
    joblib.dump(pipeline, model_path)
    return model_path


@pytest.fixture
def lr_model_path(temp_dir):
    """Create a temporary LogisticRegression model file."""
    model_path = temp_dir / "lr_model.pkl"
    pipeline = Pipeline([
        ('vect', TfidfVectorizer()),
        ('clf', MultiOutputClassifier(LogisticRegression(random_state=42, max_iter=100)))
    ])
    X = ["test message", "another message"]
    y = [[1, 0], [0, 1]]
    pipeline.fit(X, y)
    joblib.dump(pipeline, model_path)
    return model_path


@pytest.fixture
def candidate_dir_with_rf_model(temp_dir, rf_model_path):
    """RF candidate with evaluation-contract artifacts."""
    return _build_contract_candidate(
        temp_dir / "2025-11-06-rf-candidate",
        rf_model_path,
        "rf_model.pkl",
    )


@pytest.fixture
def candidate_dir_with_lr_model(temp_dir, lr_model_path):
    """LR candidate with evaluation-contract artifacts."""
    return _build_contract_candidate(
        temp_dir / "2025-11-06-lr-candidate",
        lr_model_path,
        "lr_model.pkl",
    )


@pytest.fixture
def candidate_dir_with_metrics_csv(temp_dir, lr_model_path):
    """LR candidate with contract artifacts plus performance_metrics.csv."""
    candidate_dir = _build_contract_candidate(
        temp_dir / "2025-11-06-lr-with-metrics",
        lr_model_path,
        "lr_model.pkl",
    )
    metrics_df = pd.DataFrame({
        'category': ['related', 'related'],
        'output_class': ['0', '1'],
        'precision': [0.8, 0.9],
        'recall': [0.7, 0.85],
        'f1-score': [0.75, 0.875],
        'support': [100.0, 200.0]
    })
    metrics_df.to_csv(candidate_dir / "performance_metrics.csv", index=False)
    return candidate_dir


class TestWeightedF1RelativeDrop:
    """Guardrail formula must be relative, with an exact 0.05 boundary."""

    def test_relative_drop_formula(self):
        assert weighted_f1_relative_drop(1.0, 0.95) == pytest.approx(0.05)
        assert weighted_f1_relative_drop(0.9370, 0.8966) == pytest.approx(
            (0.9370 - 0.8966) / 0.9370
        )

    def test_exact_boundary_passes_validation(self, temp_dir, lr_model_path):
        """relative drop == 0.05 must pass (<= max)."""
        candidate = _build_contract_candidate(
            temp_dir / "2026-09-21-boundary-pass",
            lr_model_path,
            "lr_model.pkl",
            baseline_weighted=1.0,
            optimized_weighted=0.95,
        )
        results = validate_candidate_model(candidate)
        assert results['validation_passed'] is True
        assert results['weighted_f1_relative_drop'] == pytest.approx(0.05)

    def test_just_over_boundary_fails_validation(self, temp_dir, lr_model_path):
        """relative drop > 0.05 must fail."""
        candidate = _build_contract_candidate(
            temp_dir / "2026-09-21-boundary-fail",
            lr_model_path,
            "lr_model.pkl",
            baseline_weighted=1.0,
            optimized_weighted=0.949,
        )
        results = validate_candidate_model(candidate)
        assert results['validation_passed'] is False
        assert any("Weighted F1 relative drop" in e for e in results['validation_errors'])
        assert results['weighted_f1_relative_drop'] == pytest.approx(0.051)


class TestEvaluationContractValidation:
    """Promotion gates enforce the train/cal/eval contract."""

    def test_contract_candidate_passes(self, candidate_dir_with_lr_model):
        results = validate_candidate_model(candidate_dir_with_lr_model)
        assert results['validation_passed'] is True
        assert results['baseline_f1_micro'] == pytest.approx(0.6458)
        assert results['eval_critical_recall'] == pytest.approx(0.6148)
        assert results['thresholds_path'] is not None
        assert results['thresholds_sha256'] is not None
        assert Path(results['model_path']).is_absolute()
        assert Path(results['thresholds_path']).is_absolute()
        assert results['optimization_split'] == "calibration"
        assert results['reporting_split'] == "frozen_eval"

    def test_fails_wrong_optimization_split(self, temp_dir, lr_model_path):
        candidate = _build_contract_candidate(
            temp_dir / "2026-09-21-bad-opt",
            lr_model_path,
            "lr_model.pkl",
            optimization_split="frozen_eval",
        )
        results = validate_candidate_model(candidate)
        assert results['validation_passed'] is False
        assert any("optimization_split" in e for e in results['validation_errors'])

    def test_fails_wrong_reporting_split(self, temp_dir, lr_model_path):
        candidate = _build_contract_candidate(
            temp_dir / "2026-09-21-bad-report",
            lr_model_path,
            "lr_model.pkl",
            reporting_split="calibration",
        )
        results = validate_candidate_model(candidate)
        assert results['validation_passed'] is False
        assert any("reporting_split" in e for e in results['validation_errors'])

    def test_fails_missing_micro_f1(self, temp_dir, lr_model_path):
        candidate = _build_contract_candidate(
            temp_dir / "2026-09-21-no-micro",
            lr_model_path,
            "lr_model.pkl",
        )
        (candidate / "training_log.json").write_text(
            json.dumps({"performance": {"overall_f1": 0.94, "samples_f1": 0.70}}),
            encoding="utf-8",
        )
        results = validate_candidate_model(candidate)
        assert results['validation_passed'] is False
        assert any("micro_f1" in e for e in results['validation_errors'])

    def test_fails_low_baseline_micro(self, temp_dir, lr_model_path):
        candidate = _build_contract_candidate(
            temp_dir / "2026-09-21-low-micro",
            lr_model_path,
            "lr_model.pkl",
            baseline_micro=0.50,
        )
        _write_training_log(candidate / "training_log.json", micro_f1=0.50)
        results = validate_candidate_model(candidate)
        assert results['validation_passed'] is False
        assert any("Baseline frozen-eval micro F1" in e for e in results['validation_errors'])

    def test_fails_low_eval_critical_recall(self, temp_dir, lr_model_path):
        candidate = _build_contract_candidate(
            temp_dir / "2026-09-21-low-cr",
            lr_model_path,
            "lr_model.pkl",
            eval_critical_recall=0.40,
        )
        results = validate_candidate_model(candidate)
        assert results['validation_passed'] is False
        assert any("critical recall" in e for e in results['validation_errors'])

    def test_fails_oversized_model(self, temp_dir, lr_model_path):
        candidate = _build_contract_candidate(
            temp_dir / "2026-09-21-too-big",
            lr_model_path,
            "lr_model.pkl",
        )
        oversized = candidate / "lr_model.pkl"
        with open(oversized, "wb") as handle:
            handle.seek(51 * 1024 * 1024)
            handle.write(b"x")
        results = validate_candidate_model(candidate)
        assert results['validation_passed'] is False
        assert any("Model size" in e for e in results['validation_errors'])

    def test_fails_multiple_pkl_files(self, temp_dir, lr_model_path, rf_model_path):
        candidate = _build_contract_candidate(
            temp_dir / "2026-09-21-multi-pkl",
            lr_model_path,
            "lr_model.pkl",
        )
        shutil.copy2(rf_model_path, candidate / "extra_model.pkl")
        results = validate_candidate_model(candidate)
        assert results['validation_passed'] is False
        assert any("Multiple model files" in e for e in results['validation_errors'])

    def test_fails_inconsistent_baseline_micro(self, temp_dir, lr_model_path):
        candidate = _build_contract_candidate(
            temp_dir / "2026-09-21-micro-mismatch",
            lr_model_path,
            "lr_model.pkl",
            baseline_micro=0.70,
        )
        _write_training_log(candidate / "training_log.json", micro_f1=0.6458)
        results = validate_candidate_model(candidate)
        assert results['validation_passed'] is False
        assert any("Inconsistent baseline micro F1" in e for e in results['validation_errors'])

    def test_fails_inconsistent_eval_critical_recall(self, temp_dir, lr_model_path):
        candidate = _build_contract_candidate(
            temp_dir / "2026-09-21-cr-mismatch",
            lr_model_path,
            "lr_model.pkl",
            eval_critical_recall=0.6148,
            metadata_eval_critical_recall=0.70,
        )
        results = validate_candidate_model(candidate)
        assert results['validation_passed'] is False
        assert any("Inconsistent critical recall" in e for e in results['validation_errors'])

    def test_fails_incomplete_thresholds_map(self, temp_dir, lr_model_path):
        candidate = _build_contract_candidate(
            temp_dir / "2026-09-21-partial-thresholds",
            lr_model_path,
            "lr_model.pkl",
            thresholds_map={"related": 0.5},
        )
        results = validate_candidate_model(candidate)
        assert results['validation_passed'] is False
        assert any("Thresholds map missing" in e for e in results['validation_errors'])

    def test_fails_out_of_range_threshold_value(self, temp_dir, lr_model_path):
        bad_map = _full_thresholds_map()
        bad_map["food"] = 1.5
        candidate = _build_contract_candidate(
            temp_dir / "2026-09-21-bad-threshold-value",
            lr_model_path,
            "lr_model.pkl",
            thresholds_map=bad_map,
        )
        results = validate_candidate_model(candidate)
        assert results['validation_passed'] is False
        assert any("invalid value" in e for e in results['validation_errors'])

    def test_fails_unloadable_model_algorithm(self, temp_dir, lr_model_path):
        candidate = _build_contract_candidate(
            temp_dir / "2026-09-21-corrupt-model",
            lr_model_path,
            "lr_model.pkl",
        )
        (candidate / "lr_model.pkl").write_bytes(b"not-a-real-pickle")
        results = validate_candidate_model(candidate)
        assert results['validation_passed'] is False
        assert results['algorithm'] == 'unknown'
        assert any("Unsupported or unloadable model" in e for e in results['validation_errors'])
        with pytest.raises(ValueError, match="structural promotion prerequisites"):
            assert_force_promotion_prerequisites(results)

    def test_missing_thresholds_is_validation_error_not_raise(self, temp_dir, lr_model_path):
        candidate = temp_dir / "2026-09-21-no-thresholds"
        candidate.mkdir()
        shutil.copy2(lr_model_path, candidate / "lr_model.pkl")
        _write_training_log(candidate / "training_log.json")
        results = validate_candidate_model(candidate)
        assert results['validation_passed'] is False
        assert any("thresholds artifact" in e for e in results['validation_errors'])

    def test_missing_model_is_validation_error_not_raise(self, temp_dir):
        candidate = temp_dir / "2026-09-21-no-model"
        candidate.mkdir()
        _write_training_log(candidate / "training_log.json")
        results = validate_candidate_model(candidate)
        assert results['validation_passed'] is False
        assert any("No model file" in e for e in results['validation_errors'])


class TestForcePathPrerequisites:
    """--force overrides gates, not structural deploy prerequisites."""

    def test_force_prerequisites_pass_when_artifacts_present(
        self, candidate_dir_with_lr_model
    ):
        results = validate_candidate_model(candidate_dir_with_lr_model)
        # Simulate metric/provenance override: mark failed but keep structural fields
        results['validation_passed'] = False
        results['validation_errors'] = ["simulated provenance failure"]
        assert_force_promotion_prerequisites(results)

    def test_force_prerequisites_reject_missing_thresholds(self, temp_dir, lr_model_path):
        candidate = temp_dir / "2026-09-21-force-no-thresholds"
        candidate.mkdir()
        shutil.copy2(lr_model_path, candidate / "lr_model.pkl")
        _write_training_log(candidate / "training_log.json")
        results = validate_candidate_model(candidate)
        assert results['thresholds_path'] is None
        with pytest.raises(ValueError, match="structural promotion prerequisites"):
            assert_force_promotion_prerequisites(results)

    def test_force_can_promote_despite_gate_failure_when_artifacts_present(
        self, temp_dir, lr_model_path
    ):
        candidate = _build_contract_candidate(
            temp_dir / "2026-09-21-force-ok",
            lr_model_path,
            "lr_model.pkl",
            optimization_split="frozen_eval",  # gate failure
        )
        results = validate_candidate_model(candidate)
        assert results['validation_passed'] is False
        assert_force_promotion_prerequisites(results)

        model_dir = temp_dir / "model"
        model_dir.mkdir()
        promotion_record = promote_model(candidate, model_dir, results)
        promoted = Path(promotion_record['promoted_model'])
        deployed = model_dir / f"{promoted.stem}_thresholds.json"
        assert deployed.exists()
        assert compute_model_hash(deployed) == results['thresholds_sha256']

    def test_promote_refuses_missing_thresholds_even_if_forced_fields_cleared(
        self, temp_dir, candidate_dir_with_lr_model
    ):
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        results = validate_candidate_model(candidate_dir_with_lr_model)
        results['thresholds_path'] = None
        results['thresholds_sha256'] = None
        with pytest.raises(ValueError, match="structural promotion prerequisites"):
            promote_model(candidate_dir_with_lr_model, model_dir, results)


class TestThresholdDeployInvariant:
    """Deployed thresholds must be the exact artifact validation scored."""

    def test_promoted_thresholds_sha_matches_validated_candidate(
        self, temp_dir, candidate_dir_with_lr_model
    ):
        model_dir = temp_dir / "model"
        model_dir.mkdir()

        validation_results = validate_candidate_model(candidate_dir_with_lr_model)
        assert validation_results['validation_passed'] is True
        expected_sha = validation_results['thresholds_sha256']
        source_path = Path(validation_results['thresholds_path'])

        promotion_record = promote_model(
            candidate_dir_with_lr_model, model_dir, validation_results
        )
        promoted_model = Path(promotion_record['promoted_model'])
        deployed = model_dir / f"{promoted_model.stem}_thresholds.json"

        assert deployed.exists()
        assert compute_model_hash(deployed) == expected_sha
        assert compute_model_hash(deployed) == compute_model_hash(source_path)
        assert deployed.read_bytes() == source_path.read_bytes()

    def test_threshold_hash_failure_leaves_no_discoverable_production_model(
        self, temp_dir, candidate_dir_with_lr_model
    ):
        model_dir = temp_dir / "model"
        model_dir.mkdir()

        validation_results = validate_candidate_model(candidate_dir_with_lr_model)
        assert validation_results['validation_passed'] is True
        validation_results['thresholds_sha256'] = '0' * 64

        with pytest.raises(ValueError, match="[Tt]hresholds .*integrity check failed"):
            promote_model(candidate_dir_with_lr_model, model_dir, validation_results)

        assert list(model_dir.glob("disaster_*_prod_*.pkl")) == []
        assert list(model_dir.glob("*_thresholds.json")) == []
        assert list(model_dir.glob(".promotion_staging_*")) == []


class TestProductionArtifactImmutability:
    """Production model/threshold filenames are immutable once created."""

    def test_idempotent_retry_with_matching_existing_pair(
        self, temp_dir, candidate_dir_with_lr_model
    ):
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        validation_results = validate_candidate_model(candidate_dir_with_lr_model)
        first = promote_model(candidate_dir_with_lr_model, model_dir, validation_results)
        prod_model = Path(first['promoted_model'])
        prod_thresholds = model_dir / f"{prod_model.stem}_thresholds.json"

        model_bytes_before = prod_model.read_bytes()
        thresholds_bytes_before = prod_thresholds.read_bytes()
        model_hash_before = compute_model_hash(prod_model)
        thresholds_hash_before = compute_model_hash(prod_thresholds)

        second = promote_model(candidate_dir_with_lr_model, model_dir, validation_results)
        assert Path(second['promoted_model']) == prod_model
        assert prod_model.read_bytes() == model_bytes_before
        assert prod_thresholds.read_bytes() == thresholds_bytes_before
        assert compute_model_hash(prod_model) == model_hash_before
        assert compute_model_hash(prod_thresholds) == thresholds_hash_before

    def test_collision_with_different_existing_thresholds(
        self, temp_dir, candidate_dir_with_lr_model
    ):
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        validation_results = validate_candidate_model(candidate_dir_with_lr_model)
        first = promote_model(candidate_dir_with_lr_model, model_dir, validation_results)
        prod_model = Path(first['promoted_model'])
        prod_thresholds = model_dir / f"{prod_model.stem}_thresholds.json"

        model_bytes_before = prod_model.read_bytes()
        thresholds_bytes_before = prod_thresholds.read_bytes()
        model_hash_before = compute_model_hash(prod_model)
        thresholds_hash_before = compute_model_hash(prod_thresholds)

        prod_thresholds.write_text(
            json.dumps({"thresholds": {"related": 0.1}, "tampered": True}),
            encoding="utf-8",
        )
        tampered_thresholds_hash = compute_model_hash(prod_thresholds)

        with pytest.raises(ValueError, match="Production artifact collision"):
            promote_model(candidate_dir_with_lr_model, model_dir, validation_results)

        assert prod_model.read_bytes() == model_bytes_before
        assert compute_model_hash(prod_model) == model_hash_before
        assert prod_thresholds.read_bytes() != thresholds_bytes_before
        assert compute_model_hash(prod_thresholds) == tampered_thresholds_hash
        assert compute_model_hash(prod_thresholds) != thresholds_hash_before

    def test_collision_with_different_existing_model(
        self, temp_dir, candidate_dir_with_lr_model
    ):
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        validation_results = validate_candidate_model(candidate_dir_with_lr_model)
        first = promote_model(candidate_dir_with_lr_model, model_dir, validation_results)
        prod_model = Path(first['promoted_model'])
        prod_thresholds = model_dir / f"{prod_model.stem}_thresholds.json"

        model_bytes_before = prod_model.read_bytes()
        thresholds_bytes_before = prod_thresholds.read_bytes()
        thresholds_hash_before = compute_model_hash(prod_thresholds)

        prod_model.write_bytes(b"tampered-production-model-bytes")
        tampered_model_hash = compute_model_hash(prod_model)

        with pytest.raises(ValueError, match="Production artifact collision"):
            promote_model(candidate_dir_with_lr_model, model_dir, validation_results)

        assert prod_model.read_bytes() == b"tampered-production-model-bytes"
        assert compute_model_hash(prod_model) == tampered_model_hash
        assert prod_model.read_bytes() != model_bytes_before
        assert prod_thresholds.read_bytes() == thresholds_bytes_before
        assert compute_model_hash(prod_thresholds) == thresholds_hash_before

    def test_incomplete_existing_pair_fails_closed(
        self, temp_dir, candidate_dir_with_lr_model
    ):
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        validation_results = validate_candidate_model(candidate_dir_with_lr_model)
        first = promote_model(candidate_dir_with_lr_model, model_dir, validation_results)
        prod_model = Path(first['promoted_model'])
        prod_thresholds = model_dir / f"{prod_model.stem}_thresholds.json"

        model_bytes_before = prod_model.read_bytes()
        model_hash_before = compute_model_hash(prod_model)
        prod_thresholds.unlink()

        with pytest.raises(ValueError, match="Incomplete production artifact pair"):
            promote_model(candidate_dir_with_lr_model, model_dir, validation_results)

        assert prod_model.exists()
        assert prod_model.read_bytes() == model_bytes_before
        assert compute_model_hash(prod_model) == model_hash_before
        assert not prod_thresholds.exists()

    def test_rejected_retry_preserves_existing_artifacts_byte_for_byte(
        self, temp_dir, candidate_dir_with_lr_model
    ):
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        validation_results = validate_candidate_model(candidate_dir_with_lr_model)
        first = promote_model(candidate_dir_with_lr_model, model_dir, validation_results)
        prod_model = Path(first['promoted_model'])
        prod_thresholds = model_dir / f"{prod_model.stem}_thresholds.json"

        # Different content in both destinations triggers collision without rewrite
        original_model = prod_model.read_bytes()
        original_thresholds = prod_thresholds.read_bytes()
        prod_model.write_bytes(original_model + b"\x00extra")
        prod_thresholds.write_text(original_thresholds.decode("utf-8") + "\n", encoding="utf-8")
        model_after_tamper = prod_model.read_bytes()
        thresholds_after_tamper = prod_thresholds.read_bytes()
        model_hash_after_tamper = compute_model_hash(prod_model)
        thresholds_hash_after_tamper = compute_model_hash(prod_thresholds)

        with pytest.raises(ValueError, match="Production artifact collision"):
            promote_model(candidate_dir_with_lr_model, model_dir, validation_results)

        assert prod_model.read_bytes() == model_after_tamper
        assert prod_thresholds.read_bytes() == thresholds_after_tamper
        assert compute_model_hash(prod_model) == model_hash_after_tamper
        assert compute_model_hash(prod_thresholds) == thresholds_hash_after_tamper


class TestRealCandidateAcceptance:
    """Acceptance behavior against checked-in experiment artifacts."""

    def test_2026_09_21_passes_when_model_present(self, temp_dir, lr_model_path):
        """If the run dir lacks a .pkl (gitignored), inject a stand-in with the real stem."""
        source = PROJECT_ROOT / "experiments" / "experimental_runs" / "2026-09-21"
        if not source.exists():
            pytest.skip("2026-09-21 experiment run not found")

        candidate = temp_dir / "2026-09-21"
        shutil.copytree(source, candidate)
        model_name = "lr_vocab15k_cal_split_model.pkl"
        if not (candidate / model_name).exists():
            shutil.copy2(lr_model_path, candidate / model_name)

        results = validate_candidate_model(candidate)
        assert results['validation_passed'] is True, results['validation_errors']
        assert results['optimization_split'] == "calibration"
        assert results['reporting_split'] == "frozen_eval"
        assert results['baseline_f1_micro'] == pytest.approx(0.6454499163453844)
        assert results['eval_critical_recall'] == pytest.approx(0.6149371857288635)

    def test_2025_11_06_fails_tune_on_eval_provenance(self, temp_dir, lr_model_path):
        source = (
            PROJECT_ROOT
            / "experiments"
            / "experimental_runs"
            / "2025-11-06-vocab15k-promotion"
        )
        if not source.exists():
            pytest.skip("2025-11-06 vocab15k promotion run not found")

        candidate = temp_dir / "2025-11-06-vocab15k-promotion"
        shutil.copytree(source, candidate)
        model_name = "lr_vocab15k_model.pkl"
        if not (candidate / model_name).exists():
            shutil.copy2(lr_model_path, candidate / model_name)

        results = validate_candidate_model(candidate)
        assert results['validation_passed'] is False
        assert any("optimization_split" in e for e in results['validation_errors'])


class TestAlgorithmDetection:
    """Test algorithm type detection from model files."""

    def test_detect_rf_algorithm(self, rf_model_path):
        algorithm = detect_algorithm_type(rf_model_path)
        assert algorithm == 'rf'

    def test_detect_lr_algorithm(self, lr_model_path):
        algorithm = detect_algorithm_type(lr_model_path)
        assert algorithm == 'lr'

    def test_detect_algorithm_handles_missing_file(self, temp_dir):
        missing_path = temp_dir / "nonexistent.pkl"
        algorithm = detect_algorithm_type(missing_path)
        assert algorithm == 'unknown'

    def test_detect_algorithm_handles_invalid_model(self, temp_dir):
        invalid_path = temp_dir / "invalid.pkl"
        invalid_path.write_text("not a valid pickle file")
        algorithm = detect_algorithm_type(invalid_path)
        assert algorithm == 'unknown'


class TestModelHashVerification:
    """Test model file hash computation and verification."""

    def test_compute_model_hash(self, rf_model_path):
        hash1 = compute_model_hash(rf_model_path)
        hash2 = compute_model_hash(rf_model_path)
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_hash_different_for_different_models(self, rf_model_path, lr_model_path):
        assert compute_model_hash(rf_model_path) != compute_model_hash(lr_model_path)


class TestPromotionFlow:
    """Test the full promotion flow."""

    def test_promote_rf_model_generates_correct_filename(self, temp_dir, candidate_dir_with_rf_model):
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        validation_results = validate_candidate_model(candidate_dir_with_rf_model)
        promotion_record = promote_model(candidate_dir_with_rf_model, model_dir, validation_results)
        promoted_path = Path(promotion_record['promoted_model'])
        assert promoted_path.name.startswith('disaster_rf_')
        assert promoted_path.exists()

    def test_promote_lr_model_generates_correct_filename(self, temp_dir, candidate_dir_with_lr_model):
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        validation_results = validate_candidate_model(candidate_dir_with_lr_model)
        promotion_record = promote_model(candidate_dir_with_lr_model, model_dir, validation_results)
        promoted_path = Path(promotion_record['promoted_model'])
        assert promoted_path.name.startswith('disaster_lr_')
        assert promoted_path.exists()

    def test_promoted_model_hash_matches_validation(self, temp_dir, candidate_dir_with_rf_model):
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        validation_results = validate_candidate_model(candidate_dir_with_rf_model)
        expected_hash = validation_results['model_hash']
        promotion_record = promote_model(candidate_dir_with_rf_model, model_dir, validation_results)
        promoted_path = Path(promotion_record['promoted_model'])
        assert compute_model_hash(promoted_path) == expected_hash

    def test_model_info_includes_algorithm_metadata(self, temp_dir, candidate_dir_with_rf_model):
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        validation_results = validate_candidate_model(candidate_dir_with_rf_model)
        promote_model(candidate_dir_with_rf_model, model_dir, validation_results)
        model_info = json.loads((model_dir / "MODEL_INFO.json").read_text(encoding='utf-8'))
        assert model_info['algorithm'] == 'rf'
        assert model_info['algorithm_name'] == 'RandomForest'

    def test_model_info_includes_lr_algorithm_metadata(self, temp_dir, candidate_dir_with_lr_model):
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        validation_results = validate_candidate_model(candidate_dir_with_lr_model)
        promote_model(candidate_dir_with_lr_model, model_dir, validation_results)
        model_info = json.loads((model_dir / "MODEL_INFO.json").read_text(encoding='utf-8'))
        assert model_info['algorithm'] == 'lr'
        assert model_info['algorithm_name'] == 'LogisticRegression'

    def test_promotion_record_includes_algorithm_info(self, temp_dir, candidate_dir_with_rf_model):
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        validation_results = validate_candidate_model(candidate_dir_with_rf_model)
        promotion_record = promote_model(candidate_dir_with_rf_model, model_dir, validation_results)
        assert promotion_record['status'] == 'promoted'
        assert 'rf' in Path(promotion_record['promoted_model']).name


class TestHashMismatchProtection:
    """Test that hash mismatch errors are caught and reported."""

    def test_promotion_fails_on_hash_mismatch(self, temp_dir, candidate_dir_with_rf_model):
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        validation_results = validate_candidate_model(candidate_dir_with_rf_model)
        validation_results['model_hash'] = '0' * 64
        with pytest.raises(ValueError, match="[Mm]odel .*integrity check failed"):
            promote_model(candidate_dir_with_rf_model, model_dir, validation_results)
        assert list(model_dir.glob("disaster_*_prod_*.pkl")) == []


class TestIntegrationWithRealModels:
    """Integration tests using actual model files if available."""

    def test_detect_algorithm_from_production_rf_model(self):
        prod_model = PROJECT_ROOT / "model" / "disaster_rf_prod_2026-01-22.pkl"
        if prod_model.exists():
            assert detect_algorithm_type(prod_model) == 'rf'
        else:
            pytest.skip("Production RF model not found")

    def test_detect_algorithm_from_experimental_lr_model(self):
        lr_model = (
            PROJECT_ROOT
            / "experiments"
            / "experimental_runs"
            / "2025-11-06-vocab15k-promotion"
            / "lr_vocab15k_model.pkl"
        )
        if lr_model.exists():
            assert detect_algorithm_type(lr_model) == 'lr'
        else:
            pytest.skip("Experimental LR model not found")


class TestFilenameGeneration:
    """Test that filenames are generated correctly based on algorithm type."""

    def test_rf_filename_format(self, temp_dir, candidate_dir_with_rf_model):
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        validation_results = validate_candidate_model(candidate_dir_with_rf_model)
        promotion_record = promote_model(candidate_dir_with_rf_model, model_dir, validation_results)
        filename = Path(promotion_record['promoted_model']).name
        parts = filename.split('_')
        assert parts[0] == 'disaster'
        assert parts[1] == 'rf'
        assert parts[2].startswith('v')
        assert 'prod' in parts
        assert filename.endswith('.pkl')

    def test_lr_filename_format(self, temp_dir, candidate_dir_with_lr_model):
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        validation_results = validate_candidate_model(candidate_dir_with_lr_model)
        promotion_record = promote_model(candidate_dir_with_lr_model, model_dir, validation_results)
        filename = Path(promotion_record['promoted_model']).name
        parts = filename.split('_')
        assert parts[0] == 'disaster'
        assert parts[1] == 'lr'
        assert parts[2].startswith('v')
        assert 'prod' in parts
        assert filename.endswith('.pkl')


class TestMetricsFileNaming:
    """Test that performance_metrics.csv uses model-specific naming."""

    def test_metrics_file_copied_with_model_specific_naming(
        self, temp_dir, candidate_dir_with_metrics_csv
    ):
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        validation_results = validate_candidate_model(candidate_dir_with_metrics_csv)
        promotion_record = promote_model(
            candidate_dir_with_metrics_csv, model_dir, validation_results
        )
        base_name = Path(promotion_record['promoted_model']).stem
        expected_metrics_file = model_dir / f"{base_name}_performance_metrics.csv"
        assert expected_metrics_file.exists()
        source_metrics = pd.read_csv(candidate_dir_with_metrics_csv / "performance_metrics.csv")
        promoted_metrics = pd.read_csv(expected_metrics_file)
        pd.testing.assert_frame_equal(source_metrics, promoted_metrics)

    def test_metrics_file_not_copied_if_missing(self, temp_dir, candidate_dir_with_lr_model):
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        validation_results = validate_candidate_model(candidate_dir_with_lr_model)
        promotion_record = promote_model(candidate_dir_with_lr_model, model_dir, validation_results)
        assert promotion_record['status'] == 'promoted'
        base_name = Path(promotion_record['promoted_model']).stem
        assert not (model_dir / f"{base_name}_performance_metrics.csv").exists()

    def test_discover_production_metrics_file_finds_model_specific_file(
        self, temp_dir, candidate_dir_with_metrics_csv
    ):
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        validation_results = validate_candidate_model(candidate_dir_with_metrics_csv)
        promotion_record = promote_model(
            candidate_dir_with_metrics_csv, model_dir, validation_results
        )
        discovered_metrics = discover_production_metrics_file(model_dir)
        base_name = Path(promotion_record['promoted_model']).stem
        expected_metrics_file = model_dir / f"{base_name}_performance_metrics.csv"
        assert discovered_metrics == expected_metrics_file

    def test_discover_production_metrics_file_falls_back_to_legacy_naming(self, temp_dir):
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        model_file = model_dir / "disaster_lr_v25-11-06_prod_2025-11-06.pkl"
        model_file.write_bytes(b"fake model data")
        legacy_metrics = model_dir / "performance_metrics.csv"
        legacy_metrics.write_text("category,output_class,precision\nrelated,1,0.9")
        discovered_metrics = discover_production_metrics_file(model_dir)
        assert discovered_metrics == legacy_metrics

    def test_discover_production_metrics_file_returns_none_when_no_metrics(self, temp_dir):
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        model_file = model_dir / "disaster_lr_v25-11-06_prod_2025-11-06.pkl"
        model_file.write_bytes(b"fake model data")
        assert discover_production_metrics_file(model_dir) is None

    def test_discover_production_metrics_file_handles_multiple_models(self, temp_dir):
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        model1 = model_dir / "disaster_lr_v25-11-05_prod_2025-11-05.pkl"
        model1.write_bytes(b"older model")
        time.sleep(0.1)
        model2 = model_dir / "disaster_lr_v25-11-06_prod_2025-11-06.pkl"
        model2.write_bytes(b"newer model")
        metrics2 = model_dir / "disaster_lr_v25-11-06_prod_2025-11-06_performance_metrics.csv"
        metrics2.write_text("category,output_class,precision\nrelated,1,0.9")
        discovered_metrics = discover_production_metrics_file(model_dir)
        assert discovered_metrics == metrics2


class TestUpdateAppConfigModelFilename:
    """Protect the config-rewrite helper from eating Config bodies again."""

    def test_auto_discovery_config_unchanged(self, temp_dir):
        config_path = temp_dir / "config.py"
        original = (
            "from pathlib import Path\n\n"
            "def _discover_latest_model(models_dir: Path) -> str:\n"
            "    return 'disaster_lr_v26-09-21_prod_2026-09-21.pkl'\n\n"
            "class Config:\n"
            "    MODELS_DIR = Path('model')\n"
            "    _MODEL_FILENAME = None\n"
            "    if _MODEL_FILENAME:\n"
            "        MODEL_FILENAME = _MODEL_FILENAME\n"
            "    else:\n"
            "        MODEL_FILENAME = _discover_latest_model(MODELS_DIR)\n"
            "    MODEL_PATH = MODELS_DIR / MODEL_FILENAME\n\n"
            "class TestConfig(Config):\n"
            "    MODEL_FILENAME = 'test_model.pkl'\n"
        )
        config_path.write_text(original, encoding="utf-8")
        before = config_path.read_bytes()

        updated = _update_app_config_model_filename(
            config_path,
            "disaster_lr_v26-09-21_prod_2026-09-21.pkl",
            backup=False,
        )

        assert updated is False
        assert config_path.read_bytes() == before

    def test_hardcoded_production_literal_rewrites_only_that_line(self, temp_dir):
        config_path = temp_dir / "config.py"
        original = (
            "class Config:\n"
            "    SECRET_KEY = 'dev'\n"
            "    MODEL_FILENAME = 'disaster_lr_v25-11-06_prod_2025-11-06.pkl'\n"
            "    MODEL_PATH = MODELS_DIR / MODEL_FILENAME\n"
            "    KEEP_ME = True\n\n"
            "class TestConfig(Config):\n"
            "    MODEL_FILENAME = 'test_model.pkl'\n"
        )
        config_path.write_text(original, encoding="utf-8")

        updated = _update_app_config_model_filename(
            config_path,
            "disaster_lr_v26-09-21_prod_2026-09-21.pkl",
            backup=False,
        )

        assert updated is True
        text = config_path.read_text(encoding="utf-8")
        assert "SECRET_KEY = 'dev'" in text
        assert "KEEP_ME = True" in text
        assert "class TestConfig(Config):" in text
        assert "MODEL_FILENAME = 'test_model.pkl'" in text
        assert (
            "MODEL_FILENAME = 'disaster_lr_v26-09-21_prod_2026-09-21.pkl'" in text
        )
        assert "disaster_lr_v25-11-06_prod_2025-11-06.pkl" not in text
        # Exactly one production disaster_* literal remains (TestConfig stays test_model).
        assert text.count("disaster_lr_v26-09-21_prod_2026-09-21.pkl") == 1
