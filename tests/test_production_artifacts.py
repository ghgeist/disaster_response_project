"""Strict production artifact resolution and fail-closed inference contracts."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.services.category_mapper import CategoryMapper
from app.services.errors import ModelServiceError
from app.services.model_predictor import ModelPredictor
from app.services.model_service import ModelService
from app.services.production_artifacts import (
    ProductionArtifactError,
    compute_file_sha256,
    resolve_production_artifacts,
)
from app.services.threshold_manager import ThresholdManager
from disasterproject.utils.config import TARGET_COLUMNS


def _write_thresholds(path: Path, overrides: dict | None = None) -> None:
    thresholds = {label: 0.5 for label in TARGET_COLUMNS}
    if overrides:
        thresholds.update(overrides)
    path.write_text(json.dumps({"thresholds": thresholds}), encoding="utf-8")


def _write_labels(path: Path, labels: list[str] | None = None) -> None:
    path.write_text(json.dumps(labels or list(TARGET_COLUMNS)), encoding="utf-8")


def _write_model_info(
    path: Path,
    *,
    model_sha: str,
    thresholds_sha: str,
    labels_sha: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                "sha256": model_sha,
                "thresholds_sha256": thresholds_sha,
                "labels_sha256": labels_sha,
            }
        ),
        encoding="utf-8",
    )


def _install_production_bundle(tmp_path: Path, stem: str = "test_prod") -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_path = model_dir / f"{stem}.pkl"
    model_path.write_bytes(b"test-production-model-bytes")

    thresholds_path = model_dir / f"{stem}_thresholds.json"
    labels_path = model_dir / f"{stem}_labels.json"
    info_path = model_dir / "MODEL_INFO.json"
    _write_thresholds(thresholds_path)
    _write_labels(labels_path)
    _write_model_info(
        info_path,
        model_sha=compute_file_sha256(model_path),
        thresholds_sha=compute_file_sha256(thresholds_path),
        labels_sha=compute_file_sha256(labels_path),
    )
    return model_path


def test_resolve_production_artifacts_valid_bundle(tmp_path: Path) -> None:
    model_path = _install_production_bundle(tmp_path)
    artifacts = resolve_production_artifacts(model_path)
    assert len(artifacts.thresholds) == len(TARGET_COLUMNS)
    assert artifacts.label_order == list(TARGET_COLUMNS)


def test_missing_thresholds_raises(tmp_path: Path) -> None:
    model_path = _install_production_bundle(tmp_path)
    thresholds = model_path.with_name(f"{model_path.stem}_thresholds.json")
    thresholds.unlink()
    with pytest.raises(ProductionArtifactError, match="thresholds"):
        resolve_production_artifacts(model_path)


def test_missing_labels_raises(tmp_path: Path) -> None:
    model_path = _install_production_bundle(tmp_path)
    labels = model_path.with_name(f"{model_path.stem}_labels.json")
    labels.unlink()
    with pytest.raises(ProductionArtifactError, match="labels"):
        resolve_production_artifacts(model_path)


def test_corrupt_thresholds_json_raises(tmp_path: Path) -> None:
    model_path = _install_production_bundle(tmp_path)
    thresholds = model_path.with_name(f"{model_path.stem}_thresholds.json")
    thresholds.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ProductionArtifactError, match="Corrupt JSON"):
        resolve_production_artifacts(model_path)


def test_model_hash_mismatch_raises(tmp_path: Path) -> None:
    model_path = _install_production_bundle(tmp_path)
    info_path = model_path.with_name("MODEL_INFO.json")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["sha256"] = "0" * 64
    info_path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ProductionArtifactError, match="model SHA-256 mismatch"):
        resolve_production_artifacts(model_path)


def test_thresholds_hash_mismatch_raises(tmp_path: Path) -> None:
    model_path = _install_production_bundle(tmp_path)
    info_path = model_path.with_name("MODEL_INFO.json")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["thresholds_sha256"] = "1" * 64
    info_path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ProductionArtifactError, match="thresholds SHA-256 mismatch"):
        resolve_production_artifacts(model_path)


def test_labels_hash_mismatch_raises(tmp_path: Path) -> None:
    model_path = _install_production_bundle(tmp_path)
    info_path = model_path.with_name("MODEL_INFO.json")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["labels_sha256"] = "2" * 64
    info_path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ProductionArtifactError, match="labels SHA-256 mismatch"):
        resolve_production_artifacts(model_path)


def test_incomplete_threshold_map_raises(tmp_path: Path) -> None:
    model_path = _install_production_bundle(tmp_path)
    thresholds = model_path.with_name(f"{model_path.stem}_thresholds.json")
    thresholds.write_text(json.dumps({"thresholds": {"medical_help": 0.4}}), encoding="utf-8")
    info_path = model_path.with_name("MODEL_INFO.json")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["thresholds_sha256"] = compute_file_sha256(thresholds)
    info_path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ProductionArtifactError, match="incomplete"):
        resolve_production_artifacts(model_path)


def test_label_order_mismatch_raises(tmp_path: Path) -> None:
    model_path = _install_production_bundle(tmp_path)
    labels = model_path.with_name(f"{model_path.stem}_labels.json")
    reversed_labels = list(reversed(TARGET_COLUMNS))
    _write_labels(labels, reversed_labels)
    info_path = model_path.with_name("MODEL_INFO.json")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["labels_sha256"] = compute_file_sha256(labels)
    info_path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ProductionArtifactError, match="order mismatch"):
        resolve_production_artifacts(model_path)


def test_model_service_fails_closed_on_provenance_mismatch(tmp_path: Path) -> None:
    model_path = _install_production_bundle(tmp_path)
    info_path = model_path.with_name("MODEL_INFO.json")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["sha256"] = "0" * 64
    info_path.write_text(json.dumps(info), encoding="utf-8")

    service = ModelService(model_path)
    with pytest.raises(ModelServiceError, match="provenance failed"):
        service.load_model()


def test_predict_proba_failure_does_not_fallback_when_disabled() -> None:
    predictor = ModelPredictor(CategoryMapper(), ThresholdManager())
    broken = MagicMock()
    broken.predict_proba.side_effect = RuntimeError("probability path failed")
    broken.predict.return_value = [[0] * len(TARGET_COLUMNS)]

    thresholds = {label: 0.5 for label in TARGET_COLUMNS}
    with pytest.raises(RuntimeError, match="probability path failed"):
        predictor.predict(
            broken,
            "test",
            list(TARGET_COLUMNS),
            thresholds,
            allow_predict_fallback=False,
        )


def test_predict_proba_failure_can_fallback_when_explicitly_allowed() -> None:
    predictor = ModelPredictor(CategoryMapper(), ThresholdManager())
    broken = MagicMock()
    broken.predict_proba.side_effect = RuntimeError("probability path failed")
    broken.predict.return_value = [[0] * len(TARGET_COLUMNS)]

    thresholds = {label: 0.5 for label in TARGET_COLUMNS}
    result = predictor.predict(
        broken,
        "test",
        list(TARGET_COLUMNS),
        thresholds,
        allow_predict_fallback=True,
    )
    assert "labels" in result
