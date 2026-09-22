"""Threshold provenance and Model Information operating-point contracts."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from flask import Flask

from app.config import Config
from app.routes.api import (
    _find_production_thresholds_file,
    _resolve_active_production_model_path,
)
from disasterproject.utils.config import TARGET_COLUMNS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_production_thresholds_sha_matches_model_info_provenance() -> None:
    """Deployed thresholds must stay byte-identical to the validated evidence SHA."""
    model_path = Config.MODEL_PATH
    thresholds_path = model_path.with_name(f"{model_path.stem}_thresholds.json")
    labels_path = model_path.with_name(f"{model_path.stem}_labels.json")
    model_info_path = model_path.with_name("MODEL_INFO.json")

    if not model_path.exists() or not thresholds_path.exists() or not model_info_path.exists():
        pytest.skip("Production model/thresholds/MODEL_INFO not available")

    info = json.loads(model_info_path.read_text(encoding="utf-8"))
    expected = info.get("thresholds_sha256")
    assert isinstance(expected, str) and len(expected) == 64, (
        "MODEL_INFO.json must record thresholds_sha256 from promotion validation"
    )
    assert _sha256(thresholds_path) == expected, (
        "Production thresholds bytes must match MODEL_INFO thresholds_sha256 "
        "(exact validated artifact; do not rewrite metadata after hashing)"
    )

    expected_labels = info.get("labels_sha256")
    assert isinstance(expected_labels, str) and len(expected_labels) == 64, (
        "MODEL_INFO.json must record labels_sha256 from promotion validation"
    )
    assert labels_path.exists(), "Production stem-bound labels artifact is required"
    assert _sha256(labels_path) == expected_labels, (
        "Production labels bytes must match MODEL_INFO labels_sha256"
    )

    # Filename stem pairs the pickle to companions; metadata.model may still
    # name the experimental candidate used during calibration.
    assert thresholds_path.name == f"{model_path.stem}_thresholds.json"
    assert labels_path.name == f"{model_path.stem}_labels.json"


def test_find_production_thresholds_prefers_active_stem_over_newer_orphan(tmp_path: Path) -> None:
    """Model Info must not bind to a newer orphan thresholds file by mtime."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    active = model_dir / "disaster_lr_v26-09-21_prod_2026-09-21.pkl"
    active.write_bytes(b"active-model")
    active_thresholds = model_dir / f"{active.stem}_thresholds.json"
    active_thresholds.write_text(
        json.dumps(
            {
                "metadata": {
                    "model": "experiments/experimental_runs/2026-09-21/lr_vocab15k_cal_split_model.pkl"
                },
                "thresholds": {"water": 0.2},
            }
        ),
        encoding="utf-8",
    )

    time.sleep(0.05)
    orphan_thresholds = model_dir / "disaster_lr_v25-11-06_prod_2025-11-06_thresholds.json"
    orphan_thresholds.write_text(
        json.dumps(
            {
                "metadata": {"model": "model/disaster_lr_v25-11-06_prod_2025-11-06.pkl"},
                "thresholds": {"water": 0.9},
            }
        ),
        encoding="utf-8",
    )
    # Ensure orphan is newer by mtime
    orphan_thresholds.touch()

    app = Flask(__name__)
    app.config["MODEL_PATH"] = active

    with app.app_context():
        resolved = _resolve_active_production_model_path(model_dir)
        assert resolved == active
        found = _find_production_thresholds_file(model_dir)
        assert found == active_thresholds
        found_explicit = _find_production_thresholds_file(model_dir, model_stem=active.stem)
        assert found_explicit == active_thresholds


def _category_stats_for_bundle(
    thresholds_map: dict[str, float],
    *,
    water_precision: float = 0.91,
    water_recall: float = 0.8,
    water_f1: float = 0.85,
    water_support: float = 100.0,
    include_water_critical: bool = True,
) -> list[dict]:
    """Minimal category_stats aligned to the inference threshold map."""
    stats = []
    for label, threshold in thresholds_map.items():
        if label == "water" and include_water_critical:
            stats.append(
                {
                    "category": "water",
                    "type": "critical",
                    "threshold": threshold,
                    "actual_recall": water_recall,
                    "precision": water_precision,
                    "f1": water_f1,
                    "support": water_support,
                }
            )
        else:
            stats.append(
                {
                    "category": label,
                    "type": "non-critical",
                    "threshold": threshold,
                    "actual_recall": 0.5,
                    "precision": 0.5,
                    "f1": 0.5,
                    "support": 0.0,
                }
            )
    return stats


def _write_stem_bound_production_bundle(
    model_path: Path,
    *,
    thresholds_extra: dict | None = None,
    model_info_fields: dict | None = None,
    category_stats: list | None = None,
    include_category_stats: bool = True,
) -> dict:
    """Write a resolver-valid stem-bound production bundle for dashboard tests."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.is_file():
        model_path.write_bytes(b"test-model")

    thresholds_map = {label: 0.5 for label in TARGET_COLUMNS}
    if thresholds_extra:
        thresholds_map.update(thresholds_extra)

    if include_category_stats:
        if category_stats is None:
            category_stats = _category_stats_for_bundle(thresholds_map)
    else:
        category_stats = None

    thresholds_path = model_path.with_name(f"{model_path.stem}_thresholds.json")
    labels_path = model_path.with_name(f"{model_path.stem}_labels.json")
    thresholds_payload: dict = {
        "metadata": {
            "model": "experiments/test/model.pkl",
            "eval_critical_recall": 0.6149,
        },
        "critical_only": {"water": thresholds_map["water"]},
        "thresholds": thresholds_map,
        "performance": {
            "optimized": {
                "f1_weighted": 0.8975,
                "critical_recall": 0.6149,
            }
        },
    }
    if category_stats is not None:
        thresholds_payload["category_stats"] = category_stats
    thresholds_path.write_text(json.dumps(thresholds_payload), encoding="utf-8")
    labels_path.write_text(json.dumps(list(TARGET_COLUMNS)), encoding="utf-8")

    info = {
        "version": "v26-09-21",
        "status": "production",
        "algorithm": "lr",
        "algorithm_name": "LogisticRegression",
        "performance": {
            "optimized_f1_weighted": 0.8975,
            "eval_critical_recall": 0.6149,
        },
    }
    if model_info_fields:
        info.update(model_info_fields)
    info["sha256"] = _sha256(model_path)
    info["thresholds_sha256"] = _sha256(thresholds_path)
    info["labels_sha256"] = _sha256(labels_path)
    (model_path.parent / "MODEL_INFO.json").write_text(json.dumps(info), encoding="utf-8")
    return thresholds_payload


def test_model_info_dashboard_binds_active_stem_despite_newer_orphan(
    client, app, tmp_path: Path
) -> None:
    """GET /api/model-info/dashboard must use v26 companions, not newer-mtime v25 orphans."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    active = model_dir / "disaster_lr_v26-09-21_prod_2026-09-21.pkl"
    active.write_bytes(b"active-v26-model")
    _write_stem_bound_production_bundle(
        active,
        thresholds_extra={"water": 0.25},
        category_stats=_category_stats_for_bundle(
            {**{label: 0.5 for label in TARGET_COLUMNS}, "water": 0.25},
            water_precision=0.91,
        ),
    )

    time.sleep(0.05)
    orphan_stem = "disaster_lr_v25-11-06_prod_2025-11-06"
    orphan_thresholds = model_dir / f"{orphan_stem}_thresholds.json"
    orphan_thresholds.write_text(
        json.dumps(
            {
                "metadata": {"model": f"model/{orphan_stem}.pkl"},
                "critical_only": {"water": 0.99},
                "thresholds": {"water": 0.99, "related": 0.1},
                "category_stats": [
                    {
                        "category": "water",
                        "type": "critical",
                        "threshold": 0.99,
                        "actual_recall": 0.1,
                        "precision": 0.11,
                        "f1": 0.1,
                        "support": 100.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    orphan_thresholds.touch()

    previous_model_path = app.config.get("MODEL_PATH")
    try:
        app.config["MODEL_PATH"] = active
        with patch("app.routes.api._get_model_dir", return_value=model_dir):
            response = client.get("/api/model-info/dashboard")
    finally:
        app.config["MODEL_PATH"] = previous_model_path

    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None

    assert payload["model"]["version"] == "v26-09-21"
    assert "V26_09_21" in payload["model"]["id"]
    assert "V25_11_06" not in payload["model"]["id"]

    water_critical = [
        row for row in payload["criticalThresholds"] if row.get("key") == "water"
    ]
    assert len(water_critical) == 1
    assert water_critical[0]["threshold"] == pytest.approx(0.25)
    assert water_critical[0]["threshold"] != pytest.approx(0.99)

    water_category = [row for row in payload["categories"] if row.get("key") == "water"]
    assert len(water_category) == 1
    assert water_category[0]["precision"] == pytest.approx(0.91)
    assert payload["metrics"]["precision"] == pytest.approx(0.91)


def test_model_info_dashboard_uses_category_stats_not_csv(
    client, app, tmp_path: Path
) -> None:
    """Per-category metrics come from thresholds category_stats, ignoring CSV."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    active = model_dir / "disaster_lr_v26-09-21_prod_2026-09-21.pkl"
    active.write_bytes(b"active-model")
    thresholds_map = {label: 0.5 for label in TARGET_COLUMNS}
    thresholds_map["water"] = 0.33
    stats = [
        {
            "category": "water",
            "type": "critical",
            "threshold": 0.33,
            "actual_recall": 0.77,
            "precision": 0.66,
            "f1": 0.71,
            "support": 50.0,
        }
    ]
    _write_stem_bound_production_bundle(
        active,
        thresholds_extra={"water": 0.33},
        category_stats=stats,
    )
    # Misleading CSV must not be consulted for dashboard categories / P/R.
    (model_dir / f"{active.stem}_performance_metrics.csv").write_text(
        "category,output_class,precision,recall,f1-score,support\n"
        "water,weighted avg,0.11,0.11,0.11,50\n",
        encoding="utf-8",
    )

    previous_model_path = app.config.get("MODEL_PATH")
    try:
        app.config["MODEL_PATH"] = active
        with patch("app.routes.api._get_model_dir", return_value=model_dir):
            response = client.get("/api/model-info/dashboard")
    finally:
        app.config["MODEL_PATH"] = previous_model_path

    assert response.status_code == 200
    payload = response.get_json()
    water = [row for row in payload["categories"] if row["key"] == "water"]
    assert len(water) == 1
    assert water[0]["precision"] == pytest.approx(0.66)
    assert water[0]["recall"] == pytest.approx(0.77)
    assert water[0]["f1"] == pytest.approx(0.71)
    assert water[0]["support"] == 50
    assert payload["metrics"]["precision"] == pytest.approx(0.66)
    assert payload["metrics"]["recall"] == pytest.approx(0.77)
    critical = [row for row in payload["criticalThresholds"] if row["key"] == "water"]
    assert len(critical) == 1
    assert critical[0]["threshold"] == pytest.approx(0.33)


def test_critical_thresholds_use_inference_map_not_stat_threshold(
    client, app, tmp_path: Path
) -> None:
    """criticalThresholds[].threshold comes from production_artifacts.thresholds."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    active = model_dir / "disaster_lr_v26-09-21_prod_2026-09-21.pkl"
    active.write_bytes(b"active-model")
    stats = [
        {
            "category": "water",
            "type": "critical",
            "threshold": 0.99,  # intentionally wrong vs inference map
            "actual_recall": 0.8,
            "precision": 0.7,
            "f1": 0.75,
            "support": 10.0,
        }
    ]
    _write_stem_bound_production_bundle(
        active,
        thresholds_extra={"water": 0.42},
        category_stats=stats,
    )

    previous_model_path = app.config.get("MODEL_PATH")
    try:
        app.config["MODEL_PATH"] = active
        with patch("app.routes.api._get_model_dir", return_value=model_dir):
            response = client.get("/api/model-info/dashboard")
    finally:
        app.config["MODEL_PATH"] = previous_model_path

    payload = response.get_json()
    critical = [row for row in payload["criticalThresholds"] if row["key"] == "water"]
    assert len(critical) == 1
    assert critical[0]["threshold"] == pytest.approx(0.42)
    assert critical[0]["threshold"] != pytest.approx(0.99)


def test_critical_thresholds_skip_keys_missing_from_inference_map(
    client, app, tmp_path: Path
) -> None:
    """Do not publish threshold 0.0 when a critical stats key is absent from the map."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    active = model_dir / "disaster_lr_v26-09-21_prod_2026-09-21.pkl"
    active.write_bytes(b"active-model")
    stats = [
        {
            "category": "water",
            "type": "critical",
            "threshold": 0.42,
            "actual_recall": 0.8,
            "precision": 0.7,
            "f1": 0.75,
            "support": 10.0,
        },
        {
            "category": "not_in_inference_map",
            "type": "critical",
            "threshold": 0.99,
            "actual_recall": 0.5,
            "precision": 0.5,
            "f1": 0.5,
            "support": 1.0,
        },
    ]
    _write_stem_bound_production_bundle(
        active,
        thresholds_extra={"water": 0.42},
        category_stats=stats,
    )

    previous_model_path = app.config.get("MODEL_PATH")
    try:
        app.config["MODEL_PATH"] = active
        with patch("app.routes.api._get_model_dir", return_value=model_dir):
            response = client.get("/api/model-info/dashboard")
    finally:
        app.config["MODEL_PATH"] = previous_model_path

    payload = response.get_json()
    critical_by_key = {row["key"]: row for row in payload["criticalThresholds"]}
    assert "water" in critical_by_key
    assert critical_by_key["water"]["threshold"] == pytest.approx(0.42)
    assert "not_in_inference_map" not in critical_by_key


def test_dashboard_null_precision_recall_when_category_stats_missing(
    client, app, tmp_path: Path
) -> None:
    """Missing category_stats yields null P/R and empty lists, not fabricated zeros."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    active = model_dir / "disaster_lr_v26-09-21_prod_2026-09-21.pkl"
    active.write_bytes(b"active-model")
    _write_stem_bound_production_bundle(
        active,
        include_category_stats=False,
        model_info_fields={
            "performance": {
                "optimized_f1_weighted": 0.8975,
                "eval_critical_recall": 0.6149,
            }
        },
    )

    previous_model_path = app.config.get("MODEL_PATH")
    try:
        app.config["MODEL_PATH"] = active
        with patch("app.routes.api._get_model_dir", return_value=model_dir):
            response = client.get("/api/model-info/dashboard")
    finally:
        app.config["MODEL_PATH"] = previous_model_path

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["metrics"]["f1"] == pytest.approx(0.8975)
    assert payload["metrics"]["precision"] is None
    assert payload["metrics"]["recall"] is None
    assert payload["metrics"]["evalCriticalRecall"] == pytest.approx(0.6149)
    assert payload["categories"] == []
    assert payload["criticalThresholds"] == []


def test_checked_in_operating_point_cross_artifact_invariants() -> None:
    """MODEL_INFO OP fields must agree with thresholds performance / metadata."""
    model_path = Config.MODEL_PATH
    thresholds_path = model_path.with_name(f"{model_path.stem}_thresholds.json")
    model_info_path = model_path.with_name("MODEL_INFO.json")
    if not model_path.exists() or not thresholds_path.exists() or not model_info_path.exists():
        pytest.skip("Production model/thresholds/MODEL_INFO not available")

    model_info = json.loads(model_info_path.read_text(encoding="utf-8"))
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    inference_map = thresholds.get("thresholds") or {}
    category_stats = thresholds.get("category_stats") or []
    assert isinstance(category_stats, list) and category_stats, (
        "checked-in thresholds must include category_stats for Model Information"
    )

    for stat in category_stats:
        assert isinstance(stat, dict)
        category = stat["category"]
        assert category in inference_map
        assert float(stat["threshold"]) == pytest.approx(float(inference_map[category]))

    optimized_f1 = (model_info.get("performance") or {}).get("optimized_f1_weighted")
    if optimized_f1 is None:
        optimized_f1 = (model_info.get("validation_results") or {}).get(
            "optimized_f1_weighted"
        )
    thresholds_f1 = (
        (thresholds.get("performance") or {}).get("optimized") or {}
    ).get("f1_weighted")
    assert optimized_f1 is not None and thresholds_f1 is not None
    assert float(optimized_f1) == pytest.approx(float(thresholds_f1))

    eval_critical = (model_info.get("performance") or {}).get("eval_critical_recall")
    if eval_critical is None:
        eval_critical = (model_info.get("validation_results") or {}).get(
            "eval_critical_recall"
        )
    meta_critical = (thresholds.get("metadata") or {}).get("eval_critical_recall")
    perf_critical = (
        (thresholds.get("performance") or {}).get("optimized") or {}
    ).get("critical_recall")
    assert eval_critical is not None
    assert meta_critical is not None
    assert perf_critical is not None
    assert float(eval_critical) == pytest.approx(float(meta_critical))
    assert float(eval_critical) == pytest.approx(float(perf_critical))


def test_dashboard_categories_match_checked_in_category_stats(client) -> None:
    """Live dashboard categories / criticalThresholds trace to the OP bundle."""
    model_path = Config.MODEL_PATH
    thresholds_path = model_path.with_name(f"{model_path.stem}_thresholds.json")
    if not model_path.exists() or not thresholds_path.exists():
        pytest.skip("Production model/thresholds not available")

    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    category_stats = thresholds.get("category_stats") or []
    inference_map = thresholds.get("thresholds") or {}
    if not category_stats:
        pytest.skip("checked-in thresholds lack category_stats")

    response = client.get("/api/model-info/dashboard")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["model"]["status"] == "production"

    by_key = {row["key"]: row for row in payload["categories"]}
    assert len(by_key) == len(category_stats)
    for stat in category_stats:
        key = stat["category"]
        assert key in by_key
        expected_recall = (
            stat["actual_recall"] if "actual_recall" in stat else stat["recall"]
        )
        assert by_key[key]["precision"] == pytest.approx(float(stat["precision"]))
        assert by_key[key]["recall"] == pytest.approx(float(expected_recall))
        assert by_key[key]["f1"] == pytest.approx(float(stat["f1"]))
        assert by_key[key]["support"] == int(float(stat["support"]))

    critical_expected = [
        stat["category"] for stat in category_stats if stat.get("type") == "critical"
    ]
    critical_actual = {row["key"]: row["threshold"] for row in payload["criticalThresholds"]}
    assert set(critical_actual) == set(critical_expected)
    for key in critical_expected:
        assert critical_actual[key] == pytest.approx(float(inference_map[key]))
