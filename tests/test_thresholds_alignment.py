"""Threshold provenance and Model Information companion discovery contracts."""

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
    _discover_production_metrics_file,
    _find_production_thresholds_file,
    _resolve_active_production_model_path,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_production_thresholds_sha_matches_model_info_provenance() -> None:
    """Deployed thresholds must stay byte-identical to the validated evidence SHA."""
    model_path = Config.MODEL_PATH
    thresholds_path = model_path.with_name(f"{model_path.stem}_thresholds.json")
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

    # Filename stem pairs the pickle to companions; metadata.model may still
    # name the experimental candidate used during calibration.
    assert thresholds_path.name == f"{model_path.stem}_thresholds.json"


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


def test_discover_production_metrics_prefers_active_stem_over_orphan(tmp_path: Path) -> None:
    """Metrics discovery follows the active production stem, not orphan companions."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    active = model_dir / "disaster_lr_v26-09-21_prod_2026-09-21.pkl"
    active.write_bytes(b"active-model")
    active_metrics = model_dir / f"{active.stem}_performance_metrics.csv"
    active_metrics.write_text("category,output_class,precision\nrelated,1,0.9\n", encoding="utf-8")

    time.sleep(0.05)
    orphan_metrics = model_dir / "disaster_lr_v25-11-06_prod_2025-11-06_performance_metrics.csv"
    orphan_metrics.write_text("category,output_class,precision\nrelated,1,0.1\n", encoding="utf-8")
    orphan_metrics.touch()

    app = Flask(__name__)
    app.config["MODEL_PATH"] = active

    with app.app_context():
        found = _discover_production_metrics_file(model_dir)
        assert found == active_metrics


def _write_metrics_csv(path: Path, *, precision: float, support: float = 100.0) -> None:
    path.write_text(
        (
            "category,output_class,precision,recall,f1-score,support\n"
            f"water,1,0.5,0.5,0.5,{support}\n"
            f"water,weighted avg,{precision},0.8,0.85,{support}\n"
        ),
        encoding="utf-8",
    )


def test_model_info_dashboard_binds_active_stem_despite_newer_orphan(
    client, app, tmp_path: Path
) -> None:
    """GET /api/model-info/dashboard must use v26 companions, not newer-mtime v25 orphans."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    active = model_dir / "disaster_lr_v26-09-21_prod_2026-09-21.pkl"
    active.write_bytes(b"active-v26-model")
    (model_dir / f"{active.stem}_thresholds.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "model": (
                        "experiments/experimental_runs/2026-09-21/"
                        "lr_vocab15k_cal_split_model.pkl"
                    )
                },
                "critical_only": {"water": 0.25},
                "thresholds": {"water": 0.25, "related": 0.5},
            }
        ),
        encoding="utf-8",
    )
    _write_metrics_csv(
        model_dir / f"{active.stem}_performance_metrics.csv", precision=0.91
    )
    (model_dir / "MODEL_INFO.json").write_text(
        json.dumps(
            {
                "version": "v26-09-21",
                "status": "production",
                "algorithm": "lr",
                "algorithm_name": "LogisticRegression",
                "performance": {"f1_weighted": 0.8975},
            }
        ),
        encoding="utf-8",
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
            }
        ),
        encoding="utf-8",
    )
    orphan_thresholds.touch()
    orphan_metrics = model_dir / f"{orphan_stem}_performance_metrics.csv"
    _write_metrics_csv(orphan_metrics, precision=0.11)
    orphan_metrics.touch()

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
