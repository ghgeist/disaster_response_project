"""Tests for scripts.compare_models helper utilities."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_compare_models() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "04_evaluation" / "compare_models.py"
    spec = importlib.util.spec_from_file_location("compare_models_for_tests", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load compare_models module for testing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def compare_models_module() -> ModuleType:
    return _load_compare_models()


def test_find_experiment_artifacts_prefers_latest_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compare_models_module: ModuleType) -> None:
    runs_dir = tmp_path / "experiments" / "experimental_runs"
    older_run = runs_dir / "2025-09-15"
    newer_run = runs_dir / "2025-09-16"
    older_run.mkdir(parents=True)
    newer_run.mkdir(parents=True)

    (older_run / "performance_metrics.csv").write_text("col\n", encoding="utf-8")
    (newer_run / "performance_metrics.csv").write_text("col\n", encoding="utf-8")
    (newer_run / "MODEL_INFO.json").write_text("{}", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    artifacts = compare_models_module.find_experiment_artifacts()
    assert artifacts is not None
    assert Path(artifacts["metrics_path"]).resolve() == (newer_run / "performance_metrics.csv").resolve()
    assert artifacts["display_name"].endswith("2025-09-16")


def test_find_experiment_artifacts_legacy_results_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compare_models_module: ModuleType) -> None:
    """
    Test that legacy results path is no longer supported.
    
    Note: Legacy support for experiments/results/ was removed in favor of
    the new experiments/experimental_runs/<date>/ structure. This test
    verifies that legacy paths return None (expected behavior).
    """
    results_dir = tmp_path / "experiments" / "results"
    results_dir.mkdir(parents=True)
    metrics_path = results_dir / "performance_metrics.csv"
    metrics_path.write_text("col\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    artifacts = compare_models_module.find_experiment_artifacts()
    # Legacy paths are no longer supported - should return None
    assert artifacts is None
