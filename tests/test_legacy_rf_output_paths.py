"""Guardrails for legacy RF training output paths."""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/02_training/04_create_production_model.py'


def _load_legacy_rf_script():
    spec = importlib.util.spec_from_file_location('legacy_rf_script', _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def legacy_rf_script():
    return _load_legacy_rf_script()


def test_default_output_is_under_experiments_legacy_rf(legacy_rf_script):
    path = legacy_rf_script.default_legacy_rf_output_path('2026-09-22')
    assert path.startswith('experiments/legacy_rf/')
    assert path.endswith('disaster_rf_legacy.pkl')


def test_model_dir_write_blocked_without_flag(legacy_rf_script):
    with pytest.raises(ValueError, match='Refusing to write'):
        legacy_rf_script.validate_legacy_rf_output_path(
            'model/disaster_rf_legacy.pkl',
            allow_model_dir=False,
        )


def test_model_dir_write_allowed_with_flag(legacy_rf_script):
    legacy_rf_script.validate_legacy_rf_output_path(
        'model/disaster_rf_legacy.pkl',
        allow_model_dir=True,
    )


def test_legacy_rf_path_not_treated_as_model_dir(legacy_rf_script):
    assert not legacy_rf_script.path_targets_model_dir(
        'experiments/legacy_rf/2026-09-22/disaster_rf_legacy.pkl'
    )
