"""Guardrails for legacy RF training output paths."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/02_training/04_create_production_model.py'
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PARAMS = str(_REPO_ROOT / 'experiments/model_candidates/vocab_15k.json')
_CLASS_WEIGHTS = str(_REPO_ROOT / 'experiments/model_candidates/class_weights.json')


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
    path = legacy_rf_script.default_legacy_rf_output_path('2026-09-22', '153045')
    assert path == 'experiments/legacy_rf/2026-09-22/153045/disaster_rf_legacy.pkl'


def test_model_dir_write_blocked_without_flag(legacy_rf_script):
    with pytest.raises(ValueError, match='Refusing to write'):
        legacy_rf_script.validate_legacy_rf_output_path(
            'model/disaster_rf_legacy.pkl',
            allow_model_dir=False,
            repo_root=_REPO_ROOT,
        )


def test_absolute_model_path_blocked(legacy_rf_script, tmp_path):
    repo = tmp_path
    (repo / 'model').mkdir()
    output = str(repo / 'model' / 'disaster_rf_legacy.pkl')
    with pytest.raises(ValueError, match='Refusing to write'):
        legacy_rf_script.validate_legacy_rf_output_path(
            output,
            allow_model_dir=False,
            repo_root=repo,
        )


def test_model_dir_write_allowed_with_flag(legacy_rf_script):
    legacy_rf_script.validate_legacy_rf_output_path(
        'model/disaster_rf_legacy.pkl',
        allow_model_dir=True,
        repo_root=_REPO_ROOT,
    )


def test_relative_legacy_path_resolves_via_cwd(legacy_rf_script, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    relative = 'experiments/legacy_rf/test/run.pkl'
    resolved = legacy_rf_script.resolve_output_path(relative)
    assert resolved == (tmp_path / relative).resolve()
    assert not legacy_rf_script.resolves_under_production_model_dir(
        resolved,
        repo_root=_REPO_ROOT,
    )


def test_production_model_directory_itself_is_blocked(legacy_rf_script):
    with pytest.raises(ValueError, match='not a directory|Refusing to write'):
        legacy_rf_script.validate_legacy_rf_output_path(
            'model/',
            allow_model_dir=False,
            repo_root=_REPO_ROOT,
            cli_path='model/',
        )


def test_legacy_rf_path_not_under_production_model_dir(legacy_rf_script):
    assert not legacy_rf_script.resolves_under_production_model_dir(
        'experiments/legacy_rf/2026-09-22/153045/disaster_rf_legacy.pkl',
        repo_root=_REPO_ROOT,
    )


def test_symlink_into_model_dir_is_blocked(legacy_rf_script, tmp_path):
    repo = tmp_path
    real_model = repo / 'model'
    real_model.mkdir()
    link_parent = repo / 'experiments' / 'legacy_rf'
    link_parent.mkdir(parents=True)
    symlink_target = link_parent / 'linked.pkl'
    symlink_target.symlink_to(real_model / 'via_symlink.pkl')
    with pytest.raises(ValueError, match='Refusing to write'):
        legacy_rf_script.validate_legacy_rf_output_path(
            str(symlink_target),
            allow_model_dir=False,
            repo_root=repo,
        )


def test_cli_refuses_model_output_without_override():
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            '--output',
            'model/disaster_rf_legacy.pkl',
            '--params',
            _PARAMS,
            '--class-weights',
            _CLASS_WEIGHTS,
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert 'Refusing to write RandomForest artifacts into model/' in result.stderr


def test_cli_cwd_relative_escape_into_model_dir():
    """../../model/... from scripts/02_training must use CWD semantics and refuse."""
    nested = _REPO_ROOT / 'scripts' / '02_training'
    target = _REPO_ROOT / 'model' / 'disaster_rf_legacy_cwd_escape.pkl'
    sidecar = _REPO_ROOT / 'model' / 'disaster_rf_legacy_cwd_escape_performance_metrics.csv'
    for path in (target, sidecar):
        if path.exists():
            path.unlink()

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            '--output',
            '../../model/disaster_rf_legacy_cwd_escape.pkl',
            '--params',
            _PARAMS,
            '--class-weights',
            _CLASS_WEIGHTS,
        ],
        cwd=nested,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert 'Refusing to write RandomForest artifacts into model/' in result.stderr
    assert not target.exists()
    assert not sidecar.exists()


def test_cli_refuses_model_directory_output_without_override():
    """--output model/ must fail before any writes into production model/."""
    model_dir = _REPO_ROOT / 'model'
    before = {p.name for p in model_dir.iterdir()} if model_dir.is_dir() else set()

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            '--output',
            'model/',
            '--params',
            _PARAMS,
            '--class-weights',
            _CLASS_WEIGHTS,
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert (
        'not a directory' in result.stderr
        or 'Refusing to write RandomForest artifacts into model/' in result.stderr
    )
    after = {p.name for p in model_dir.iterdir()} if model_dir.is_dir() else set()
    assert after == before
