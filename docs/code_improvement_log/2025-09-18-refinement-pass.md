# 2025-09-18 Refinement Pass

## Changes by File
- `app/forms.py`: Fixed malformed module docstring to unblock imports and applied consistent Black formatting.
- `app/services.py`: Swapped inline metrics CSV loader for shared utility usage and pulled metrics paths from central config.
- `pyproject.toml`: Added Ruff lint table to mirror the new tool configuration.
- `requirements.txt`: Rewrote with pinned versions and ensured trailing newline integrity.
- `src/disasterproject/utils/__init__.py`: Exported the new `read_metrics_csv` helper for package consumers.
- `src/disasterproject/utils/config.py`: Introduced explicit `Path`-based project constants for metrics and models.
- `src/disasterproject/utils/metrics_io.py`: Added reusable loader with defensive parsing + logging.
- `tests/test_metrics_io.py`: Added coverage for success, missing-file, and parse-error paths.
- `tests/test_request_logging_utils.py`: Added request-context coverage for `format_request_context` helper.

## New Tests
- `tests/test_metrics_io.py`
- `tests/test_request_logging_utils.py`

## Commands Run
- `black src/disasterproject/utils/config.py src/disasterproject/utils/metrics_io.py tests/test_metrics_io.py tests/test_request_logging_utils.py app/forms.py` (pass)
- `ruff check src/disasterproject/utils/metrics_io.py tests/test_metrics_io.py tests/test_request_logging_utils.py` (pass)
- `PYTHONPATH=src pytest tests/test_metrics_io.py tests/test_request_logging_utils.py -q` (pass)
- `PYTHONPATH=src pytest -q` (fails: baseline suite expects production model + legacy mocks)
- `PYTHONPATH=src python run.py --help` (fails: runtime exits when model artifacts are absent)

## Next Follow-Ups
1. Backfill pytest fixtures or skip decorators so full suite can execute without production model artifacts.
2. Introduce targeted tests for the new `DataServiceError`/`ModelServiceError` surfaces to lock in response semantics.
