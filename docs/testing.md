# Testing Guide

This repository ships with a curated pytest suite designed to reassure reviewers that the Flask + ML stack is production ready. The tests intentionally mix fast smoke coverage with targeted integration, security, and deployment checks.

## Layout

| Module | Purpose |
| --- | --- |
| `tests/test_smoke.py` | Core Flask smoke coverage for `/` and `/go` |
| `tests/test_app_smoke.py` | Happy-path prediction flow with the test configuration |
| `tests/test_csrf_smoke.py` | CSRF token capture and guarded POST cycle |
| `tests/test_flask_standardized.py` | Production configuration and model wiring validation |
| `tests/test_perf.py` | Reload performance guardrail for the production model |
| `tests/test_gdrive_deployment.py` | Google Drive download contract (fully mocked) |
| `tests/test_security.py` | Hardened subprocess and filename validation |
| `tests/test_compare_models_paths.py` | Experiment artifact discovery fallbacks |
| `tests/test_optimization.py` | NLTK setup and performance diagnostics guards |

## Running the suite

The default configuration (via `pytest.ini`) skips Google Drive and performance marks so the core checks finish in under a second when the lightweight model is present.

```bash
pytest -q                                   # default run (excludes gdrive/perf)
pytest -q -m "not gdrive and not perf and not slow"   # leanest loop when iterating
pytest -q -m perf                           # performance SLA verification
pytest -q -m gdrive                         # download helper contract (mocked)
pytest -q -m "gdrive or perf"               # full extended validation
```

## Markers and skips

- **`gdrive`** – Exercises the Google Drive download logic. All network calls are mocked; use when touching `ModelService` download code.
- **`perf`** – Time-sensitive reload checks that require a local production model file.
- **`integration`** – Flask-factory or multi-service flows.
- **`security`** – Hardened validation around subprocess boundaries.
- **`slow`** – Tests that need the production artifact or perform multi-step form flows. Combine with `-m "not slow"` for the absolute quickest run.

Tests that truly need the production pickle call `skip_if_no_model(...)` which aborts early with a clear reason if the artifact is missing. This keeps CI deterministic while still broadcasting the requirement.

## Test environment variables

| Variable | Purpose | When to set |
| --- | --- | --- |
| `GDRIVE_MODEL_ID` | Points the Google Drive download helper at a real artifact. Placeholder values keep the real-download test skipped. | Set when running `tests/test_gdrive_deployment.py::test_gdrive_integration_with_real_id` or when you want the perf suite to hydrate the model via Drive instead of relying on the local pickle. |

All other suites run hermetically. If you need to surface additional inputs, document them alongside the relevant tests so contributors know how to enable the path.

## Debugging tips

- Add `-vv` for verbose assertion messages when diagnosing a failure.
- Pair `--maxfail=1` with markers to quickly reproduce a flaky case.
- Use `pytest --disable-warnings` when triaging to focus on assertion output.
- Export `GDRIVE_MODEL_ID` if you want to exercise the optional real download path.
- When working on Flask routes, run `pytest tests/test_smoke.py -x` to validate only the main loop.

## Extending the tests

1. Prefer reusing `create_test_app` and the shared `client` fixture rather than instantiating Flask manually.
2. Mark any test that touches slow or optional resources (`gdrive`, `perf`, `slow`) so CI can opt-in explicitly.
3. When a test requires the production model, call `skip_if_no_model(Config)` for a consistent skip reason.
4. Keep assertion messages human-readable—most reviewers scan them directly in CI logs.
5. Use temporary paths (`tmp_path`) and mocks for any filesystem or network interaction.

## CI recipe

A minimal GitHub Actions step after installing dependencies:

```yaml
- name: Run unit tests
  run: |
    pytest -q
- name: Run security + integration spotlight
  run: |
    pytest -q -m "integration or security" --maxfail=1 --disable-warnings
- name: Optional extended checks
  if: github.event_name == 'schedule'
  run: |
    pytest -q -m "gdrive or perf"
```

Tailor the final step to your deployment needs; the suite was designed so the first command is safe for every push.

### Expected counts

As of this revision, `pytest -q` (with the default marker filter) reports **111 passed, 7 skipped, 15 deselected** in roughly five seconds on a typical developer laptop. If the totals drop unexpectedly, confirm new tests are marked correctly rather than silently excluded.
