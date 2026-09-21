# Testing Guide

## Index
- Performance testing: `docs/testing/performance.md`
- Google Drive testing (historical): `docs/testing/gdrive.md`
- Cursor Web UI: `docs/testing/cursor-web-ui.md`
- FAQ: `docs/testing/faq.md`

This repository ships with a curated pytest suite designed to reassure reviewers that the Flask + ML stack is production ready. The tests intentionally mix fast smoke coverage with targeted integration, security, and deployment checks.

## Layout

| Module | Purpose |
| --- | --- |
| `tests/test_smoke.py` | Core Flask smoke coverage for `/` and `/go` |
| `tests/test_app_smoke.py` | Happy-path prediction flow with the test configuration |
| `tests/test_csrf_smoke.py` | CSRF token capture and guarded POST cycle |
| `tests/test_flask_standardized.py` | Production configuration and model wiring validation |
| `tests/test_perf.py` | Reload performance guardrail for the production model |
| `tests/test_thresholds_alignment.py` | Stem-bound companion discovery + thresholds SHA provenance |
| `tests/test_security.py` | Hardened subprocess and filename validation |
| `tests/test_compare_models_paths.py` | Experiment artifact discovery fallbacks |
| `tests/test_optimization.py` | NLTK setup and performance diagnostics guards |

> **Historical note:** `tests/test_gdrive_deployment.py` and the `gdrive` marker documented a Google Drive download helper that is **no longer part of the Flask runtime**. Production models load from the local `model/` directory (git-tracked `*_prod_*.pkl`). Keep `docs/testing/gdrive.md` only as archival contract notes if resurrecting offline helpers.

## Running the suite

The default configuration (via `pytest.ini`) skips performance marks so the core checks stay fast when the lightweight model is present.

### Recommended: Use the test runner script

For maximum portability across environments (local, Replit, Cursor Web UI, CI/CD), use the test runner script:

```bash
python scripts/run_tests.py -q                                   # default run (excludes perf)
python scripts/run_tests.py -q -m "not perf and not slow"        # leanest loop when iterating
python scripts/run_tests.py -q -m perf                           # performance SLA verification
```

The test runner automatically detects the best available pytest command (`pytest`, `python3 -m pytest`, or `python -m pytest`) based on your environment.

### Direct pytest invocation

If pytest is available in your PATH, you can also run pytest directly:

```bash
pytest -q                                   # default run (excludes perf)
pytest -q -m "not perf and not slow"        # leanest loop when iterating
pytest -q -m perf                           # performance SLA verification
```

## Markers and skips

- **`perf`** – Time-sensitive reload checks that require a local production model file.
- **`integration`** – Flask-factory or multi-service flows.
- **`security`** – Hardened validation around subprocess boundaries.
- **`slow`** – Tests that need the production artifact or perform multi-step form flows. Combine with `-m "not slow"` for the absolute quickest run.
- **`gdrive`** (historical) – Formerly exercised Drive download helpers. Not required for current local-artifact deployments.

Tests that truly need the production pickle call `skip_if_no_model(...)` which aborts early with a clear reason if the artifact is missing. This keeps CI deterministic while still broadcasting the requirement.

## Test environment variables

| Variable | Purpose | When to set |
| --- | --- | --- |
| `MODEL_FILENAME` | Override production-model auto-discovery with an explicit `model/*.pkl` name | Rare; use when comparing two local production candidates |
| `GDRIVE_MODEL_ID` | **Historical only.** Ignored by the current `ModelLoader` (no runtime Drive download) | Do not set for new work |

All other suites run hermetically. If you need to surface additional inputs, document them alongside the relevant tests so contributors know how to enable the path.

## Debugging tips

- Add `-vv` for verbose assertion messages when diagnosing a failure.
- Pair `--maxfail=1` with markers to quickly reproduce a flaky case.
- Use `pytest --disable-warnings` when triaging to focus on assertion output.
- When working on Flask routes, run `pytest tests/test_smoke.py -x` to validate only the main loop.
- When touching promotion or Model Information companions, run `pytest tests/test_thresholds_alignment.py -q`.

## Extending the tests

1. Prefer reusing `create_test_app` and the shared `client` fixture rather than instantiating Flask manually.
2. Mark any test that touches slow or optional resources (`perf`, `slow`) so CI can opt-in explicitly.
3. When a test requires the production model, call `skip_if_no_model(Config)` for a consistent skip reason.
4. Keep assertion messages human-readable—most reviewers scan them directly in CI logs.
5. Use temporary paths (`tmp_path`) and mocks for any filesystem or network interaction.
6. Do **not** rewrite production thresholds JSON after promotion validation; assert SHA identity against `MODEL_INFO.thresholds_sha256` instead.

## CI recipe

A minimal GitHub Actions step after installing dependencies:

```yaml
- name: Install dependencies
  run: |
    python -m pip install -r requirements.txt -r requirements-dev.txt
    python -m pip install -e .
    python -c "import sklearn, imblearn; print(sklearn.__version__, imblearn.__version__)"
    python -m pip check
- name: Run unit tests
  run: |
    python scripts/run_tests.py -q
- name: Run security + integration spotlight
  run: |
    python scripts/run_tests.py -q -m "integration or security" --maxfail=1 --disable-warnings
- name: Optional performance checks
  if: github.event_name == 'schedule'
  run: |
    python scripts/run_tests.py -q -m perf
```

Tailor the final step to your deployment needs; the suite was designed so the first command is safe for every push.

### Expected counts

As of 2026-09-21, `python scripts/run_tests.py -q` (default marker filter) reports on the order of **~250 passed** with a handful of skips/deselections when the production model is present. If the totals drop unexpectedly, confirm new tests are marked correctly rather than silently excluded.
