# Running Tests in Cursor Web UI

## Problem

Cursor Web UI environments may not have `python` available in PATH, causing test execution to fail with errors like:

```
python: command not found (no Python runtime available in this environment)
```

## Solution

Use the portable test runner script that automatically detects the best available Python/pytest command:

```bash
python scripts/run_tests.py tests/test_app_smoke.py -q
```

Or for the full test suite:

```bash
python scripts/run_tests.py -q
```

## How It Works

The test runner (`scripts/run_tests.py`) follows pytest best practices by:

1. **Trying direct `pytest` invocation first** (recommended by pytest documentation)
2. **Falling back to `python3 -m pytest`** (common in web environments)
3. **Falling back to `python -m pytest`** (standard local development)

This ensures tests can run across different environments:
- Local development (with venv)
- Replit (managed environment)
- Cursor Web UI (limited runtime)
- CI/CD environments

## Environment Detection

The project includes environment detection utilities in `src/disasterproject/utils/env.py`:

- `is_replit()` - Detects Replit environments
- `is_cursor_web_ui()` - Detects Cursor Web UI environments
- `get_venv_status()` - Comprehensive environment status

## Alternative: Direct pytest

If `pytest` is available directly in your PATH, you can use it directly:

```bash
pytest tests/test_app_smoke.py -q
```

However, the test runner script is recommended for maximum portability.
