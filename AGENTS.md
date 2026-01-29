# Codex CLI Agent Guide

This document explains how to work on the Disaster Response project when operating as the Codex CLI coding agent.

## Repository Overview
- `src/disasterproject/`: Core Python package for data ETL, model training/evaluation, and shared utilities.
- `app/`: Flask web UI (templates, static assets, factory). Always launch via `run.py`.
- `scripts/`: Reproducible pipelines for training, experiments, and maintenance tasks.
- `data/`: Project datasets (`01_raw`, `02_stg`, `04_fct`). Large artifacts live here and stay out of git.
- `experiments/`: Tracked experiment configs, metrics, and artifacts.
- `tests/`: Pytest suite (smoke + functional coverage).
- `model/`: Saved models, parameters, and thresholds consumed by the app.

## Environment & Shell Conventions
- Target Python 3.12+. Virtual environment is conditionally required based on execution environment (see below).
- Install dependencies with `python -m pip install -r requirements.txt` and `python -m pip install -r requirements-dev.txt` when present.
- **Do not use `source .venv/bin/activate`.** Always call the interpreter explicitly (e.g., `./.venv/bin/python -m pytest -q`).
- In the Codex CLI, prefer `shell` calls of the form `["bash", "-lc", "<command>"]` and always set the `workdir` parameter. Avoid `cd` chains; invoke commands from the project root when possible.
- Use `rg`/`rg --files` for fast code or file search. Fall back to `grep` only if ripgrep is unavailable.
- The workspace has restricted network access; plan around offline execution (no package downloads unless already vendored).

## Codex Cloud (Web Codex) Quickstart
- **How to set up env**: `bash scripts/ci.sh` (creates `.venv` if missing and installs deps).
- **How to run tests**: `bash scripts/ci.sh` (canonical command).
- Do not fetch network resources at test runtime.
- If tests require secrets, they must be mocked or guarded to keep CI deterministic.

### Virtual Environment (conditional based on environment)

- **Local Development (Windows/Linux)**: Virtual environment is **required**
  - Standard venv location: `.venv/` at the project root.
  - Create once: `python -m venv .venv`.
  - Do **not** activate the venv; call the interpreter directly:
    - Bash: `./.venv/bin/python scripts/04_create_production_model.py`
    - PowerShell: `.\.venv\Scripts\python scripts\04_create_production_model.py`

- **Replit Environment (SSH)**: Virtual environment is **not required**
  - Replit manages Python environment automatically
  - Dependencies are installed globally in the Replit container
  - Environment detection: Uses `REPLIT_DB_URL` or `REPL_ID` environment variables
  - Scripts automatically skip venv checks when running in Replit
  - In Codex CLI shell calls, execute commands directly without venv activation:
    - Bash example: `["bash", "-lc", "python scripts/04_create_production_model.py"]` (set `workdir` to the repo root)
    - PowerShell example: `["pwsh", "-NoProfile", "-Command", "python scripts/04_create_production_model.py"]`

- Prefer `python -m pip ...` to ensure installs land in the active venv (local) or correct environment (Replit).

## Core Commands

### Setup & Data Preparation
```bash
pip install -r requirements.txt
pip install -e .
python scripts/process_data.py data/01_raw/disaster_messages.csv data/01_raw/disaster_categories.csv data/02_stg/stg_disaster_response.db
```

### Model Training & Evaluation
```bash
python scripts/04_create_production_model.py
python scripts/01_test_sampling_strategies.py data/02_stg/stg_disaster_response.db
python scripts/compare_models.py
```

### Application & Quality Gates
```bash
python run.py              # Flask app on http://localhost:5000
pytest -q                  # Full test suite (if pytest in PATH)
python scripts/run_tests.py -q  # Portable test runner (recommended for web UI/CI)
pytest tests/test_smoke.py -q   # Specific test file
python scripts/run_tests.py tests/test_smoke.py -q  # Portable version
pre-commit run --all-files
```

## Coding Standards
- Follow Black + Ruff defaults; keep imports well grouped and tidy. Maintain 4-space indents and type hints for public functions.
- Functions should stay focused and readable (single responsibility, <50 lines when practical). Document tricky logic with concise comments or docstrings.
- Prefer explicit exception handling (e.g., `ValueError`) over bare `Exception`.
- Keep modules cohesive; add new utilities under `src/disasterproject/` rather than ad hoc script code.

## Testing Expectations
- Add or update pytest coverage for new behavior. Keep tests fast and deterministic.
- Run at least the smoke suite (`pytest tests/test_smoke.py -q`) before handing changes back; run the full suite when time allows.
- Use fixtures and temporary paths instead of mutating checked-in data.

## Commit & PR Checklist
- Conventional Commit messages (`feat:`, `fix:`, `refactor:`, etc.).
- Ensure `pre-commit` and tests pass locally.
- Update `README.md`/`docs/` when behavior, interfaces, or commands change.
- For UI edits, capture a local screenshot for PR context.

## Data & Security Notes
- Never commit secrets. Configure `GDRIVE_MODEL_ID` through the environment when the app needs to download models.
- Confirm `data/02_stg/stg_disaster_response.db` exists before running the Flask app.
- Store large datasets and serialized models under `data/` or `model/` (both git-ignored by default).
