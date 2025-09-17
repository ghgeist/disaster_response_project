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
- Target Python 3.12+. Create/activate a virtual environment before running project commands.
- Install dependencies with `pip install -r requirements.txt` followed by `pip install -e .` so `src/` is importable as `disasterproject`.
- In the Codex CLI, prefer `shell` calls of the form `["bash", "-lc", "<command>"]` and always set the `workdir` parameter. Avoid `cd` chains; invoke commands from the project root when possible.
- Use `rg`/`rg --files` for fast code or file search. Fall back to `grep` only if ripgrep is unavailable.
- The workspace has restricted network access; plan around offline execution (no package downloads unless already vendored).

## Core Commands

### Setup & Data Preparation
```bash
pip install -r requirements.txt
pip install -e .
python data/process_data.py data/01_raw/disaster_messages.csv data/01_raw/disaster_categories.csv data/02_stg/stg_disaster_response.db
```

### Model Training & Evaluation
```bash
python scripts/04_create_production_model.py
python scripts/06_create_lightweight_model.py
python scripts/01_test_sampling_strategies.py data/02_stg/stg_disaster_response.db
python scripts/compare_models.py
```

### Application & Quality Gates
```bash
python run.py              # Flask app on http://localhost:5000
pytest -q                  # Full test suite
pytest tests/test_smoke.py -q
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
