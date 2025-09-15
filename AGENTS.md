# Repository Guidelines

## Project Structure & Module Organization
- `src/disasterproject/`: Core ML package (`data/`, `models/`, `evaluation/`, `utils/`).
- `app/`: Flask web app (`app.py`, `templates/`, `static/`). Run via `run.py`.
- `scripts/`: Training, experimentation, and utilities (e.g., `04_create_production_model.py`).
- `data/`: Raw (`01_raw`), staged (`02_stg`), and facts/results (`04_fct`).
- `experiments/`: Configs and results for tracked runs.
- `tests/`: Pytest suite (smoke and functional tests).
- `model/`: Trained artifacts, parameters, thresholds.

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt` and `pip install -e .` (enables `src/`).
- Format/lint: `pre-commit run --all-files` (Black, Ruff).
- Tests: `pytest -q` or quick check `pytest tests/test_smoke.py -q`.
- Data ETL: `python data/process_data.py data/01_raw/disaster_messages.csv data/01_raw/disaster_categories.csv data/02_stg/stg_disaster_response.db`.
- Train: `python scripts/04_create_production_model.py` or `python scripts/06_create_lightweight_model.py`.
- Run app: `python run.py` then open `http://localhost:5000`.

## Coding Style & Naming Conventions
- Python 3.12, 4‑space indentation, prefer type hints.
- Use Black for formatting and Ruff for linting; keep imports and style consistent.
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants, modules/files lowercase.
- Keep modules focused (single responsibility) and add clear docstrings.

## Testing Guidelines
- Framework: `pytest` with tests in `tests/` named `test_*.py`.
- Add tests for new behavior and edge cases; prefer small, fast unit tests.
- Run `pytest -q` locally; ensure smoke tests pass at minimum before PR.

## Commit & Pull Request Guidelines
- Commits: follow Conventional Commits (`feat:`, `fix:`, `refactor:`, etc.).
- PRs: concise description, link issues, list changes, include screenshots for UI updates.
- Checklist before opening PR: ETL/model steps (if relevant) reproducible, `pre-commit` clean, tests passing, docs updated (`README.md`/`docs/`).

## Security & Configuration Tips
- Do not commit secrets. Configure `GDRIVE_MODEL_ID` for model download when needed.
- Ensure `data/02_stg/stg_disaster_response.db` exists before running the app.
- Large data/model files belong in `data/` or `model/` (git‑ignored as appropriate).
