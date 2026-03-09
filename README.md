![Storm Signal dashboard preview](images/image.png)

# Storm Signal

Storm Signal is a disaster-response message classifier built with Python, scikit-learn, Flask, and React. It takes short text messages, predicts one or more disaster-related categories, and exposes the results through a local web app and JSON APIs.

The repository includes:

- A reusable Python package in `src/disasterproject/` for data processing, training, evaluation, and hierarchy logic
- A Flask application in `app/` served through `run.py`
- A React dashboard in `_vendor/figma_make/` that Flask serves as static assets
- Training, evaluation, and operations scripts in `scripts/`
- Experiment outputs and configs in `experiments/`

## What It Does

- Classifies messages into 36 disaster-related categories
- Supports multi-label prediction rather than forcing a single class
- Includes hierarchy-aware post-processing for parent/child label consistency
- Ships with scripts for data prep, model training, comparison, and promotion
- Provides a browser-based UI plus JSON endpoints for classification and model metadata

## Repository Layout

```text
disaster_response_project/
├── app/                    # Flask app, routes, services, templates, static files
├── data/                   # Raw, staged, and derived datasets
├── docs/                   # Supporting documentation
├── experiments/            # Experiment configs, runs, comparisons, logs
├── model/                  # Trained model artifacts and thresholds
├── scripts/                # Data, training, evaluation, and ops scripts
├── src/disasterproject/    # Core Python package
├── tests/                  # Pytest suite
├── run.py                  # Local app entry point
└── wsgi.py                 # WSGI entry point for Gunicorn
```

## Quick Start

### Requirements

- Python 3.12+
- Node.js and npm if you plan to work on the React dashboard

### 1. Create a local environment

Use a virtual environment for local development. Call the interpreter directly instead of activating it.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pip install -e .
```

On macOS or Linux, use `./.venv/bin/python` instead of `.\.venv\Scripts\python`.

### 2. Build the training database

```bash
python scripts/01_data/process_data.py data/01_raw/disaster_messages.csv data/01_raw/disaster_categories.csv data/02_stg/stg_disaster_response.db
```

This creates the SQLite database the training scripts and app expect at `data/02_stg/stg_disaster_response.db`.

### 3. Train or provide a model

To create a production model locally:

```bash
python scripts/02_training/04_create_production_model.py --params experiments/model_candidates/vocab_15k.json --class-weights experiments/model_candidates/class_weights.json
```

The app loads the newest production model in `model/` matching `disaster_*_prod_*.pkl`, unless `MODEL_FILENAME` is set explicitly.

### 4. Run the app

```bash
python run.py
```

Open `http://localhost:5000`.

The app will fail fast if either of these is missing:

- `data/02_stg/stg_disaster_response.db`
- A trained model in `model/`

## Web App

The Flask app is created by `app.app:create_app()` and started locally through `run.py`. In production, `wsgi.py` exposes the same application for Gunicorn.

Useful routes:

- `GET /dashboard` for the main dashboard SPA
- `GET /production-model` for the model information dashboard
- `GET /about` for the public about page
- `POST /api/classify` for JSON classification
- `GET /api/feed`, `GET /api/metrics`, and `GET /api/categories` for dashboard data
- `GET /health` and `GET /health/detailed` for health checks

## React Dashboard

The frontend lives in `_vendor/figma_make/`. During development, you can run it with Vite. To make Flask serve your frontend changes, rebuild and copy the static assets into `app/static/dashboard/`.

```bash
cd _vendor/figma_make
npm install
npm run dev

# when you want Flask to serve the updated build
npm run build
python ../../scripts/build_dashboard.py
```

See `_vendor/figma_make/README.md` for frontend-specific notes.

## Training And Experiments

The project keeps most ML workflows under `scripts/` and `experiments/`.

Common commands:

```bash
# compare model outputs across experiment runs
python scripts/04_evaluation/compare_models.py

# evaluate hierarchy behavior
python scripts/04_evaluation/evaluate_hierarchy.py

# test sampling strategies
python scripts/02_training/01_test_sampling_strategies.py data/02_stg/stg_disaster_response.db

# promote an experiment artifact to production
python scripts/07_operations/promote_model.py experiments/experimental_runs/<run-folder> --dry-run
```

For deeper workflow docs, see:

- `scripts/README.md`
- `experiments/README.md`
- `docs/adr/adr-008-class-weighting-over-sampling.md`

## Testing

Use the portable test runner when possible:

```bash
python scripts/run_tests.py -q
python scripts/run_tests.py tests/test_app_smoke.py -q
```

If `pytest` is already available in your environment, direct invocation works too:

```bash
pytest -q
pytest -q -m "not perf and not slow"
```

See `docs/testing.md` for marker usage, optional suites, and CI guidance.

## Deployment Notes

- Local development uses `python run.py`
- Production WSGI entry point is `wsgi.py`
- Gunicorn command:

```bash
gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 wsgi:application
```

- Replit and other hosted environments still need the SQLite database and a production model artifact available on disk

## Troubleshooting

If the app cannot start:

- Rebuild the database with `python scripts/01_data/process_data.py ...`
- Train or copy a production model into `model/`
- Confirm `MODEL_FILENAME`, `PORT`, and related environment overrides are set correctly

If the dashboard does not reflect React changes:

- Run `npm run build` in `_vendor/figma_make/`
- Run `python scripts/build_dashboard.py`

If training scripts cannot import `disasterproject`:

- Reinstall the editable package with `python -m pip install -e .`

## License

This project is licensed under the MIT License. See `LICENSE`.
