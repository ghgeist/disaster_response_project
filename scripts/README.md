# Scripts

This directory holds the command-line entry points for data prep, model training, evaluation, deployment checks, and maintenance tasks. The scripts are grouped by workflow stage, but the root-level helpers such as `run_tests.py`, `build_dashboard.py`, and `ci.sh` are part of the normal day-to-day workflow too.

## Layout

```text
scripts/
├── 01_data/             # ETL and evaluation-set preparation
├── 02_training/         # Training and experiment creation
├── 03_optimization/     # Threshold and hierarchy tuning
├── 04_evaluation/       # Comparison and reporting
├── 05_analysis/         # Analysis helpers
├── 06_validation/       # Validation and deployment checks
├── 07_operations/       # Promotion and operational helpers
├── utils/               # Shared utility scripts
├── archive/             # Older one-off or superseded scripts
├── build_dashboard.py   # Build/copy React dashboard into Flask static assets
├── ci.sh                # Create venv, install deps, run pytest
└── run_tests.py         # Portable pytest wrapper
```

## Common Workflows

### Prepare the database

```bash
python scripts/01_data/process_data.py data/01_raw/disaster_messages.csv data/01_raw/disaster_categories.csv data/02_stg/stg_disaster_response.db
```

### Train a production model

```bash
python scripts/02_training/04_create_production_model.py --params experiments/model_candidates/vocab_15k.json --class-weights experiments/model_candidates/class_weights.json
```

This script now expects explicit `--params` and `--class-weights` arguments.

### Compare or evaluate models

```bash
python scripts/04_evaluation/compare_models.py
python scripts/04_evaluation/evaluate_hierarchy.py
```

### Run tests

```bash
python scripts/run_tests.py -q
python scripts/run_tests.py tests/test_app_smoke.py -q
```

### Build the frontend for Flask

```bash
python scripts/build_dashboard.py
```

## Root-Level Helpers

### `build_dashboard.py`

Builds the React app in `_vendor/figma_make/` and copies the output into `app/static/dashboard/` so Flask serves the latest frontend bundle.

Useful commands:

```bash
python scripts/build_dashboard.py
python scripts/build_dashboard.py --watch
```

### `run_tests.py`

Portable pytest wrapper that detects the best available pytest invocation for the current environment. Use this when you want one command that works across local development, Replit, Cursor Web UI, and CI.

### `ci.sh`

Bootstraps `.venv`, installs dependencies, and runs the default pytest suite. This is the closest thing to the repo's canonical CI-style local check.

```bash
bash scripts/ci.sh
```

## Directory Notes

### `01_data/`

- `process_data.py`: Builds the staged SQLite database from the raw CSV inputs
- `create_frozen_eval_ids.py`: Creates a stable evaluation split for repeatable comparisons

### `02_training/`

- `01_test_sampling_strategies.py`: Interactive sampler comparison workflow
- `02_test_hyperparameters.py`: Grid-search style hyperparameter exploration using the staged database and experiment configs
- `03_create_experimental_model.py`: Produces candidate experiment artifacts under `experiments/`
- `04_create_production_model.py`: Trains and writes a production-ready model artifact plus metrics/logs
- `run_batch_experiments.py`: Runs multiple experiment configurations without the interactive menu
- `test_experimental_model.py`: Validates a trained experimental model

### `03_optimization/`

- `optimize_hierarchy_threshold_reduction.py`: Tunes hierarchy threshold reduction behavior
- `optimize_per_category_thresholds.py`: Searches for better per-label decision thresholds

### `04_evaluation/`

- `compare_models.py`: Compares experiment outputs
- `compare_vocabulary_models.py`: Focuses on vocabulary-size tradeoffs
- `compare_child_alone.py`: Specialized analysis for the `child_alone` label
- `evaluate_hierarchy.py`: Measures hierarchy consistency before and after correction
- `visualize_performance.py`: Produces visual summaries of model results

### `05_analysis/`

- `analyze_vocabulary_distribution.py`: Vocabulary and feature-distribution analysis
- `eda_functions.py`: Shared EDA helpers

### `06_validation/`

- `system_validation.py`: Checks whether the local project setup is usable
- `validate_multilabel_sampling.py`: Validates multilabel sampling behavior
- `deployment_health_check.py`: Sanity checks for deployment readiness

### `07_operations/`

- `promote_model.py`: Validates and promotes an experiment artifact into `model/`
- `model_naming_utility.py`: Naming helpers for model artifacts
- `check_thresholds.py`: Shows which threshold file the app would currently pick up
- `manual_smoke_test.py`: Flask route smoke test helper

Promotion examples:

```bash
# Validate a candidate only
python scripts/07_operations/promote_model.py experiments/experimental_runs/<run-folder> --dry-run

# Promote and print the resulting model path
python scripts/07_operations/promote_model.py experiments/experimental_runs/<run-folder> --print-new-path
```

### `utils/`

- `ensure_venv.py`: Environment guard for local workflows that require a venv
- `estimate_search_time.py`: Runtime estimation helper for longer search or optimization tasks

## Inputs And Outputs

Most scripts depend on some combination of:

- `data/02_stg/stg_disaster_response.db`
- configs in `experiments/model_candidates/`
- configs in `experiments/experimental_configs/`
- source code in `src/disasterproject/`

Typical outputs land in:

- `model/` for promoted or production artifacts
- `experiments/experimental_runs/` for candidate runs
- `experiments/results/` and `experiments/comparisons/` for reports
- `experiments/logs/` for execution logs

## Notes On Older Scripts

`scripts/archive/` contains older or one-off scripts kept for reference. Treat that directory as historical context, not as the default place to look for the current workflow.
