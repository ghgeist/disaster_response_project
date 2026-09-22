# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment

- **Python Version**: 3.12+ required
- **Virtual Environment**: Activate with `source .venv/Scripts/activate` (Windows) before running commands
- **Package**: Installed as `disasterproject` from `src/` directory using setuptools

## Core Commands

### Data Processing
```bash
python scripts/process_data.py data/01_raw/disaster_messages.csv data/01_raw/disaster_categories.csv data/02_stg/stg_disaster_response.db
```

### Model Training
```bash
# Note: Scripts now use package imports. Run with PYTHONPATH or install package:
# Option 1: pip install -e . (installs disasterproject package)
# Option 2: PYTHONPATH=src python <script>

# Production candidate (LogisticRegression) + promotion
PYTHONPATH=src python scripts/02_training/03_create_experimental_model.py --algorithm logistic_regression
PYTHONPATH=src python scripts/03_optimization/optimize_per_category_thresholds.py --model-path <candidate.pkl>
PYTHONPATH=src python scripts/07_operations/promote_model.py experiments/experimental_runs/<date> --dry-run

# Legacy RandomForest experiments (outputs under experiments/legacy_rf/)
PYTHONPATH=src python scripts/02_training/04_create_production_model.py --params experiments/model_candidates/vocab_15k.json --class-weights experiments/model_candidates/class_weights.json

# Test sampling strategies
PYTHONPATH=src python scripts/02_training/01_test_sampling_strategies.py data/02_stg/stg_disaster_response.db

# Compare models
PYTHONPATH=src python scripts/04_evaluation/compare_models.py
```

### Web Application
```bash
# Use run.py as entry point (not app/app.py directly)
python run.py
# Runs on http://localhost:5000
```

### Code Quality
```bash
pylint src/ scripts/ data/ app/
python scripts/system_validation.py
```

## Architecture Overview

Disaster response message classification system with modular ML pipeline targeting 36 disaster categories.

### Core Package (`src/disasterproject/`)
- **data/**: ETL pipeline (loader, preprocessor, column definitions)
- **model/**: Trained models and artifacts with metadata
- **evaluation/**: Multi-label classification metrics
- **utils/**: Configuration, experiment tracking, I/O

### Key Data Flow
1. Raw CSV → SQLite staging DB (via `scripts/process_data.py`)
2. ETL pipeline processes text + creates multi-label targets
3. Class imbalance handled mainly via per-label thresholds (class-weighting available; see ADR-008)
4. LogisticRegression with MultiOutputClassifier for 36 categories (RandomForest retained for explicit experimental comparisons)
5. Candidate training → calibration threshold tuning → frozen-eval reporting → promotion
6. Promoted production artifacts live under `model/` with stem-bound thresholds/labels and `MODEL_INFO` provenance (strict model/threshold/label binding)

### Experiment System
- Organized experiments in `experiments/` with clear structure:
  - `experimental_runs/{YYYY-MM-DD}/` - Dated experiment results (models, metrics, logs)
  - `experimental_configs/` - Reusable configurations (hyperparameters, sampling strategies)
  - `comparisons/` - Timestamped model comparison reports
  - `logs/` - Training and execution logs
  - `model_candidates/` - Hyperparameter optimization results
- Each dated folder contains complete experiment artifacts from that date
- Experiment tracker manages reproducibility and metadata

### Web Application
- Flask factory pattern in `app/`
- **Important**: Use `run.py` as entry point (handles factory setup)
- Single React app in `_vendor/figma_make` serves both Storm Signal dashboard and Model Information; build with `scripts/build_dashboard.py`, output in `app/static/dashboard/`
- Loads locally promoted production pickles from `model/` (historical Google Drive auto-download path is retired; see ADR-003)
- Optimized for both local development and Replit

## AI Model Usage

### When to use Gemini
- Initial codebase exploration and research to conserve Claude tokens
- Large file analysis to identify relevant sections before detailed work
- Broad architectural questions about the disaster response system
- Understanding unfamiliar code patterns or libraries in the codebase

### When to use Codex
- Code completion and boilerplate generation for ML pipelines
- Writing test cases and unit tests for the disaster response models
- Generating data transformation and preprocessing functions
- Creating utility functions and helper methods
- Quick prototyping of new features before full implementation
- Note: Codex has multiple model options - choose appropriate model based on task complexity

## Experiment Organization

### Folder Structure Rules
- **Dated Runs**: All experiment artifacts go in `experiments/experimental_runs/{YYYY-MM-DD}/`
- **Date Consistency**: Only put artifacts from that specific date in each dated folder
- **Sub-experiments**: Use subfolders for related experiments (e.g., `hierarchy_initial/`, `hierarchy_optimized/`)
- **Configuration Separation**: Reusable configs in `experimental_configs/`, results in `experimental_runs/`
- **Legacy Placement**: Old or one-off artifacts in `experimental_runs/legacy/`

### File Placement Guidelines
- Model files (.pkl): `experimental_runs/{date}/`
- Metrics/results (.csv): `experimental_runs/{date}/`
- Hyperparameter configs (.json): `experimental_configs/hyperparameters/`
- Comparison reports: `comparisons/` with timestamps
- Training logs: `logs/` with clear naming

## Code Conventions

From `.cursor/rules/`:
- Functions under 50 lines, single responsibility
- Specific exception handling (ValueError vs Exception)
- Import grouping: standard library, third-party, local
- Focus on functional changes over linting issues

## Workflow

**IMPORTANT**: Always seek user approval before implementing code changes. Present suggestions and plans first, then wait for explicit approval before proceeding with implementation.