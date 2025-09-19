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

# Production model (recommended)
PYTHONPATH=src python scripts/04_create_production_model.py

# Lightweight model (faster inference)
PYTHONPATH=src python scripts/06_create_lightweight_model.py

# Test sampling strategies
PYTHONPATH=src python scripts/01_test_sampling_strategies.py data/02_stg/stg_disaster_response.db

# Compare models
PYTHONPATH=src python scripts/compare_models.py
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
3. Sampling strategies handle severe class imbalance
4. RandomForest with MultiOutputClassifier for 36 categories
5. Models serialized with joblib in `model/` directory

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
- Auto-downloads models from Google Drive for cloud deployment
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