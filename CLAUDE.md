# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment

- **Python Version**: 3.12+ required
- **Virtual Environment**: Activate with `source .venv/Scripts/activate` (Windows) before running commands
- **Package**: Installed as `disasterproject` from `src/` directory using setuptools

## Core Commands

### Data Processing
```bash
python data/process_data.py data/01_raw/disaster_messages.csv data/01_raw/disaster_categories.csv data/02_stg/stg_disaster_response.db
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
- **models/**: ML pipeline with sampling strategies (SMOTE, ADASYN) + RandomForest
- **evaluation/**: Multi-label classification metrics
- **utils/**: Configuration, experiment tracking, I/O

### Key Data Flow
1. Raw CSV → SQLite staging DB (via `data/process_data.py`)
2. ETL pipeline processes text + creates multi-label targets
3. Sampling strategies handle severe class imbalance
4. RandomForest with MultiOutputClassifier for 36 categories
5. Models serialized with joblib in `models/` directory

### Experiment System
- Structured experiments in `experiments/` with naming convention
- Each contains: models, parameters, metrics, visualizations
- Experiment tracker manages reproducibility

### Web Application
- Flask factory pattern in `app/`
- **Important**: Use `run.py` as entry point (handles factory setup)
- Auto-downloads models from Google Drive for cloud deployment
- Optimized for both local development and Replit

## Code Conventions

From `.cursor/rules/`:
- Functions under 50 lines, single responsibility
- Specific exception handling (ValueError vs Exception)
- Import grouping: standard library, third-party, local
- Focus on functional changes over linting issues