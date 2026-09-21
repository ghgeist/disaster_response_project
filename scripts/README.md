# Scripts Directory

This directory contains scripts for training, testing, and analyzing disaster response classification models, organized by workflow stage for easy navigation.

## Directory Structure

```
scripts/
├── 01_data/                    # Data processing & preparation
├── 02_training/                # Model training scripts
├── 03_optimization/            # Model optimization scripts
├── 04_evaluation/              # Model evaluation & comparison
├── 05_analysis/                # Data & model analysis
├── 06_validation/              # Validation & testing
├── 07_operations/              # MLOps & model management
├── utils/                      # Shared utilities
└── archive/                    # Legacy scripts
```

## 01_data/ - Data Processing & Preparation

### `process_data.py`
ETL pipeline for processing raw disaster messages and categories.
- **Use when**: Preparing data for model training
- **Usage**: `python scripts/01_data/process_data.py data/01_raw/disaster_messages.csv data/01_raw/disaster_categories.csv data/02_stg/stg_disaster_response.db`
- **Output**: SQLite database at `data/02_stg/stg_disaster_response.db`

### `create_frozen_eval_ids.py`
Creates frozen evaluation datasets for consistent model comparison.
- **Use when**: Creating reproducible evaluation sets
- **Output**: Evaluation ID sets for consistent testing

### `create_frozen_cal_ids.py`
Carves a train-side calibration UID set from the non-eval pool for threshold tuning.
- **Use when**: Establishing the three-way train/cal/eval contract
- **Usage**: `python scripts/01_data/create_frozen_cal_ids.py`
- **Output**: `experiments/experimental_configs/eval_sets/cal_ids.json` (includes per-critical-label positive support)
- **Note**: Does not modify `eval_ids.json`; fails if any critical label has zero cal positives

## 02_training/ - Model Training Scripts

### `01_test_sampling_strategies.py`
Tests different sampling methods for handling class imbalance with interactive menu.
- **Use when**: Comparing sampling approaches (SMOTE, ADASYN, conservative)
- **Usage**: `python scripts/02_training/01_test_sampling_strategies.py data/02_stg/stg_disaster_response.db`
- **Output**: Models trained with different sampling strategies
- **Interactive**: Yes - provides menu for strategy selection

### `02_test_hyperparameters.py`
Performs hyperparameter optimization using GridSearchCV.
- **Use when**: Finding optimal model parameters
- **Usage**: `python scripts/02_training/02_test_hyperparameters.py data/02_stg/stg_disaster_response.db model/optimized.pkl`
- **Output**: Optimized model with best parameters
- **Dependencies**: `experiments/configs/hyperparameter_optimization.json`

### `03_create_experimental_model.py`
Creates experimental models using candidate configurations (active LogisticRegression training path).
- **Use when**: Testing new model configurations before production; retraining LR under the train/cal/eval contract
- **Usage**: `python scripts/02_training/03_create_experimental_model.py --algorithm logistic_regression --params experiments/model_candidates/vocab_15k.json`
- **Output**: Experimental models saved to `experiments/` directory; diagnostic `{stem}_f2_thresholds.json` (not the canonical thresholds file)
- **Split**: With frozen IDs, fits on residual train only (`~(cal ∪ eval)`); requires `cal_ids.json`; inline F2 tunes on cal

### `04_create_production_model.py`
Creates a production disaster response classification model with optional class weighting via config (currently disabled in `experiments/model_candidates/class_weights.json`).
- **Use when**: Creating the main production model for deployment
- **Usage**: `python scripts/02_training/04_create_production_model.py --params experiments/model_candidates/vocab_15k.json --class-weights experiments/model_candidates/class_weights.json`
- **Output**: Production model under `model/`, performance metrics, and training logs
- **Dependencies**: Parameter and class weight configuration files

### `run_batch_experiments.py`
Runs multiple experiments in batch mode without user interaction.
- **Use when**: Testing multiple configurations systematically
- **Usage**: `python scripts/02_training/run_batch_experiments.py`
- **Output**: Multiple models and comparison results
- **Runs**: baseline, smote, adasyn, and conservative experiments

### `test_experimental_model.py`
Tests experimental models for validation.
- **Use when**: Validating experimental model performance
- **Output**: Test results and validation metrics

## 03_optimization/ - Model Optimization Scripts

### `optimize_per_category_thresholds.py`
Tunes per-label thresholds on the calibration split; reports metrics on frozen eval only.
- **Use when**: Fine-tuning per-category decision thresholds under the three-way contract
- **Usage**: `python scripts/03_optimization/optimize_per_category_thresholds.py --model-path <model.pkl>`
- **Output**: `{model_stem}_thresholds.json` with `calibration_stats` (diagnostic) and `category_stats` / `performance` (eval-reported)

### `optimize_hierarchy_threshold_reduction.py`
Optimizes hierarchy post-processing threshold reduction parameter.
- **Use when**: Tuning hierarchy constraint enforcement
- **Output**: Optimized hierarchy threshold reduction values

## 04_evaluation/ - Model Evaluation & Comparison

### `compare_models.py`
Compares performance between different experiment results.
- **Use when**: Analyzing differences between model versions
- **Usage**: `python scripts/04_evaluation/compare_models.py`
- **Output**: Performance comparison reports
- **Dependencies**: Experiment tracker system

### `compare_vocabulary_models.py`
Compares models with different vocabulary sizes.
- **Use when**: Analyzing vocabulary size impact on performance
- **Output**: Vocabulary comparison reports

### `compare_child_alone.py`
Specific analysis for the `child_alone` category.
- **Use when**: Analyzing child_alone label performance
- **Output**: Category-specific analysis

### `evaluate_hierarchy.py`
Evaluates hierarchy constraint enforcement.
- **Use when**: Testing hierarchy post-processing logic
- **Usage**: `python scripts/04_evaluation/evaluate_hierarchy.py`
- **Output**: Hierarchy violation reports and metrics

### `visualize_performance.py`
Creates performance visualizations.
- **Use when**: Generating charts and graphs for analysis
- **Output**: Performance visualization files

## 05_analysis/ - Data & Model Analysis

### `analyze_vocabulary_distribution.py`
Analyzes vocabulary distribution across the dataset.
- **Use when**: Understanding vocabulary characteristics
- **Output**: Vocabulary distribution analysis

### `eda_functions.py`
Contains exploratory data analysis functions.
- **Use when**: Data exploration and analysis
- **Output**: Analysis results and insights

## 06_validation/ - Validation & Testing

### `system_validation.py`
Validates system components and dependencies.
- **Use when**: Checking system health and dependencies
- **Usage**: `python scripts/06_validation/system_validation.py`
- **Output**: Validation reports

### `validate_multilabel_sampling.py`
Validates multi-label sampling implementations.
- **Use when**: Testing sampling method correctness
- **Usage**: `python scripts/06_validation/validate_multilabel_sampling.py`
- **Output**: Validation reports and performance metrics

### `deployment_health_check.py`
Performs deployment health verification.
- **Use when**: Verifying deployment readiness
- **Output**: Health check reports

## 07_operations/ - MLOps & Model Management

### `promote_model.py`
Promotes validated experimental models to production.
- **Use when**: Promoting experimental models to production
- **Usage**: 
  ```bash
  # Validate only
  python scripts/07_operations/promote_model.py experiments/experimental_runs/2025-09-18 --dry-run
  
  # Promote (default: leave app/config.py alone — Config auto-discovers model/*_prod_*.pkl)
  python scripts/07_operations/promote_model.py experiments/experimental_runs/2025-09-18 --keep-old 1 --print-new-path
  
  # Opt-in: rewrite a disaster_* MODEL_FILENAME literal in app/config.py
  python scripts/07_operations/promote_model.py experiments/experimental_runs/2025-09-18 --update-config
  ```
- **Notes**: 
  - Enforces the train/cal/eval evaluation contract (ADR-007): baseline frozen-eval micro F1, frozen-eval critical recall, size, weighted-F1 relative-drop guardrail, and `optimization_split=calibration` / `reporting_split=frozen_eval` provenance
  - Discovers `{model_stem}_thresholds.json` for both validation and deploy; stages model+thresholds, verifies hashes, finalizes with production model last
  - Requires a complete `thresholds` map covering every `TARGET_COLUMNS` label with values in `[0, 1]`
  - Validation gates use `PERFORMANCE_THRESHOLDS` from `src/disasterproject/utils/config.py`
  - On success, promoted file named `disaster_{algorithm}_<version>_prod_<YYYY-MM-DD>.pkl` placed under `model/`
  - Does **not** rewrite `app/config.py` by default (auto-discovery); pass `--update-config` only if you maintain a hardcoded `disaster_*` literal
  - Archives prior production **metadata** + SHA256 to `experiments/model_archive/` (does **not** copy the `.pkl`); rollback is via Git history of tracked `model/*_prod_*.pkl`
  - Missing/invalid evidence fails closed; `--force` overrides metric/provenance gates only (model + thresholds artifacts with hashes, and a loadable `lr`/`rf` model, remain required)
### `model_naming_utility.py`
Model naming helper utilities.
- **Use when**: Generating consistent model names
- **Output**: Standardized model naming functions

## utils/ - Shared Utilities

### `ensure_venv.py`
Ensures virtual environment is activated for local development.
- **Use when**: Checking venv status in local development
- **Usage**: `python scripts/utils/ensure_venv.py`
- **Note**: Automatically detects Replit environment and skips venv check

### `configure_gh_git_credentials.sh`
Configures **repo-local** git credentials so `git push` works with the `gh` CLI on Replit/Cursor SSH (avoids broken `replit-git-askpass`).
- **Use when**: `git push` fails with `could not read Username for 'https://github.com'`
- **Usage**:
  ```bash
  gh auth login -h github.com -p https -w   # once
  bash scripts/utils/configure_gh_git_credentials.sh
  ```
- **Note**: Writes only under `.git/` (not committed); see `AGENTS.md`

### `estimate_search_time.py`
Time estimation utilities for optimization tasks.
- **Use when**: Estimating optimization runtime
- **Output**: Time estimates for search operations

## Archive Directory

The `archive/` directory contains legacy scripts that are no longer actively used:
- `compare_results.py` - Legacy result comparison
- `run_all_experiments.py` - Legacy experiment runner
- `systematic_testing_framework.py` - Legacy testing framework
- `train_classifier_original.py` - Original training script
- `train_classifier.py` - Legacy training script
- `validate_structure.py` - Legacy structure validation
- `prepare_15k_model_for_promotion.py` - One-off script for 2025-11-06 model promotion (completed)
- `optimize_critical_thresholds_inc1.py` - Incremental threshold optimization for obsolete model version
- `compare_csv_models.py` - Enhanced model comparison tool for arbitrary CSV files (archived 2026-01-26)
- `migrate_experimental_paths.py` - Migration utility (migration completed 2026-01-26)
- `validate_production_model.py` - Production model validation utility (archived 2026-01-26)
- `validate_threshold_optimization_results.py` - Threshold optimization validation utility (archived 2026-01-26)
- `validate_ml_execution_environment.py` - Pre-execution environment validation utility (archived 2026-01-26)
- `test_deployment_scenarios.py` - Deployment scenario testing utility (archived 2026-01-26)

## Quick Reference

### Common Workflows

**Data Preparation:**
```bash
python scripts/01_data/process_data.py data/01_raw/disaster_messages.csv data/01_raw/disaster_categories.csv data/02_stg/stg_disaster_response.db
```

**Model Training:**
```bash
# Create production model
python scripts/02_training/04_create_production_model.py --params experiments/model_candidates/vocab_15k.json --class-weights experiments/model_candidates/class_weights.json

# Test sampling strategies (interactive)
python scripts/02_training/01_test_sampling_strategies.py data/02_stg/stg_disaster_response.db

# Run batch experiments
python scripts/02_training/run_batch_experiments.py
```

**Model Evaluation:**
```bash
# Compare models
python scripts/04_evaluation/compare_models.py

# Evaluate hierarchy
python scripts/04_evaluation/evaluate_hierarchy.py
```

**System Validation:**
```bash
python scripts/06_validation/system_validation.py
python scripts/06_validation/validate_multilabel_sampling.py
```

## Script Dependencies

- **Data**: `data/02_stg/stg_disaster_response.db`
- **Parameters**: `model/parameters.json` or experiment-specific configs
- **Class Weights**: `model/class_weights.json` or experiment-specific configs
- **Hyperparameters**: `experiments/configs/hyperparameter_optimization.json`
- **Source Code**: `src/disasterproject/`

## Output Locations

- **Models**: `model/` directory
- **Results**: `data/04_fct/` directory
- **Experiments**: `experiments/` directory
- **Logs**: `app.log` and console output
- **Visualizations**: `images/` directory
