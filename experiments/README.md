# Experiments

This folder contains experimental model runs, configurations, and comparison results.

## Quick Navigation

**Common Tasks:**
- **Find latest experiment**: `experimental_runs/` sorted by date (newest first)
- **Find production models**: `model/` directory (see `model/README.md`)
- **Find tested hyperparameters**: `model_candidates/` (ready-to-use configs)
- **Find search space templates**: `experimental_configs/hyperparameters/` (INPUT definitions)
- **Compare models**: `comparisons/` (timestamped reports)
- **Production history**: `model_archive/` (archived production models)

**Where are model files?**
- Experimental models: Saved to paths specified when training (typically in `experimental_runs/{date}/` or `experiments/results/`)
- Production models: `model/` directory (see `model/README.md`)

## Folder Structure

### `experimental_runs/{YYYY-MM-DD}/`
Dated experiment results. Each folder contains models, metrics, logs, and reports from that date's experiments. Self-contained and organized by date.

**Naming Convention:**
- Standard format: `YYYY-MM-DD` (e.g., `2025-11-06/`)
- For special cases, use suffixes: `YYYY-MM-DD-{description}` (e.g., `2025-11-06-vocab15k-promotion/`)
- Legacy exception: `2024/` folder uses old format from project initialization (contains early project experiments with different naming conventions)

**Subfolders:**
- Use subfolders for related experiments on the same date (e.g., `hierarchy_initial/`, `hierarchy_optimized/`, `vocab15k/`, `vocab20k/`)
- Each subfolder should contain complete artifacts for that sub-experiment

**Common Files:**
- `MODEL_INFO.json` - Model metadata and training details
- `PROMOTION_INFO.json` - Promotion metadata (if promoted to production)
- `performance_metrics.csv` - Evaluation metrics
- `training_log.json` - Training process logs
- `*_comparison_report.md` - Detailed experiment summaries
- `vocabulary_comparison_report.md` - Vocabulary experiment summaries
- `*.pkl` - Serialized model files (saved to path specified during training, typically in the experiment directory)
- `{model_stem}_thresholds.json` - Per-category threshold optimizations (preferred naming)
- `optimized_critical_thresholds.json` - **Deprecated** legacy threshold file name (still supported for backward compatibility)

**Example Structure:**
```
experimental_runs/2025-11-06-vocab15k-promotion/
  ├── lr_vocab15k_model.pkl              # Model file (if saved)
  ├── lr_vocab15k_model_thresholds.json  # Thresholds (standard naming)
  ├── MODEL_INFO.json                     # Model metadata
  ├── PROMOTION_INFO.json                 # Promotion details
  ├── performance_metrics.csv             # Evaluation metrics
  └── training_log.json                   # Training logs
```

### `model_candidates/`
**Optimized hyperparameter configurations and ready-to-use model parameter sets.**

Contains:
- **Optimized hyperparameters** from grid searches (e.g., `2025-09-16-comprehensive-grid-search-optimized-hyperparameters.json`)
  - These are OUTPUT files from hyperparameter optimization
  - Format: Single best values (e.g., `"n_estimators": 100`)
  - Includes metadata: best score, experiment ID, search method
- **Vocabulary configurations** (e.g., `vocab_15k.json`, `vocab_20k.json`, `vocab_25k.json`)
- **Class weight configurations** (e.g., `class_weights.json`)
- **Other tested parameter sets** ready for model training

**Usage:** These files are used as INPUT by:
- `scripts/02_training/03_create_experimental_model.py`
- `scripts/02_training/04_create_production_model.py`

**Note:** These are specific tested parameter sets, not reusable templates. For reusable experiment templates, see `experimental_configs/`.

### `experimental_configs/`
**Reusable experiment configurations and templates (INPUT definitions).**

Contains:
- **`hyperparameters/`** - Grid search space definitions
  - These are INPUT files for hyperparameter optimization
  - Format: Lists of values to try (e.g., `"n_estimators": [100, 200]`)
  - Includes optimization config (scoring metrics, refit metric)
  - Used by: `scripts/02_training/02_test_hyperparameters.py`
  - Example: `2025-09-16_comprehensive-grid-search.json`
- **`sampling_strategies/`** - Data sampling strategy definitions
- **`eval_sets/`** - Evaluation dataset identifiers (frozen test sets)
- **`optimization_metrics/`** - Metric definitions for optimization (currently empty)

**Key Distinction:**
- **`experimental_configs/`** = Reusable templates/definitions (search spaces, strategies, eval sets)
  - INPUT files: Define what to search (lists of values)
  - Example: `"n_estimators": [100, 200, 300]` (search space)
- **`model_candidates/`** = Specific tested/optimized parameter sets ready for training
  - OUTPUT files: Best values found from optimization
  - Example: `"n_estimators": 200` (single best value)

**Visual Flow:**
```
experimental_configs/hyperparameters/  →  [Grid Search]  →  model_candidates/
     (INPUT: search space)                    (process)         (OUTPUT: best params)
```

### `comparisons/`
Timestamped model comparison reports showing performance differences between models.

Format: `{YYYY-MM-DD}_{HHMMSS}_model_comparison.txt`

### `model_archive/`
Archived production models and their metadata. Contains:
- `archive_record_{model_name}_{training_date}_{archive_date}_{timestamp}.json` - Archive metadata
  - Format breakdown: `archive_record_disaster_rf_prod_2026-01-22_2026-02-03_12-27-35.json`
    - `{model_name}`: Full model filename (e.g., `disaster_rf_prod_2026-01-22`)
    - `{training_date}`: When model was trained (YYYY-MM-DD)
    - `{archive_date}`: When model was archived (YYYY-MM-DD)
    - `{timestamp}`: Exact archive time (HH-MM-SS)
- `MODEL_INFO_{model_name}.json` - Model information snapshots
- `{model_name}_training.json` - Training configuration snapshots
- `{model_name}_thresholds.json` - Threshold configurations
- `{model_name}_labels.json` - Label order snapshots
- `promotion_record_{date}_{timestamp}.json` - Promotion history

### `logs/`
Training and execution logs from experimental runs.

Format: `{YYYY-MM-DD}_{description}.log`

### `results/`
**Legacy folder** - Currently empty. Reserved for backward compatibility with older scripts.

## Workflow: Hyperparameter Optimization

The typical workflow for hyperparameter optimization:

1. **Define search space** → Create/edit config in `experimental_configs/hyperparameters/`
   ```json
   {
     "vect__ngram_range": [[1, 1], [1, 2]],  // Lists = search space
     "clf__estimator__n_estimators": [100, 200],
     "optimization_config": { "refit_metric": "f1_weighted" }
   }
   ```

2. **Run grid search** → `scripts/02_training/02_test_hyperparameters.py`
   - Reads from: `experimental_configs/hyperparameters/{config}.json`
   - Performs RandomizedSearchCV or GridSearchCV
   - Saves detailed results to: `experiments/results/{config}-detailed-results.json`

3. **Save optimized parameters** → Automatically saved to `model_candidates/`
   - Output file: `{config}-optimized-hyperparameters.json`
   - Format: Single best values (e.g., `"n_estimators": 100`)
   - Includes metadata: best score, experiment ID, search method

4. **Train model with optimized params** → Use the optimized config
   ```bash
   python scripts/02_training/03_create_experimental_model.py \
     --params experiments/model_candidates/2025-09-16-comprehensive-grid-search-optimized-hyperparameters.json
   ```

## Conventions

1. **Date-based organization**: Each dated folder in `experimental_runs/` is self-contained with all artifacts from that experiment date.

2. **Metadata files**: 
   - Look for `MODEL_INFO.json` for model metadata
   - Look for `PROMOTION_INFO.json` for promotion details
   - Look for `*_comparison_report.md` or `vocabulary_comparison_report.md` for detailed experiment summaries

3. **File naming**: 
   - Use hyphens for dates: `YYYY-MM-DD` (consistent throughout)
   - Use underscores for descriptive names: `model_info.json`
   - Include dates in filenames when helpful: `2025-11-06_performance_metrics.csv`
   - Config files use underscores: `2025-09-16_comprehensive-grid_search.json`
   - Optimized params use hyphens: `2025-09-16-comprehensive-grid-search-optimized-hyperparameters.json`
   - **Threshold files**: Use `{model_stem}_thresholds.json` (e.g., `lr_vocab15k_model_thresholds.json`)
     - Legacy `optimized_critical_thresholds.json` is deprecated but still supported

4. **Sub-experiments**: Use subfolders within dated runs for related experiments (e.g., vocabulary size comparisons, threshold optimizations).

5. **Input vs Output**:
   - **INPUT** (definitions/templates): `experimental_configs/`
   - **OUTPUT** (optimized/tested sets): `model_candidates/`

## Quick Reference

- **Find latest experiment**: Look in `experimental_runs/` sorted by date (newest first)
- **Find hyperparameters**: 
  - Check `model_candidates/` for tested/optimized sets (ready to use)
  - Check `experimental_configs/hyperparameters/` for search space templates
- **Compare models**: Check `comparisons/` for timestamped comparison reports
- **Production history**: Check `model_archive/` for archived production models
- **Run hyperparameter search**: Use configs from `experimental_configs/hyperparameters/` with `scripts/02_training/02_test_hyperparameters.py`

