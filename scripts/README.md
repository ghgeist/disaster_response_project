# Scripts Directory

This directory contains scripts for training, testing, and analyzing disaster response classification models.

## Core Model Creation Scripts

### `create_production_model.py`
Creates a production disaster response classification model with class weighting.
- **Use when**: Creating the main production model for deployment
- **Output**: Production model (`model/disaster_rf_v1-2-0_prod_2025-09-11.pkl`), performance metrics, and training logs
- **Dependencies**: `model/parameters.json`, `model/class_weights.json`

## Experimental Scripts

### `test_sampling_strategies.py`
Tests different sampling methods for handling class imbalance with interactive menu.
- **Use when**: Comparing sampling approaches (SMOTE, ADASYN, conservative)
- **Output**: Models trained with different sampling strategies
- **Interactive**: Yes - provides menu for strategy selection

### `test_hyperparameters.py`
Performs hyperparameter optimization using GridSearchCV.
- **Use when**: Finding optimal model parameters
- **Output**: Optimized model with best parameters
- **Dependencies**: `experiments/configs/hyperparameter_optimization.json`

### `run_batch_experiments.py`
Runs multiple experiments in batch mode without user interaction.
- **Use when**: Testing multiple configurations systematically
- **Output**: Multiple models and comparison results
- **Runs**: baseline, smote, adasyn, and conservative experiments

## Analysis Scripts

### `compare_models.py`
Compares performance between different experiment results.
- **Use when**: Analyzing differences between model versions
- **Output**: Performance comparison reports
- **Dependencies**: Experiment tracker system

### `compare_csv_models.py`
Enhanced model comparison tool for CSV prediction results.
- **Use when**: Comparing models using saved CSV results
- **Output**: Detailed performance comparison reports
- **Features**: Crystal-clear comparison for portfolio reviewers

### `validate_multilabel_sampling.py`
Validates multi-label sampling implementations.
- **Use when**: Testing sampling method correctness
- **Output**: Validation reports and performance metrics

### `visualize_performance.py`
Creates performance visualizations.
- **Use when**: Generating charts and graphs for analysis
- **Output**: Performance visualization files

## Utility Scripts

### `prepare_data.py`
Prepares and preprocesses data for training.
- **Use when**: Data preprocessing and preparation
- **Output**: Processed data ready for training

### `system_validation.py`
Validates system components and dependencies.
- **Use when**: Checking system health and dependencies
- **Output**: Validation reports

### `eda_functions.py`
Contains exploratory data analysis functions.
- **Use when**: Data exploration and analysis
- **Output**: Analysis results and insights

## Archive Directory

The `archive/` directory contains legacy scripts that are no longer actively used:
- `compare_results.py` - Legacy result comparison
- `run_all_experiments.py` - Legacy experiment runner
- `systematic_testing_framework.py` - Legacy testing framework
- `train_classifier_original.py` - Original training script
- `train_classifier.py` - Legacy training script
- `validate_structure.py` - Legacy structure validation

## Usage Examples

```bash
# Create production model
python scripts/create_production_model.py

# Test sampling strategies (interactive)
python scripts/test_sampling_strategies.py data/02_stg/stg_disaster_response.db

# Test hyperparameters
python scripts/test_hyperparameters.py data/02_stg/stg_disaster_response.db model/optimized.pkl

# Run batch experiments
python scripts/run_batch_experiments.py

# Compare models from experiments
python scripts/compare_models.py experiment1 experiment2

# Compare CSV results
python scripts/compare_csv_models.py results1.csv results2.csv
```

## Script Dependencies

- **Data**: `data/02_stg/stg_disaster_response.db`
- **Parameters**: `model/parameters.json`
- **Class Weights**: `model/class_weights.json`
- **Hyperparameters**: `experiments/configs/hyperparameter_optimization.json`
- **Source Code**: `src/disasterproject/`

## Output Locations

- **Models**: `model/` directory
- **Results**: `data/04_fct/` directory
- **Experiments**: `experiments/` directory
- **Logs**: `app.log` and console output
- **Visualizations**: `images/` directory
