# Scripts Directory

This directory contains scripts for training, testing, and analyzing disaster response classification models.

## Core Model Creation Scripts

### `create_baseline_model.py`
Creates a baseline model using default parameters without any class imbalance handling.
- **Use when**: Establishing baseline performance metrics
- **Output**: Baseline model with standard performance metrics
- **Dependencies**: `model/parameters.json`

### `create_weighted_model.py`
Creates a model with class weighting to handle multi-label class imbalance.
- **Use when**: Creating production models with improved minority class detection
- **Output**: Class-weighted model with balanced performance
- **Dependencies**: `model/parameters.json`

## Experimental Scripts

### `test_sampling_strategies.py`
Tests different sampling methods for handling class imbalance.
- **Use when**: Comparing sampling approaches (SMOTE, ADASYN, conservative)
- **Output**: Models trained with different sampling strategies
- **Interactive**: Yes - provides menu for strategy selection

### `test_hyperparameters.py`
Performs hyperparameter optimization using GridSearchCV.
- **Use when**: Finding optimal model parameters
- **Output**: Optimized model with best parameters
- **Dependencies**: `experiments/configs/hyperparameter_optimization.json`

### `run_batch_experiments.py`
Runs multiple experiments in batch mode.
- **Use when**: Testing multiple configurations systematically
- **Output**: Multiple models and comparison results

## Analysis Scripts

### `compare_models.py`
Compares performance between different models.
- **Use when**: Analyzing differences between model versions
- **Output**: Performance comparison reports

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

## Usage Examples

```bash
# Create baseline model
python scripts/create_baseline_model.py --out models/baseline.pkl

# Create weighted model
python scripts/create_weighted_model.py --out models/weighted.pkl

# Test sampling strategies
python scripts/test_sampling_strategies.py data/02_stg/stg_disaster_response.db

# Test hyperparameters
python scripts/test_hyperparameters.py data/02_stg/stg_disaster_response.db models/optimized.pkl

# Compare models
python scripts/compare_models.py models/baseline.pkl models/weighted.pkl
```

## Script Dependencies

- **Data**: `data/02_stg/stg_disaster_response.db`
- **Parameters**: `model/parameters.json`
- **Hyperparameters**: `experiments/configs/hyperparameter_optimization.json`
- **Source Code**: `src/disaster_classifier/`

## Output Locations

- **Models**: `models/` directory
- **Results**: `data/04_fct/` directory
- **Logs**: `app.log` and console output
- **Visualizations**: `images/` directory
