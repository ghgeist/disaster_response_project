# ML Model Naming Convention

**Version**: 1.0  
**Date**: 2025-09-12  
**Status**: Active

## Overview

This document establishes the standardized naming convention for disaster response ML models to ensure consistent model versioning, deployment tracking, and artifact management.

## Naming Format

```
{domain}_{algorithm}_{version}_{environment}_{date}.pkl
```

### Component Definitions

| Component | Description | Valid Values | Example |
|-----------|-------------|--------------|---------|
| **domain** | Business context | `disaster`, `emergency`, `crisis` | `disaster` |
| **algorithm** | ML algorithm family | `rf`, `lr`, `tfidf`, `xgb`, `bert` | `rf` |
| **version** | Semantic versioning | `v1-0-0`, `v2-1-3` | `v1-2-0` |
| **environment** | Deployment target | `prod`, `stg`, `dev`, `exp` | `prod` |
| **date** | Training date | `YYYY-MM-DD` | `2025-09-11` |

### Algorithm Codes

| Code | Full Name | Description |
|------|-----------|-------------|
| `rf` | RandomForest | Random Forest Classifier |
| `lr` | LogisticRegression | Logistic Regression |
| `tfidf` | TF-IDF + LogisticRegression | TF-IDF Vectorizer + Logistic Regression |
| `xgb` | XGBoost | Gradient Boosting |
| `bert` | BERT-based | BERT or transformer models |

### Environment Codes

| Code | Full Name | Description |
|------|-----------|-------------|
| `prod` | Production | Production-ready models |
| `stg` | Staging/UAT | User acceptance testing |
| `dev` | Development | Development testing |
| `exp` | Experimental | Research/experimental models |

## Examples

### Current Production Model
```
disaster_rf_v1-2-0_prod_2025-09-11.pkl
```
- **Domain**: Disaster response
- **Algorithm**: RandomForest
- **Version**: 1.2.0
- **Environment**: Production
- **Training Date**: September 11, 2025

### Future Models
```
disaster_tfidf_v2-0-0_stg_2025-09-13.pkl    # Next staging deployment
disaster_lr_v1-0-0_exp_2025-09-12.pkl       # Experimental lightweight model
```

## Artifact Naming

Supporting files follow the same base name with descriptive suffixes:

```
disaster_rf_v1-2-0_prod_2025-09-11.pkl              # Main model
disaster_rf_v1-2-0_prod_2025-09-11_thresholds.json  # Per-label thresholds
disaster_rf_v1-2-0_prod_2025-09-11_labels.json      # Label ordering
disaster_rf_v1-2-0_prod_2025-09-11_metadata.json    # Model metadata
disaster_rf_v1-2-0_prod_2025-09-11_metrics.csv      # Performance metrics
disaster_rf_v1-2-0_prod_2025-09-11_training.json    # Training log
disaster_rf_v1-2-0_prod_2025-09-11_weights.json     # Class weights
disaster_rf_v1-2-0_prod_2025-09-11_params.json      # Hyperparameters
```

## Directory Organization

```
model/
├── current/                                          # Current production
│   └── disaster_rf_v1-2-0_prod_2025-09-11.pkl      
├── staging/                                          # Next deployment
│   └── disaster_tfidf_v2-0-0_stg_2025-09-13.pkl    
└── archive/                                          # Previous versions
    ├── disaster_rf_v1-1-0_prod_2025-09-04.pkl      
    └── disaster_rf_v1-0-0_prod_2025-08-15.pkl      
```

## Usage

### Using the Naming Utility

The project includes a naming utility script at `scripts/model_naming_utility.py`:

#### Generate New Model Name
```bash
python scripts/model_naming_utility.py --generate-name --algorithm rf --version 1.3.0 --environment prod
```

#### Rename Existing Model
```bash
# Dry run (shows what would be renamed)
python scripts/model_naming_utility.py --rename-current

# Execute the rename
python scripts/model_naming_utility.py --rename-current --execute
```

### Configuration Updates

When deploying a new model, update the Flask configuration:

```python
# app/config.py
MODEL_FILENAME = 'disaster_rf_v1-2-0_prod_2025-09-11.pkl'
MODEL_PATH = MODELS_DIR / MODEL_FILENAME
```

## Deployment Workflow

### 1. Training Phase
```bash
# Train model with experiments framework
python scripts/04_create_production_model.py

# Generate standardized name
python scripts/model_naming_utility.py --generate-name --algorithm rf --version 1.3.0
```

### 2. Testing Phase  
```bash
# Rename for testing
python scripts/model_naming_utility.py --rename-current --execute

# Test locally
python test_standardized_model.py
```

### 3. Production Deployment
```bash
# Upload to Google Drive with standardized name
# disaster_rf_v1-3-0_prod_2025-09-12.pkl

# Update environment variable
export GDRIVE_MODEL_ID="your_new_file_id"

# Test Google Drive deployment
python test_gdrive_deployment.py
```

## Version Management

### Semantic Versioning Rules

- **Major (X.0.0)**: Breaking changes, new algorithms, major architecture changes
- **Minor (1.X.0)**: New features, performance improvements, backward compatible
- **Patch (1.2.X)**: Bug fixes, small improvements, hyperparameter tuning

### Examples
```
v1-0-0  # Initial production model
v1-1-0  # Improved performance, same architecture  
v1-1-1  # Bug fix for threshold handling
v2-0-0  # New algorithm (RandomForest → TF-IDF)
```

## Benefits

### For Development Team
- **Clear History**: Easy to track model evolution
- **Quick Identification**: Instantly know algorithm, version, environment
- **Deployment Safety**: Clear distinction between prod/staging/experimental

### For Operations Team  
- **Rollback Clarity**: Easy to identify previous stable versions
- **Environment Tracking**: No confusion between staging/production models
- **Artifact Management**: All related files clearly grouped

### For Business Team
- **Performance Tracking**: Link versions to performance improvements
- **Release Planning**: Clear version progression for roadmap planning
- **Compliance**: Audit trail of model changes and deployments

## Migration from Legacy Naming

### Current State
```
classifier.pkl                    # Generic name
original_classifier.pkl          # Unclear versioning
experimental_classifier.pkl      # No environment distinction
```

### Standardized State
```
disaster_rf_v1-2-0_prod_2025-09-11.pkl      # Clear, informative
disaster_rf_v1-1-0_prod_2025-09-04.pkl      # Previous version
disaster_tfidf_v2-0-0_exp_2025-09-12.pkl    # Experimental next-gen
```

