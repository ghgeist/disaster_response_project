# ML Model Naming Convention

**Version**: 2.0  
**Date**: 2026-02-03  
**Status**: Active  
**Previous Version**: 1.0 (2025-09-12) - Used semantic versioning

## Overview

This document establishes the standardized naming convention for disaster response ML models to ensure consistent model versioning, deployment tracking, and artifact management.

**Important**: As of 2026-02-03, the project uses **date-based versioning** instead of semantic versioning. The version format `v{YY}-{MM}-{DD}` is derived from the training date.

## Naming Format

```
{domain}_{algorithm}_{version}_prod_{training_date}.pkl
```

### Component Definitions

| Component | Description | Valid Values | Example |
|-----------|-------------|--------------|---------|
| **domain** | Business context | `disaster`, `emergency`, `crisis` | `disaster` |
| **algorithm** | ML algorithm family | `rf`, `lr` | `lr` |
| **version** | Date-based versioning | `v{YY}-{MM}-{DD}` derived from training date | `v25-11-06` |
| **environment** | Deployment target | `prod` (production models) | `prod` |
| **training_date** | Training date (YYYY-MM-DD) | `YYYY-MM-DD` | `2025-11-06` |

**Critical**: The version (`v25-11-06`) and the date field (`2025-11-06`) **must match** - they both refer to the training date. The promotion date is stored separately in `MODEL_INFO.json`.

### Algorithm Codes

| Code | Full Name | Description | Detection |
|------|-----------|-------------|-----------|
| `rf` | RandomForest | Random Forest Classifier | Auto-detected during promotion |
| `lr` | LogisticRegression | Logistic Regression | Auto-detected during promotion |

**Algorithm Detection**: The promotion script (`scripts/07_operations/promote_model.py`) automatically detects the algorithm type by inspecting the model file structure. This prevents manual errors and ensures consistency.

### Environment Codes

| Code | Full Name | Description |
|------|-----------|-------------|
| `prod` | Production | Production-ready models |

**Note**: Currently, only production models (`prod`) use this naming convention. Experimental models are stored in `experiments/experimental_runs/` with directory-based naming.

## Examples

### Current Production Model (2026-02-03)
```
disaster_lr_v25-11-06_prod_2025-11-06.pkl
```
- **Domain**: Disaster response
- **Algorithm**: LogisticRegression (auto-detected)
- **Version**: `v25-11-06` (derived from training date)
- **Environment**: Production
- **Training Date**: November 6, 2025
- **Promotion Date**: February 3, 2026 (stored in `MODEL_INFO.json`)

**Breaking it down**:
- `disaster` - Domain prefix
- `lr` - Algorithm code (LogisticRegression)
- `v25-11-06` - Version derived from training date (2025-11-06 → v25-11-06)
- `prod` - Environment (production)
- `2025-11-06` - Training date (YYYY-MM-DD format)

### Version Format Explanation

The version format `v{YY}-{MM}-{DD}` is derived from the training date:
- Training date: `2025-11-06` → Version: `v25-11-06`
- Training date: `2026-01-15` → Version: `v26-01-15`

**Why date-based versioning?**
- Provides clear traceability to training date
- Ensures version and date fields always match
- Simplifies model lineage tracking
- Reduces manual version management errors

### Future Models
```
disaster_lr_v26-02-15_prod_2026-02-15.pkl    # Future production model
disaster_rf_v26-03-01_prod_2026-03-01.pkl    # Future RandomForest model
```

**Note**: Experimental models use directory-based naming in `experiments/experimental_runs/` (e.g., `2025-11-06-vocab15k-promotion/`).

## Artifact Naming

Supporting files follow the same base name (model stem) with descriptive suffixes:

```
disaster_lr_v25-11-06_prod_2025-11-06.pkl                                    # Main model
disaster_lr_v25-11-06_prod_2025-11-06_thresholds.json                        # Per-label thresholds (preferred)
disaster_lr_v25-11-06_prod_2025-11-06_labels.json                            # Label ordering
disaster_lr_v25-11-06_prod_2025-11-06_training.json                          # Training log
disaster_lr_v25-11-06_prod_2025-11-06_performance_metrics.csv               # Performance metrics (preferred, model-specific)
MODEL_INFO.json                                                               # Model metadata (shared, contains promotion info)
```

### File Naming Patterns

**Required Files**:
- `{model_stem}.pkl` - Serialized model file
- `MODEL_INFO.json` - Model metadata (algorithm, version, performance, promotion info)

**Optional Files** (model-specific naming preferred):
- `{model_stem}_thresholds.json` - Per-category classification thresholds (preferred)
- `thresholds.json` - Legacy fallback thresholds file (deprecated)
- `{model_stem}_performance_metrics.csv` - Detailed performance metrics (preferred)
- `performance_metrics.csv` - Legacy fallback metrics file (deprecated)

**Deprecated**: `optimized_critical_thresholds.json` and `optimized_all_thresholds.json` are deprecated. Use model-specific naming instead.

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

### Model Promotion Workflow

Models are promoted using the promotion script, which handles naming automatically:

#### 1. Train Experimental Model
```bash
python scripts/02_training/03_create_experimental_model.py \
  --config experiments/model_candidates/vocab_15k.json \
  --output-dir experiments/experimental_runs/2025-11-06-vocab15k-promotion
```

#### 2. Validate and Promote
```bash
# Dry run (validate without promoting)
python scripts/07_operations/promote_model.py \
  experiments/experimental_runs/2025-11-06-vocab15k-promotion \
  --dry-run

# Actual promotion (algorithm auto-detected, filename auto-generated)
python scripts/07_operations/promote_model.py \
  experiments/experimental_runs/2025-11-06-vocab15k-promotion \
  --print-new-path
```

The promotion script:
- **Auto-detects** algorithm type (rf/lr)
- **Generates** filename from training date
- **Copies** model and metadata files
- **Verifies** file integrity (hash check)
- **Updates** `MODEL_INFO.json` with promotion metadata
- **Archives** previous production model

### Configuration Updates

The promotion script can optionally update `app/config.py` automatically:
- Use `--no-update-config` to skip config updates
- By default, updates `MODEL_FILENAME` to point to the new model

**Manual Override**: Set `MODEL_FILENAME` environment variable to use a specific model.

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

### Legacy State (Pre-2025-09-19)
```
classifier.pkl                    # Generic name
original_classifier.pkl          # Unclear versioning
experimental_classifier.pkl      # No environment distinction
```

### Standardized State (Current)
```
disaster_lr_v25-11-06_prod_2025-11-06.pkl      # Current production (date-based versioning)
disaster_rf_v1-2-0_prod_2025-09-11.pkl          # Previous production (semantic versioning, deprecated)
```

**Note**: Models using semantic versioning (`v1-2-0`) are legacy. All new models use date-based versioning (`v25-11-06`).

## Common Pitfalls to Avoid

### ❌ Don't: Use Promotion Date in Filename
**Wrong**: `disaster_lr_v25-11-06_prod_2026-02-03.pkl`
- Version says training date: 2025-11-06
- Filename date says: 2026-02-03 (promotion date)
- **Confusing**: Two different dates!

**Correct**: `disaster_lr_v25-11-06_prod_2025-11-06.pkl`
- Both version and date refer to training date: 2025-11-06
- Promotion date stored in `MODEL_INFO.json`

### ❌ Don't: Manually Rename Model Files
Always use the promotion script (`scripts/07_operations/promote_model.py`). It:
- Detects algorithm type automatically
- Generates correct filenames
- Updates `MODEL_INFO.json`
- Archives previous models

### ❌ Don't: Hardcode Algorithm Codes
The promotion script detects the algorithm automatically. Don't hardcode `rf` or `lr` in filenames - let the script do it.

### ✅ Do: Check MODEL_INFO.json After Promotion
Verify that:
- `algorithm` field matches the actual model type
- `algorithm_name` is correct
- `promotion_timestamp` reflects when it was promoted
- File hash matches the model file

