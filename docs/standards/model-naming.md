# ML Model Naming Convention

**Version**: 2.1  
**Date**: 2026-09-21  
**Status**: Active  
**Previous Version**: 2.0 (2026-02-03) - Date-based versioning with `v25-11-06` production example; 1.0 (2025-09-12) used semantic versioning

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
| **version** | Date-based versioning | `v{YY}-{MM}-{DD}` derived from training date | `v26-09-21` |
| **environment** | Deployment target | `prod` (production models) | `prod` |
| **training_date** | Training date (YYYY-MM-DD) | `YYYY-MM-DD` | `2026-09-21` |

**Critical**: The version (`v26-09-21`) and the date field (`2026-09-21`) **must match** - they both refer to the training date. The promotion date is stored separately in `MODEL_INFO.json`.

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

### Current Production Model (2026-09-21)
```
disaster_lr_v26-09-21_prod_2026-09-21.pkl
```
- **Domain**: Disaster response
- **Algorithm**: LogisticRegression (auto-detected)
- **Version**: `v26-09-21` (derived from training date)
- **Environment**: Production
- **Training Date**: September 21, 2026
- **Promotion Date**: September 21, 2026 (stored in `MODEL_INFO.json`)
- **Split contract**: train / cal / eval (thresholds on calibration; metrics on frozen eval)

**Breaking it down**:
- `disaster` - Domain prefix
- `lr` - Algorithm code (LogisticRegression)
- `v26-09-21` - Version derived from training date (2026-09-21 → v26-09-21)
- `prod` - Environment (production)
- `2026-09-21` - Training date (YYYY-MM-DD format)

### Historical Production Example (2026-02-03 → 2026-09-21)
```
disaster_lr_v25-11-06_prod_2025-11-06.pkl
```
- Prior LogisticRegression production artifact (tune-on-eval workflow)
- Metadata archived under `experiments/model_archive/` after the 2026-09-21 promotion
- Keep as a naming example; do not treat as the live operating point

### Version Format Explanation

The version format `v{YY}-{MM}-{DD}` is derived from the training date:
- Training date: `2025-11-06` → Version: `v25-11-06`
- Training date: `2026-09-21` → Version: `v26-09-21`

**Why date-based versioning?**
- Provides clear traceability to training date
- Ensures version and date fields always match
- Simplifies model lineage tracking
- Reduces manual version management errors

### Future Models
```
disaster_lr_v26-10-01_prod_2026-10-01.pkl    # Future production model
disaster_rf_v26-11-01_prod_2026-11-01.pkl    # Future RandomForest model
```

**Note**: Experimental models use directory-based naming in `experiments/experimental_runs/` (e.g., `2026-09-21/`).

## Artifact Naming

Supporting files follow the same base name (model stem) with descriptive suffixes:

```
disaster_lr_v26-09-21_prod_2026-09-21.pkl                                    # Main model
disaster_lr_v26-09-21_prod_2026-09-21_thresholds.json                        # Per-label thresholds (preferred)
disaster_lr_v26-09-21_prod_2026-09-21_labels.json                            # Label ordering
disaster_lr_v26-09-21_prod_2026-09-21_training.json                          # Training log
disaster_lr_v26-09-21_prod_2026-09-21_performance_metrics.csv               # Performance metrics (preferred, model-specific)
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
├── disaster_lr_v26-09-21_prod_2026-09-21.pkl   # Current production (flat model/ layout)
├── disaster_lr_v26-09-21_prod_2026-09-21_*.json
└── MODEL_INFO.json
```

> **Historical note**: Earlier drafts of this standard described `current/` / `staging/` / `archive/` subfolders and Google Drive uploads. Active deploys use the flat `model/` layout above; see [deployment runbook](../runbooks/deployment.md).

## Usage

### Model Promotion Workflow

Models are promoted using the promotion script, which handles naming automatically:

#### 1. Train Experimental Model
```bash
python scripts/02_training/03_create_experimental_model.py \
  --algorithm logistic_regression \
  --params experiments/model_candidates/vocab_15k.json \
  --class-weights experiments/model_candidates/class_weights.json \
  --output experiments/experimental_runs/2026-09-21/lr_vocab15k_model.pkl
```

#### 2. Calibrate Thresholds
```bash
python scripts/03_optimization/optimize_per_category_thresholds.py \
  --model-path experiments/experimental_runs/2026-09-21/lr_vocab15k_model.pkl
```

#### 3. Validate and Promote
```bash
# Dry run (validate without promoting)
python scripts/07_operations/promote_model.py \
  experiments/experimental_runs/2026-09-21 \
  --dry-run

# Actual promotion (algorithm auto-detected, filename auto-generated)
python scripts/07_operations/promote_model.py \
  experiments/experimental_runs/2026-09-21 \
  --print-new-path
```

The promotion script:
- **Auto-detects** algorithm type (rf/lr)
- **Generates** filename from training date
- **Copies** model and metadata files
- **Verifies** file integrity (hash check)
- **Updates** `MODEL_INFO.json` with promotion metadata
- **Archives** previous production model metadata (binaries via Git history)

### Configuration Updates

`app/config.py` auto-discovers the newest `model/*_prod_*.pkl` (or uses the `MODEL_FILENAME` env override). Promotion therefore **does not** rewrite config by default.

- Pass `--update-config` only if you maintain a hardcoded `disaster_*` `MODEL_FILENAME` string literal to rewrite
- `--no-update-config` remains accepted as a no-op alias (skipping is already the default)

**Manual Override**: Set `MODEL_FILENAME` environment variable to use a specific model.

## Deployment Workflow

### 1. Training Phase
```bash
python scripts/02_training/03_create_experimental_model.py \
  --algorithm logistic_regression \
  --params experiments/model_candidates/vocab_15k.json \
  --class-weights experiments/model_candidates/class_weights.json \
  --output experiments/experimental_runs/2026-09-21/lr_vocab15k_model.pkl

python scripts/03_optimization/optimize_per_category_thresholds.py \
  --model-path experiments/experimental_runs/2026-09-21/lr_vocab15k_model.pkl
```

Legacy RandomForest re-runs (not promoted to production) use `scripts/02_training/04_create_production_model.py` and write under `experiments/legacy_rf/`.

### 2. Testing Phase
```bash
# Promote with dry-run gates, then load via app smoke tests
python scripts/07_operations/promote_model.py experiments/experimental_runs/2026-09-21 --dry-run
python scripts/run_tests.py tests/test_app_smoke.py -q
```

### 3. Production Deployment
```bash
# Ensure the promoted pickle is present in model/ (git-tracked for current LR artifact)
ls -lh model/disaster_lr_v26-09-21_prod_2026-09-21.pkl

# Optional: pin an explicit filename instead of auto-discovery
export MODEL_FILENAME=disaster_lr_v26-09-21_prod_2026-09-21.pkl

python run.py
```

> **Historical (retired)**: Older guides uploaded models to Google Drive and set `GDRIVE_MODEL_ID`. The Flask loader no longer downloads from Drive; see ADR-003 and the deployment runbook for archival notes.

## Version Management

### Date-based Versioning (current)

- Version = training date as `v{YY}-{MM}-{DD}`
- Filename date field must match the version
- Promotion timestamp lives in `MODEL_INFO.json` only

### Semantic Versioning Rules (legacy, pre-2025-11)

- **Major (X.0.0)**: Breaking changes, new algorithms, major architecture changes
- **Minor (1.X.0)**: New features, performance improvements, backward compatible
- **Patch (1.2.X)**: Bug fixes, small improvements, hyperparameter tuning

### Examples
```
v1-0-0  # Initial production model (legacy semantic)
v25-11-06  # Historical date-based LR production
v26-09-21  # Current date-based LR production
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
disaster_lr_v26-09-21_prod_2026-09-21.pkl      # Current production (date-based versioning)
disaster_lr_v25-11-06_prod_2025-11-06.pkl      # Prior production (historical)
disaster_rf_v1-2-0_prod_2025-09-11.pkl          # Earlier RF production (semantic versioning, deprecated)
```

**Note**: Models using semantic versioning (`v1-2-0`) are legacy. All new models use date-based versioning (`v26-09-21`).

## Common Pitfalls to Avoid

### ❌ Don't: Use Promotion Date in Filename
**Wrong**: `disaster_lr_v26-09-21_prod_2026-09-22.pkl`
- Version says training date: 2026-09-21
- Filename date says: 2026-09-22 (promotion date)
- **Confusing**: Two different dates!

**Correct**: `disaster_lr_v26-09-21_prod_2026-09-21.pkl`
- Both version and date refer to training date: 2026-09-21
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
