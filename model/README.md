# Model Artifacts

This folder stores production model files and their companion metadata used by the Flask app and evaluation scripts.

## Model Naming Convention

**Format**: `disaster_{algorithm}_{version}_prod_{training_date}.pkl`

### Important: Date Field is Training Date, Not Promotion Date

⚠️ **CRITICAL**: The date in the filename (`{training_date}`) refers to **when the model was trained**, NOT when it was promoted to production.

**Example**:
```
disaster_lr_v25-11-06_prod_2025-11-06.pkl
```

Breaking it down:
- `disaster` - Domain prefix
- `lr` - Algorithm code (LogisticRegression)
- `v25-11-06` - Version derived from training date (2025-11-06 → v25-11-06)
- `prod` - Environment (production)
- `2025-11-06` - **Training date** (YYYY-MM-DD format)

**Why this matters**: The version (`v25-11-06`) and the date (`2025-11-06`) should **always match** - they both refer to the training date. The promotion date is stored separately in `MODEL_INFO.json` as `promotion_timestamp`.

### Algorithm Detection

The promotion script (`scripts/07_operations/promote_model.py`) automatically detects the algorithm type from the model file:
- **RandomForest** → `rf`
- **LogisticRegression** → `lr`

The algorithm code is embedded in the filename to prevent confusion. The script validates that the copied model matches the expected algorithm type.

### Version Format

Versions are derived from the training date:
- Training date: `2025-11-06` → Version: `v25-11-06`
- Format: `v{YY}-{MM}-{DD}` (last 2 digits of year, month, day)

## Current Production Model

**Model File**: `disaster_lr_v25-11-06_prod_2025-11-06.pkl`
- **Algorithm**: LogisticRegression
- **Size**: 4.53 MB
- **Performance**: F1-weighted=0.9379, F1-micro=0.6502
- **Training Date**: 2025-11-06
- **Promotion Date**: 2026-02-03 (stored in MODEL_INFO.json)

## Model Discovery

The Flask app uses auto-discovery to find the latest production model:

1. **Pattern Matching**: Looks for files matching `disaster_*_prod_*.pkl`
2. **Sorting**: Sorts by modification time (newest first)
3. **Selection**: Uses the most recently modified file

**Manual Override**: Set `MODEL_FILENAME` environment variable to use a specific model.

## File Structure

### Required Files

- `{model_name}.pkl` - Serialized model file
- `MODEL_INFO.json` - Model metadata (algorithm, version, performance, promotion info)

### Optional Files

- `{model_name}_thresholds.json` - Per-category classification thresholds
- `optimized_critical_thresholds.json` - Legacy thresholds file (fallback)
- `optimized_all_thresholds.json` - All-category thresholds
- `performance_metrics.csv` - Detailed performance metrics

### MODEL_INFO.json Structure

```json
{
  "sha256": "model_file_hash",
  "promoted_from": "experiments/experimental_runs/...",
  "promotion_timestamp": "2026-02-03T12:28:29.427768",
  "model_size_mb": 4.53,
  "algorithm": "lr",
  "algorithm_name": "LogisticRegression",
  "version": "v25-11-06",
  "status": "production",
  "performance": {
    "f1_weighted": 0.9379,
    "f1_micro": 0.6502
  }
}
```

**Key Fields**:
- `algorithm` - Algorithm code (`rf`, `lr`, etc.)
- `algorithm_name` - Full algorithm name
- `version` - Model version (derived from training date)
- `promotion_timestamp` - **When** the model was promoted (ISO format)
- `performance` - Model performance metrics

## Model Promotion Workflow

### 1. Train Experimental Model

```bash
python scripts/02_training/03_create_experimental_model.py \
  --config experiments/model_candidates/vocab_15k.json \
  --output-dir experiments/experimental_runs/2025-11-06-vocab15k-promotion
```

### 2. Validate and Promote

```bash
# Dry run (validate without promoting)
python scripts/07_operations/promote_model.py \
  experiments/experimental_runs/2025-11-06-vocab15k-promotion \
  --dry-run

# Actual promotion
python scripts/07_operations/promote_model.py \
  experiments/experimental_runs/2025-11-06-vocab15k-promotion \
  --print-new-path
```

### 3. What Happens During Promotion

1. **Algorithm Detection**: Script inspects the model file to detect algorithm type
2. **Filename Generation**: Creates filename using training date (from candidate directory name)
3. **File Copy**: Copies model file to `model/` directory
4. **Hash Verification**: Verifies copied file matches expected hash
5. **Metadata Creation**: Creates/updates `MODEL_INFO.json` with algorithm info
6. **Archive**: Archives previous production model metadata

### 4. Verification

After promotion, verify:
- Model file exists and loads correctly
- `MODEL_INFO.json` contains correct algorithm information
- App auto-discovery picks up the new model
- Dashboard displays correct algorithm name

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

Always use the promotion script. It:
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

## Troubleshooting

### Model Not Found

If the app can't find the model:
1. Check that a file matching `disaster_*_prod_*.pkl` exists
2. Verify file permissions
3. Check `MODEL_FILENAME` environment variable (if set)

### Wrong Algorithm Detected

If the promotion script detects the wrong algorithm:
1. Verify the model file is not corrupted
2. Check that the model uses a supported algorithm (RF or LR)
3. Ensure sklearn version compatibility

### Hash Mismatch During Promotion

If promotion fails with hash mismatch:
1. Verify the candidate model file hasn't been modified
2. Check file system for corruption
3. Ensure sufficient disk space

## Script Dependencies (2026-01-22)

**Note**: `model/parameters.json` and `model/class_weights.json` were removed on 2026-01-22.

- **Reason**: These files were defaults for `scripts/04_create_production_model.py` (RandomForest only), but the current production model is LogisticRegression created via `scripts/03_create_experimental_model.py`.
- **Impact**: If you need to run `scripts/04_create_production_model.py`, you must provide `--params` and `--class-weights` arguments pointing to appropriate config files.
- **Current workflow**: Use `scripts/03_create_experimental_model.py` with configs from `experiments/model_candidates/` (e.g., `vocab_15k.json`), then promote via `scripts/07_operations/promote_model.py`.

## Related Documentation

- **Naming Standard**: See `docs/standards/model-naming.md` for full naming convention details
- **Promotion Script**: See `scripts/07_operations/promote_model.py` for promotion implementation
- **Tests**: See `tests/test_promote_model.py` for promotion script tests
