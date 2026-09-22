# Model Artifacts

This folder stores production model files and their companion metadata used by the Flask app and evaluation scripts.

## Quick Navigation

**Common Tasks:**
- **Find current production model**: Look for `disaster_*_prod_*.pkl` files (newest by modification time)
- **Find model metadata**: `MODEL_INFO.json` (contains algorithm, version, performance, promotion date)
- **Find thresholds**: `{model_stem}_thresholds.json` (e.g., `disaster_lr_v26-09-21_prod_2026-09-21_thresholds.json`)
- **Find experimental models**: `experiments/experimental_runs/` (see `experiments/README.md`)
- **Find archived models**: `experiments/model_archive/` (prior production **metadata** + SHA256 records; `.pkl` binaries are not stored there—restore via Git)
- **Promote a model**: Use `scripts/07_operations/promote_model.py`

**Current Production Model:**
- File: `disaster_lr_v26-09-21_prod_2026-09-21.pkl`
- Algorithm: LogisticRegression (vocab15k, three-way train/cal/eval)
- Training Date: 2026-09-21
- Promotion Date: 2026-09-21 (see `MODEL_INFO.json`)
- Operating point: frozen-eval critical recall ≈61.5%, threshold-optimized weighted F1 ≈89.75%, baseline micro F1 ≈64.5%

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

**Model File**: `disaster_lr_v26-09-21_prod_2026-09-21.pkl`
- **Algorithm**: LogisticRegression (vocab15k)
- **Size**: 4.59 MB
- **Evaluation contract**: thresholds tuned on calibration; metrics reported on frozen eval
- **Performance (promoted operating point)**:
  - Eval critical recall ≈ **0.615**
  - Threshold-optimized weighted F1 ≈ **0.8975**
  - Baseline frozen-eval micro F1 ≈ **0.645**
- **Training Date**: 2026-09-21
- **Promotion Date**: 2026-09-21 (stored in `MODEL_INFO.json`)
- **Prior production** (archived metadata): `disaster_lr_v25-11-06_prod_2025-11-06` — historical tune-on-eval ~65% critical recall / 92.76% F1 (not this operating point)

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

- `{model_name}_thresholds.json` - Per-category classification thresholds (preferred)
- `thresholds.json` - Legacy fallback thresholds file
- `{model_name}_performance_metrics.csv` - Detailed performance metrics (preferred, model-specific naming)
- `performance_metrics.csv` - Legacy fallback metrics file (deprecated, use model-specific naming)

**Deprecated**: `optimized_critical_thresholds.json` and `optimized_all_thresholds.json` are deprecated and have been removed. Use model-specific naming (`{model_stem}_thresholds.json`) instead.

### MODEL_INFO.json Structure

```json
{
  "sha256": "model_file_hash",
  "promoted_from": "experiments/experimental_runs/2026-09-21",
  "promotion_timestamp": "2026-09-21T22:36:13.089914",
  "model_size_mb": 4.59,
  "algorithm": "lr",
  "algorithm_name": "LogisticRegression",
  "version": "v26-09-21",
  "status": "production",
  "optimization_split": "calibration",
  "reporting_split": "frozen_eval",
  "performance": {
    "eval_critical_recall": 0.6149,
    "optimized_f1_weighted": 0.8975,
    "baseline_f1_micro": 0.6454
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

Promotion enforces the train/cal/eval **evaluation contract** (see [ADR-007](../docs/adr/adr-007-model-promotion-gating.md)): baseline frozen-eval micro F1, frozen-eval critical recall, size cap, weighted-F1 relative-drop guardrail, and `optimization_split=calibration` / `reporting_split=frozen_eval` provenance. The exact `{model_stem}_thresholds.json` that validation scored is copied **byte-for-byte** to `{prod_model_stem}_thresholds.json` (same SHA256). Promotion must not rewrite that JSON after hashing. Production association is the filename stem + `MODEL_INFO.thresholds_sha256`; inner `metadata.model` may still name the experimental candidate used for calibration.

```bash
# Dry run (validate without promoting) — honest three-way candidate
python scripts/07_operations/promote_model.py \
  experiments/experimental_runs/2026-09-21 \
  --dry-run

# Actual promotion
python scripts/07_operations/promote_model.py \
  experiments/experimental_runs/2026-09-21 \
  --print-new-path
```

Tune-on-eval candidates (for example `2025-11-06-vocab15k-promotion`) fail provenance checks unless `--force` is used intentionally.

### 3. What Happens During Promotion

1. **Algorithm Detection**: Script inspects the model file to detect algorithm type
2. **Evaluation-contract validation**: Baseline micro F1, eval critical recall, size, weighted-F1 relative drop, threshold provenance
3. **Filename Generation**: Creates filename using training date (from candidate directory name)
4. **File Copy**: Copies model file to `model/` directory
5. **Hash Verification**: Verifies copied model matches expected hash
6. **Threshold deploy**: Copies the validated `{model_stem}_thresholds.json` to `{prod_stem}_thresholds.json` and verifies SHA256 identity (bytes unchanged)
7. **Metadata Creation**: Creates/updates `MODEL_INFO.json` with algorithm, contract metrics, and `thresholds_sha256`
8. **Archive**: Copies previous production **metadata** to `experiments/model_archive/` (not the `.pkl`); records SHA256 for Git-history rollback
9. **Cleanup**: Removes superseded `model/*_prod_*.pkl` binaries per `--keep-old` (prefer removing orphan companions too so dashboards cannot mix stems)

### 4. Verification

After promotion, verify:
- Model file exists and loads correctly
- Deployed `{prod_stem}_thresholds.json` SHA matches the candidate thresholds artifact that validation inspected
- `MODEL_INFO.json` contains correct algorithm information and contract metrics
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

- **Reason**: These files were defaults for legacy `scripts/02_training/04_create_production_model.py` (RandomForest only). Production is LogisticRegression via `03_create_experimental_model.py` + `promote_model.py`.
- **Impact**: Legacy RF runs require `--params` and `--class-weights`, default to `experiments/legacy_rf/<date>/<HHMMSS>/`, and refuse `model/` unless `--allow-write-to-model-dir` is set.
- **Current workflow**: Train with `scripts/02_training/03_create_experimental_model.py` (configs from `experiments/model_candidates/`, e.g. `vocab_15k.json`), calibrate thresholds with `scripts/03_optimization/optimize_per_category_thresholds.py`, then promote via `scripts/07_operations/promote_model.py`.

## Related Documentation

- **Naming Standard**: See `docs/standards/model-naming.md` for full naming convention details
- **Promotion Script**: See `scripts/07_operations/promote_model.py` for promotion implementation
- **Tests**: See `tests/test_promote_model.py` for promotion script tests
