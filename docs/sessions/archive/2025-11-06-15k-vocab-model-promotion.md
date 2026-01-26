---
title: "15K Vocabulary Model Promotion to Production"
date: "2025-11-06"
status: "active"
tags: ["ml", "promotion", "production", "vocabulary-optimization"]
author: "ML Engineer Agent"
related: [
  "docs/sessions/completed/2025-11-06-vocabulary-size-optimization.md"
]
---

# 15K Vocabulary Model Promotion to Production

## Model Summary

**Model**: `lr_vocab15k_model.pkl`  
**Location**: `experiments/experimental_runs/2025-11-06-vocab15k-promotion/`  
**Performance**: F1=93.79%, Critical Recall=64.97%, Size=4.5MB  
**Vocabulary**: max_features=15000, min_df=3, max_df=0.90

## Validation Results ✅

- **F1-weighted**: 0.9379 ✅ (exceeds 0.5 threshold)
- **F1-micro**: 0.6502 ✅ (exceeds 0.6 threshold)
- **Model size**: 4.5MB ✅ (well under 1000MB limit)
- **All validation gates passed**

## Promotion Steps

### 1. Preparation (COMPLETE ✅)

The model has been prepared in:
```
experiments/experimental_runs/2025-11-06-vocab15k-promotion/
├── lr_vocab15k_model.pkl
├── lr_vocab15k_model_thresholds.json (standard naming)
├── optimized_critical_thresholds.json (legacy naming)
├── training_log.json
├── performance_metrics.csv
├── label_order.json
└── PROMOTION_INFO.json
```

### 2. Promote to Production

**With venv activated**, run:

```powershell
# Activate virtual environment
. .venv\Scripts\Activate.ps1

# Promote model (auto-updates app/config.py)
python scripts/promote_model.py experiments/experimental_runs/2025-11-06-vocab15k-promotion --print-new-path

# Or promote without auto-updating config
python scripts/promote_model.py experiments/experimental_runs/2025-11-06-vocab15k-promotion --no-update-config
```

### 3. What Happens During Promotion

1. **Archives current production model** metadata to `experiments/model_archive/`
2. **Copies model** to `model/disaster_lr_v<version>_prod_2025-11-06.pkl`
3. **Copies thresholds** with standard naming: `{model_stem}_thresholds.json`
4. **Creates MODEL_INFO.json** with promotion metadata
5. **Updates app/config.py** (unless `--no-update-config` is used)
6. **Cleans up old models** (keeps 1 by default, configurable with `--keep-old`)

### 4. Post-Promotion Verification

After promotion, verify:

1. **Model file exists**:
   ```powershell
   ls model/disaster_lr_*_prod_*.pkl
   ```

2. **Thresholds file exists**:
   ```powershell
   ls model/*_thresholds.json
   ```

3. **App config updated** (if auto-update was used):
   ```powershell
   # Check app/config.py for MODEL_FILENAME
   ```

4. **Test the app**:
   ```powershell
   python run.py
   # Test a few predictions to ensure thresholds are loaded
   ```

## Expected Production Model Name

The promotion script will generate a name like:
```
disaster_lr_v25-11-06_prod_2025-11-06.pkl
```

With corresponding thresholds:
```
disaster_lr_v25-11-06_prod_2025-11-06_thresholds.json
```

## Performance Comparison

| Metric | Current Production (RF) | New 15K Vocab (LR) | Improvement |
|--------|------------------------|-------------------|-------------|
| F1-weighted | 90.07% | 93.79% | +4.1% |
| Critical Recall | ~0% | 64.97% | +∞ |
| Model Size | 915MB | 4.5MB | **-99.5%** |
| Training Time | ~30min | 24.7s | **-98.6%** |

## Key Benefits

1. **93.3% size reduction** (915MB → 4.5MB)
2. **Critical recall breakthrough** (0% → 65%)
3. **Faster training** (30min → 25s)
4. **Better F1** (90% → 94%)
5. **Optimized thresholds** included

## Rollback Plan

If issues occur:

1. **Restore previous model** from archive:
   ```powershell
   # Check archive
   ls experiments/model_archive/
   
   # Restore model and thresholds from archive record
   ```

2. **Revert app/config.py**:
   ```powershell
   # Restore from backup
   cp app/config.py.bak app/config.py
   ```

3. **Restart app** to load previous model

## Notes

- Thresholds are automatically loaded by the app (standard naming)
- Smart defaults are in place if thresholds file is missing
- All critical categories have optimized thresholds (65% target recall)
- Model uses LogisticRegression (not RandomForest) - ensure app compatibility

---

**Status**: ✅ Ready for promotion  
**Validation**: ✅ All gates passed  
**Next Step**: Run promotion command above

