# Gitignore Model Tracking Policy

## Policy

**Track production models, ignore experimental models.**

## Rationale

1. **Production models are small** (4.5MB vs 915MB for old RF model)
2. **Reproducibility**: Production models should be version controlled for deployment consistency
3. **Deployment**: Having production models in git enables easy deployment without external storage
4. **Experimental models are large**: Keep repo clean by ignoring experimental runs

## Implementation

### .gitignore Rules

```gitignore
# Ignore all .pkl files by default
*.pkl

# BUT: Track production models (small, important for reproducibility)
# Production models are named: *_prod_*.pkl
!model/*_prod_*.pkl

# Also track production threshold files (small JSON, critical for deployment)
!model/*_prod_*_thresholds.json
```

### What Gets Tracked

✅ **Tracked**:
- `model/disaster_lr_v25-11-06_prod_2025-11-06.pkl` (4.5MB)
- `model/disaster_lr_v25-11-06_prod_2025-11-06_thresholds.json`
- Any file matching `model/*_prod_*.pkl` pattern
- Any file matching `model/*_prod_*_thresholds.json` pattern

❌ **Ignored**:
- `experiments/experimental_runs/**/*.pkl` (experimental models)
- `model/backups/**/*.pkl` (backup models)
- Any `.pkl` file not matching `*_prod_*` pattern

## Size Considerations

### Current Production Model
- **Size**: 4.5MB (vs 915MB for old RF model)
- **Impact**: Minimal on repo size
- **Benefit**: High (reproducibility, easy deployment)

### Size Limits

If production models exceed **10MB**, consider:
1. Using Git LFS (Large File Storage)
2. External storage (Google Drive, S3)
3. Re-evaluating model size optimization

### Monitoring

Track repo size over time:
```bash
git count-objects -vH
du -sh .git
```

## Benefits

1. **Reproducibility**: Anyone can clone and run with exact production model
2. **Deployment**: No external dependencies for model loading
3. **Version Control**: Model changes tracked alongside code changes
4. **Rollback**: Easy to revert to previous production model
5. **CI/CD**: Automated testing with production models

## Trade-offs

### Pros
- ✅ Easy deployment (no external storage needed)
- ✅ Full reproducibility
- ✅ Version control for models
- ✅ Small size (4.5MB is manageable)

### Cons
- ⚠️ Repo size grows with each production model
- ⚠️ Need to clean up old production models periodically
- ⚠️ If models get large (>10MB), may need Git LFS

## Maintenance

### Cleanup Old Production Models

Keep only the current and previous production model:

```powershell
# List production models
ls model/*_prod_*.pkl

# Remove old ones (keep 2 most recent)
# Promotion script handles this automatically with --keep-old flag
```

### Adding New Production Models

Production models are automatically tracked when:
1. Named with `_prod_` pattern
2. Placed in `model/` directory
3. Promotion script creates them

No manual `.gitignore` changes needed.

## Related

- Model Naming Standard: `docs/standards/model-naming.md`
- Promotion Workflow: `scripts/README.md`
- ADR-003: Hybrid Model Deployment Strategy

