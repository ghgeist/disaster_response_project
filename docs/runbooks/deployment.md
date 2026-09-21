# Deployment Configuration Guide

## Deployment Strategy Overview (current as of 2026-09-21)

- **Production / Development**: Load a **local, git-tracked** production pickle from `model/`
- **Discovery**: App auto-selects the newest `disaster_*_prod_*.pkl` (override with `MODEL_FILENAME`)
- **Current artifact**: `disaster_lr_v26-09-21_prod_2026-09-21.pkl` (≈4.59 MB LogisticRegression)

> **Historical (2025-09 → early 2026)**: Production originally used Google Drive downloads via `GDRIVE_MODEL_ID` so large binaries could stay out of the repo. That runtime path has been removed from the Flask app. Keep old Drive IDs and scripts only for archival context — do not configure them for new deployments.

## Configuration by Environment

### Production Environment

**Required / recommended variables:**
```bash
FLASK_ENV=production
SECRET_KEY=your-secure-production-secret
LOG_LEVEL=WARNING
# Optional: pin a specific artifact instead of auto-discovery
# MODEL_FILENAME=disaster_lr_v26-09-21_prod_2026-09-21.pkl
```

**Model storage:**
- ✅ Ship / checkout `model/disaster_lr_v26-09-21_prod_2026-09-21.pkl` and companion files (`_thresholds.json`, etc.)
- ✅ App loads from disk only (no Google Drive download)
- ❌ Do not rely on `GDRIVE_MODEL_ID` — it is ignored by the current loader

**Deployment process:**
```bash
export FLASK_ENV=production
export SECRET_KEY="your-secure-key"
# Ensure model/ contains the production *.pkl from git (or upload it once)
python run.py
# or: gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 wsgi:application
```

### Development Environment

```bash
FLASK_ENV=development
LOG_LEVEL=DEBUG
# unset MODEL_FILENAME to use auto-discovery
```

**Model resolution:**
1. If `MODEL_FILENAME` is set → use `model/<that file>`
2. Else → newest `model/disaster_*_prod_*.pkl` by mtime
3. Else → startup / health checks fail with a clear missing-model error

```bash
# Typical local loop
python run.py
```

## Model File Management

### Include in deployment / checkout

```
model/
├── disaster_lr_v26-09-21_prod_2026-09-21.pkl
├── disaster_lr_v26-09-21_prod_2026-09-21_thresholds.json
├── disaster_lr_v26-09-21_prod_2026-09-21_labels.json
├── disaster_lr_v26-09-21_prod_2026-09-21_performance_metrics.csv
├── disaster_lr_v26-09-21_prod_2026-09-21_training.json
└── MODEL_INFO.json
```

Companion metadata for the prior production artifact may still appear under `experiments/model_archive/` (metadata + SHA256; binaries restored via Git history).

### Replit

1. Import / pull the repo (production pickle is tracked).
2. If the binary is missing, upload `disaster_lr_v26-09-21_prod_2026-09-21.pkl` into `model/`.
3. Ensure `data/02_stg/stg_disaster_response.db` exists.
4. Run via Replit **Autoscale** + Gunicorn (`wsgi:application`).

## Environment Variable Configuration

### Option 1: Environment variables
```bash
export FLASK_ENV=production
export SECRET_KEY="your-secure-key"
# export MODEL_FILENAME=disaster_lr_v26-09-21_prod_2026-09-21.pkl
```

### Option 2: `.env` (development)
```env
FLASK_ENV=development
LOG_LEVEL=DEBUG
SECRET_KEY=dev-only-change-me
```

### Option 3: Platform-specific

#### Heroku
```bash
heroku config:set FLASK_ENV=production
heroku config:set SECRET_KEY="your-secure-key"
# Ensure the slug includes model/*.pkl or provide them via release assets
```

#### Docker
```yaml
# docker-compose.yml
services:
  app:
    environment:
      - FLASK_ENV=production
      - SECRET_KEY=your-secure-key
    volumes:
      - ./model:/app/model:ro
```

## Testing Your Configuration

### Confirm local model load
```bash
python -c "from app.config import Config; from app.services.model_service import ModelService; print(Config.MODEL_PATH); ModelService(Config.MODEL_PATH).load_model(); print('ok')"
```

### Smoke the app factory
```bash
python -c "from app.app import create_app; from app.config import Config; create_app(Config); print('app ok')"
```

## Troubleshooting

### "Model file not found"
**Solution**: Ensure `model/disaster_lr_v26-09-21_prod_2026-09-21.pkl` (or another `disaster_*_prod_*.pkl`) exists, or set `MODEL_FILENAME` to a file that is present.

### scikit-learn / imbalanced-learn install conflicts
**Solution**: Use the ranges in `requirements.txt` (`scikit-learn>=1.7.1,<1.8.0` with `imbalanced-learn>=0.14.0,<0.15.0`). Do not pair sklearn 1.7 with imbalanced-learn 0.12/0.13.

### Stale docs mentioning `GDRIVE_MODEL_ID`
**Solution**: That variable is historical only. See the Historical section below and ADR-003.

## Current Configuration

**Active production model:**
- **Path**: `model/disaster_lr_v26-09-21_prod_2026-09-21.pkl`
- **Size**: ≈4.59 MB
- **Algorithm**: LogisticRegression (vocab15k, train/cal/eval)
- **Load path**: Local disk via auto-discovery

## Historical: Google Drive hybrid deployment (archived)

The following documents the **retired** 2025 hybrid strategy for auditability. Do not use these values for new deploys.

```bash
# Historical only — ignored by current ModelLoader
GDRIVE_MODEL_ID="1s_sBXnUdJ-rWm4-YEsDixHCbxBca-oXh"
```

- **Intent**: Keep ~32 MB RF binaries out of git; download on first startup
- **Prior Drive-hosted example**: `disaster_rf_v1-2-0_prod_2025-09-11.pkl` (32 MB)
- **Prior LR prod (pre-three-way)**: `disaster_lr_v25-11-06_prod_2025-11-06.pkl`
- **Current replacement**: git-tracked `disaster_lr_v26-09-21_prod_2026-09-21.pkl`

See [ADR-003](../adr/adr-003-hybrid-model-deployment-strategy.md) for the original decision and the 2026-09-21 amendment.

## Security Notes

- **SECRET_KEY**: Keep secure; change the default for production
- **Model content**: Trained ML weights — not credentials, but treat deployment integrity seriously
- **Google Drive (historical)**: Public link sharing was acceptable for non-sensitive model binaries only
