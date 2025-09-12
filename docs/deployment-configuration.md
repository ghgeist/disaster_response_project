# Deployment Configuration Guide

## Deployment Strategy Overview

- **Production**: Google Drive model storage (required)
- **Development**: Google Drive primary, local model fallback

## Configuration by Environment

### 🚀 Production Environment

**Required Environment Variables:**
```bash
# REQUIRED: Google Drive model file ID
GDRIVE_MODEL_ID="1s_sBXnUdJ-rWm4-YEsDixHCbxBca-oXh"

# Production settings
FLASK_ENV=production
SECRET_KEY=your-secure-production-secret
LOG_LEVEL=WARNING
```

**Model Storage:**
- ❌ No local model files in production deployment
- ✅ Model downloaded from Google Drive on startup
- ✅ Cached locally after first download

**Deployment Process:**
```bash
# 1. Set environment variables
export GDRIVE_MODEL_ID="1s_sBXnUdJ-rWm4-YEsDixHCbxBca-oXh"
export FLASK_ENV=production
export SECRET_KEY="your-secure-key"

# 2. Deploy without model files (lightweight deployment)
# Your deployment script here

# 3. Application will download model on first startup
```

### 💻 Development Environment

**Environment Variables:**
```bash
# OPTIONAL: Set to test Google Drive functionality
GDRIVE_MODEL_ID="1s_sBXnUdJ-rWm4-YEsDixHCbxBca-oXh"

# Development settings
FLASK_ENV=development
LOG_LEVEL=DEBUG
```

**Model Storage Priority:**
1. 🥇 **Local model** (if exists): `model/disaster_rf_v1-2-0_prod_2025-09-11.pkl`
2. 🥈 **Google Drive** (if GDRIVE_MODEL_ID set): Downloads from Google Drive
3. 🥉 **Error**: No model available

**Development Scenarios:**

#### Scenario 1: Local Development (Fast)
```bash
# Don't set GDRIVE_MODEL_ID
unset GDRIVE_MODEL_ID

# Uses local model for fast development
python run.py
```

#### Scenario 2: Test Production Behavior
```bash
# Set GDRIVE_MODEL_ID and remove local model
export GDRIVE_MODEL_ID="1s_sBXnUdJ-rWm4-YEsDixHCbxBca-oXh"
mv model/*.pkl model/backup/

# Forces Google Drive download (tests production behavior)
python run.py
```

#### Scenario 3: Hybrid Development
```bash
# Set GDRIVE_MODEL_ID but keep local model
export GDRIVE_MODEL_ID="1s_sBXnUdJ-rWm4-YEsDixHCbxBca-oXh"

# Uses local model (faster) but validates Google Drive configuration
python run.py
```

## Model File Management

### Production Deployment Files

**Include in deployment:**
```
app/
├── config.py
├── services.py
├── routes.py
└── ...

# DO NOT INCLUDE:
# model/*.pkl  ← Exclude from production builds
```

**Exclude from production builds:**
```dockerfile
# Example Dockerfile
COPY . /app
# Exclude model files to keep deployment lightweight
RUN rm -rf /app/model/*.pkl
```

### Development Files

**Keep for development:**
```
model/
├── disaster_rf_v1-2-0_prod_2025-09-11.pkl          # Local development
├── disaster_rf_v1-2-0_prod_2025-09-11_thresholds.json
├── disaster_rf_v1-2-0_prod_2025-09-11_labels.json
└── ...
```

## Environment Variable Configuration

### Option 1: Environment Variables
```bash
export GDRIVE_MODEL_ID="1s_sBXnUdJ-rWm4-YEsDixHCbxBca-oXh"
export FLASK_ENV=production
export SECRET_KEY="your-secure-key"
```

### Option 2: .env File (Development)
```env
# .env file for local development
FLASK_ENV=development
GDRIVE_MODEL_ID=1s_sBXnUdJ-rWm4-YEsDixHCbxBca-oXh
LOG_LEVEL=DEBUG
```

### Option 3: Platform-Specific

#### Heroku
```bash
heroku config:set GDRIVE_MODEL_ID="1s_sBXnUdJ-rWm4-YEsDixHCbxBca-oXh"
heroku config:set FLASK_ENV=production
```

#### Docker
```yaml
# docker-compose.yml
services:
  app:
    environment:
      - GDRIVE_MODEL_ID=1s_sBXnUdJ-rWm4-YEsDixHCbxBca-oXh
      - FLASK_ENV=production
```

## Testing Your Configuration

### Test Google Drive Download
```bash
# Set environment and test
export GDRIVE_MODEL_ID="1s_sBXnUdJ-rWm4-YEsDixHCbxBca-oXh"
python test_gdrive_deployment.py
```

### Test Production Scenario
```bash
# Remove local model and test Google Drive fallback
mv model/*.pkl model/backup/
export GDRIVE_MODEL_ID="1s_sBXnUdJ-rWm4-YEsDixHCbxBca-oXh"
python -c "from app.app import create_app; from app.config import Config; create_app(Config)"
```

### Test Development Scenario
```bash
# Restore local model and test fallback
mv model/backup/*.pkl model/
unset GDRIVE_MODEL_ID
python -c "from app.app import create_app; from app.config import Config; create_app(Config)"
```

## Troubleshooting

### Common Issues

#### "Model file not found and GDRIVE_MODEL_ID not configured"
**Solution**: Set the GDRIVE_MODEL_ID environment variable
```bash
export GDRIVE_MODEL_ID="1s_sBXnUdJ-rWm4-YEsDixHCbxBca-oXh"
```

#### "Google Drive returned HTML instead of the model file"
**Solution**: Ensure the Google Drive file is publicly accessible:
1. Right-click file in Google Drive
2. Select "Get shareable link"  
3. Set to "Anyone with the link can view"

#### "Download timed out"
**Solution**: Check network connectivity and try again. The 32MB model typically downloads in 2-5 seconds.

## Current Configuration

**Google Drive Model:**
- **File ID**: `1s_sBXnUdJ-rWm4-YEsDixHCbxBca-oXh`
- **Model**: `disaster_rf_v1-2-0_prod_2025-09-11.pkl` (32MB)
- **URL**: https://drive.google.com/file/d/1s_sBXnUdJ-rWm4-YEsDixHCbxBca-oXh/view

**Local Model:**
- **Path**: `model/disaster_rf_v1-2-0_prod_2025-09-11.pkl`
- **Size**: 32MB
- **Use**: Development fallback

## Security Notes

- **Google Drive File**: Must be publicly readable (anyone with link)
- **File ID**: Not sensitive, can be stored in environment variables
- **Model Content**: Contains trained ML model, not sensitive data
- **SECRET_KEY**: Keep secure, change default value for production
