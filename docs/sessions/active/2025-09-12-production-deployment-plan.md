---
title: "Production Deployment: Hybrid Model Deployment Strategy"
date: "2025-09-12"
status: "active"
session_type: "deploy"
priority: "high"
tags: ["deployment", "production", "ml", "disaster-response", "hybrid-deployment"]
author: "deployment-team"
related: ["docs/adr/adr-003-hybrid-model-deployment-strategy.md", "docs/sessions/completed/2025-09-03-execute-ml-optimization-COMPLETED.md"]
---

# Production Deployment: Hybrid Model Deployment Strategy

**Session Type**: DEPLOY  
**Priority**: High  
**Estimated Duration**: 1-2 days  
**Status**: Active  
**Architecture**: Hybrid Google Drive + Local Fallback

## 🎯 Deployment Objective

Deploy the current production-ready RandomForest model (`disaster_rf_v1-2-0_prod_2025-09-11.pkl`) using the hybrid deployment strategy with Google Drive storage and module compatibility layer.

## 📊 Current Production Model Status

| Specification | Details |
|---------------|---------|
| **Model File** | `disaster_rf_v1-2-0_prod_2025-09-11.pkl` |
| **Algorithm** | RandomForest with TF-IDF |
| **Model Size** | 32MB |
| **Load Time** | <100ms (cached), 2-5s (Google Drive download) |
| **F1-Score** | 0.9357 (validated) |
| **Architecture** | Hybrid deployment with module compatibility layer |
| **Storage** | Google Drive (production), Local fallback (development) |
| **Module Compatibility** | ✅ Handles `disaster_classifier` → `disasterproject` mapping |

## 🚀 Phase 1: Pre-Deployment Validation (Day 1)

### ✅ Hybrid Deployment Readiness Checklist

- [ ] **Model Artifacts Verified**
  ```bash
  # Verify standardized model naming and artifacts
  ls -la model/disaster_rf_v1-2-0_prod_2025-09-11*
  # Expected files:
  # - disaster_rf_v1-2-0_prod_2025-09-11.pkl (32MB)
  # - disaster_rf_v1-2-0_prod_2025-09-11_thresholds.json
  # - disaster_rf_v1-2-0_prod_2025-09-11_labels.json
  # - disaster_rf_v1-2-0_prod_2025-09-11_metadata.json
  ```

- [ ] **Google Drive Configuration Validated**
  ```bash
  # Test Google Drive model download
  export GDRIVE_MODEL_ID="1s_sBXnUdJ-rWm4-YEsDixHCbxBca-oXh"
  python test_gdrive_deployment.py
  ```

- [ ] **Module Compatibility Layer Test**
  ```bash
  # Test disaster_classifier → disasterproject module mapping
  python -c "
  from app.services import ModelService
  from pathlib import Path
  service = ModelService(Path('model/disaster_rf_v1-2-0_prod_2025-09-11.pkl'))
  print('✅ Module compatibility layer working')
  "
  ```

- [ ] **Deployment Scenarios Validation**
  ```bash
  # Test all deployment scenarios
  python scripts/test_deployment_scenarios.py
  ```

- [ ] **Load Time Benchmark**
  ```bash
  # Test both local and Google Drive load times
  python -c "
  import time
  from app.services import ModelService
  
  # Test local load
  start = time.time()
  service = ModelService()
  print(f'Local load time: {time.time()-start:.3f}s')
  "
  ```

### 🧪 Critical Test Cases

- [ ] **Negation Handling** (3/6 currently passing - acceptable for v1)
- [ ] **High-Impact Labels** (All 8 labels have non-zero recall)
- [ ] **API Response Time** (<200ms end-to-end)
- [ ] **Concurrent Load** (Handle 10+ simultaneous requests)

## 🚀 Phase 2: Hybrid Deployment Execution (Day 2)

### 1. **Environment Configuration**
```bash
# Set production environment variables
export GDRIVE_MODEL_ID="1s_sBXnUdJ-rWm4-YEsDixHCbxBca-oXh"
export FLASK_ENV=production
export SECRET_KEY="your-secure-production-secret"
export LOG_LEVEL=WARNING

# Verify configuration
echo "GDRIVE_MODEL_ID: $GDRIVE_MODEL_ID"
echo "Environment configured for hybrid deployment"
```

### 2. **Google Drive Model Verification**
```bash
# Verify Google Drive model is accessible and current
python -c "
import requests
import os

gdrive_id = os.environ.get('GDRIVE_MODEL_ID')
url = f'https://drive.google.com/uc?id={gdrive_id}'
response = requests.head(url, allow_redirects=True)
print(f'✅ Google Drive model accessible: {response.status_code == 200}')
print(f'Content-Length: {response.headers.get(\"Content-Length\", \"Unknown\")} bytes')
"
```

### 3. **Production Deployment (Lightweight)**
```bash
# Deploy application WITHOUT model files (hybrid deployment)
# Model will be downloaded from Google Drive on startup

# Example deployment commands (adjust for your platform):
# Docker:
docker build -t disaster-response-app .
docker run -e GDRIVE_MODEL_ID="1s_sBXnUdJ-rWm4-YEsDixHCbxBca-oXh" -p 3000:3000 disaster-response-app

# Or Heroku:
# heroku config:set GDRIVE_MODEL_ID="1s_sBXnUdJ-rWm4-YEsDixHCbxBca-oXh"
# git push heroku main

# Or systemd:
# sudo systemctl restart disaster-response-app
```

### 4. **Model Loading Verification**
```bash
# Test model loads successfully with module compatibility layer
python -c "
from app.services import ModelService
import os

# Verify environment
print(f'GDRIVE_MODEL_ID: {os.environ.get(\"GDRIVE_MODEL_ID\")}')

# Test model service initialization
service = ModelService()
result = service.predict('Emergency: people trapped in building')
print('✅ Model loads and predicts successfully')
print(f'Prediction result: {result}')
"
```

### 5. **Production Smoke Test**
```bash
# Test critical endpoints with hybrid deployment
curl -X POST http://your-production-url/classify \
  -H "Content-Type: application/json" \
  -d '{"message": "People trapped on roof. Send search and rescue."}'

# Expected responses:
# - search_and_rescue=True 
# - Response time <200ms (first request may take 2-5s for model download)
# - Subsequent requests <100ms (cached model)

# Test health endpoint
curl -X GET http://your-production-url/health
# Expected: {"status": "healthy", "model_loaded": true}
```

### 6. **Deployment Validation**
```bash
# Run comprehensive deployment validation
python scripts/deployment_health_check.py --production

# Verify all deployment scenarios work
python scripts/test_deployment_scenarios.py --validate-production
```

## 🚀 Phase 3: Production Monitoring (Day 3)

### 📊 Key Metrics to Monitor

1. **Hybrid Deployment Performance Metrics**
   - **First Request Response Time**: Target <5s (includes Google Drive download)
   - **Cached Response Time**: Target <200ms (local model access)
   - **Model Load Success Rate**: Target 100% (Google Drive availability)
   - **Memory usage**: Target <150MB (32MB model + application overhead)
   - **CPU utilization**: Monitor for spikes during model loading
   - **Error rate**: Target <0.1%

2. **Model Quality Metrics**
   - **F1-Score consistency**: Maintain ~0.9357 performance
   - **Critical label activation**: All 8 labels show non-zero predictions
   - **Prediction confidence scores**: Monitor distribution
   - **Module compatibility**: Zero errors from disaster_classifier mapping

3. **Infrastructure Metrics**
   - **Google Drive download success rate**: Target >99%
   - **Network connectivity**: Monitor for timeouts
   - **Fallback scenarios**: Track local vs remote model usage
   - **Module compatibility layer**: Monitor performance impact

### 🔍 Hybrid Deployment Monitoring Setup

```python
# Add to your monitoring system
CRITICAL_LABELS = [
    'medical_help', 'search_and_rescue', 'water', 'food',
    'shelter', 'hospitals', 'security', 'weather_related'
]

# Hybrid deployment alert thresholds
ALERTS = {
    'first_request_response_ms': 5000,  # Alert if first request >5s
    'cached_response_ms': 200,  # Alert if cached response >200ms
    'model_load_failures': 1,  # Alert on any model load failure
    'memory_usage_mb': 200,  # Alert if >200MB
    'gdrive_download_failures': 1,  # Alert on Google Drive failures
    'module_compatibility_errors': 1,  # Alert on module mapping errors
    'error_rate_percent': 0.5  # Alert if >0.5% errors
}

# Google Drive specific monitoring
GDRIVE_MONITORING = {
    'model_id': '1s_sBXnUdJ-rWm4-YEsDixHCbxBca-oXh',
    'expected_size_mb': 32,
    'download_timeout_s': 30,
    'retry_attempts': 3
}
```

## ⚠️ Hybrid Deployment Rollback Plan

### Scenario 1: Google Drive Model Issues
```bash
# 1. Switch to previous Google Drive model (if available)
export GDRIVE_MODEL_ID="previous_model_id_here"

# 2. Restart application
sudo systemctl restart disaster-response-app
# OR docker-compose restart app

# 3. Verify rollback
curl -X GET http://your-production-url/health
```

### Scenario 2: Module Compatibility Issues
```bash
# 1. Use automated rollback script
python scripts/rollback_model.py --reason="module-compatibility"

# 2. Verify application starts
curl -X GET http://your-production-url/health
```

### Scenario 3: Complete Deployment Rollback
```bash
# 1. Revert to previous deployment
# (Platform-specific commands)

# For Docker:
docker run -e GDRIVE_MODEL_ID="backup_model_id" disaster-response-app:previous

# For Heroku:
heroku releases:rollback v123

# For systemd:
sudo systemctl stop disaster-response-app
# Deploy previous version
sudo systemctl start disaster-response-app

# 2. Verify rollback successful
python scripts/deployment_health_check.py --validate-rollback
```

### Emergency Fallback: Local Model Override
```bash
# If Google Drive is completely unavailable
# 1. Deploy with local model (temporary)
unset GDRIVE_MODEL_ID
cp model/backup/disaster_rf_v1-2-0_prod_2025-09-11.pkl model/

# 2. Restart with local model
sudo systemctl restart disaster-response-app

# 3. Monitor and plan Google Drive fix
echo "⚠️ Running on local model - fix Google Drive ASAP"
```

## 🎯 Success Criteria

### Hybrid Deployment Success ✅
- [ ] **Google Drive Integration**: Model downloads successfully from Google Drive
- [ ] **Module Compatibility**: Zero errors from disaster_classifier → disasterproject mapping
- [ ] **Performance**: First request <5s, cached requests <200ms
- [ ] **Reliability**: Zero production errors in first 24 hours
- [ ] **Model Quality**: All 8 critical labels show non-zero predictions
- [ ] **Memory Efficiency**: Memory usage <200MB (32MB model + overhead)
- [ ] **Fallback Capability**: Local model fallback works if Google Drive unavailable

### Business Success ✅
- [ ] **Accuracy Maintained**: F1-Score ~0.9357 performance maintained
- [ ] **Response Quality**: No increase in false positive emergency responses
- [ ] **Scalability**: System handles peak load during disaster events
- [ ] **Deployment Efficiency**: 99.85% reduction in deployment package size
- [ ] **Operational Excellence**: Team comfortable with hybrid deployment procedures

## 📋 Post-Deployment Tasks

### Week 1: Immediate Monitoring
- [ ] Daily performance reports
- [ ] Monitor critical label predictions
- [ ] Collect user feedback
- [ ] Document any issues

### Week 2-4: Optimization
- [ ] Fine-tune thresholds based on production data
- [ ] Implement negation handling improvements
- [ ] A/B test against backup model if needed
- [ ] Performance optimization based on real usage

## 🔧 Hybrid Deployment Scripts Status

### ✅ Available Scripts
1. **`test_gdrive_deployment.py`** - Google Drive model deployment testing
2. **`scripts/test_deployment_scenarios.py`** - All deployment scenario validation
3. **`scripts/model_naming_utility.py`** - Standardized model naming utility

### 📋 Scripts to Create/Update
1. **`scripts/deployment_health_check.py`** - Hybrid deployment verification
2. **`scripts/rollback_model.py`** - Automated rollback with Google Drive support
3. **`scripts/validate_production_model.py`** - Model validation with module compatibility
4. **`scripts/monitor_hybrid_deployment.py`** - Google Drive and performance monitoring

## 📞 Escalation Contacts

- **Technical Lead**: [Your Name]
- **ML Engineer**: [ML Team Lead]
- **DevOps**: [DevOps Team]
- **Product Owner**: [Product Team]

## 🎉 Expected Outcomes

**Immediate Benefits:**
- **99.85% reduction in deployment package size** (32MB model excluded from builds)
- **Hybrid deployment flexibility** (Google Drive production, local development)
- **Professional ML operations** with standardized naming and versioning
- **Module compatibility** preserving existing model without retraining
- **Zero production downtime** during deployment

**Long-term Benefits:**
- **Scalable model management** with independent model updates
- **Improved team collaboration** through standardized conventions  
- **Reduced infrastructure costs** with lightweight deployments
- **Foundation for ML operations maturity** and future enhancements
- **Audit trail and versioning** for compliance and debugging

**Operational Excellence:**
- **Comprehensive monitoring** for hybrid deployment scenarios
- **Multiple rollback strategies** for different failure modes
- **Validated deployment procedures** across all environments
- **Professional tooling** for model lifecycle management

---

**Ready for Hybrid Production Deployment** ✅

The hybrid deployment strategy successfully bridges immediate production needs with long-term ML operations best practices, providing a robust foundation for scalable model management while solving critical deployment blockers.
