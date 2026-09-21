---
title: "Hybrid Model Deployment Strategy with Standardized Naming"
date: "2025-09-12"
status: "accepted"
tags: ["ml-operations", "deployment", "model-management", "production"]
author: "ML Engineering Team"
related: ["adr-002-tokenization-trade-offs.md"]
---

# Hybrid Model Deployment Strategy with Standardized Naming

**Date**: 2025-09-12  
**Status**: Accepted (amended 2026-09-21)  
**Deciders**: ML Engineering Team, DevOps Team  
**Tags**: ml-operations, deployment, model-management, production

## Amendment (2026-09-21): Local tracked production artifact

**Current deployment behavior** (supersedes the Google Drive–required production path below for new work):

- **Production model**: `disaster_lr_v26-09-21_prod_2026-09-21.pkl` (LogisticRegression, vocab15k, train/cal/eval)
- **Storage**: Git-tracked file under `model/`; Flask loads from local disk only
- **Discovery**: Newest `disaster_*_prod_*.pkl` via `app/config.py` (optional `MODEL_FILENAME` override)
- **Google Drive / `GDRIVE_MODEL_ID`**: Retained below as **historical context** for the original hybrid design. The runtime download path is no longer used by the app loader.

**Prior production (historical)**: `disaster_lr_v25-11-06_prod_2025-11-06.pkl` — tune-on-eval metrics; archived metadata under `experiments/model_archive/`.

The original decision text that follows is preserved so the 2025 hybrid/GDrive design history is not erased.

## Context

The disaster response classification system faced a critical production deployment blocker: a module path mismatch between the trained model (`disaster_classifier`) and the refactored codebase (`disasterproject`). This created a `ModuleNotFoundError` preventing model loading and system functionality.

Additionally, the project lacked professional ML operations practices:
- Generic model naming (`classifier.pkl`) without version control
- No clear deployment strategy for the 32MB production model
- Inconsistent model artifact management
- Missing deployment environment differentiation

The system required immediate production deployment capability while establishing sustainable model lifecycle management practices.

## Decision

We have implemented a **Hybrid Model Deployment Strategy** with the following components:

### 1. Module Compatibility Layer
- **Runtime module path patching** in `ModelService.load_model()`
- Creates fake `disaster_classifier` module structure in `sys.modules`
- Maps old paths to new `disasterproject.data.preprocessor.tokenize` function
- Preserves existing model without retraining requirements

### 2. Standardized Model Naming Convention
- **Format**: `{domain}_{algorithm}_{version}_prod_{training_date}.pkl`
- **Current production model** (as of 2026-09-21): `disaster_lr_v26-09-21_prod_2026-09-21.pkl`
- **Historical production example** (2026-02-03 → 2026-09-21): `disaster_lr_v25-11-06_prod_2025-11-06.pkl`
- **Version format**: Date-based `v{YY}-{MM}-{DD}` derived from training date (e.g., `v26-09-21`)
- **Artifact consistency**: All supporting files follow same naming pattern with model-specific suffixes
- **Algorithm detection**: Automatic during promotion via `scripts/07_operations/promote_model.py`
- **Note**: Version format changed from semantic (`v1-2-0`) to date-based (`v25-11-06` / `v26-09-21`) for better traceability

#### Legacy Artifact Handling
- Renamed `classifier.pkl` to `legacy_classifier.pkl` and moved it to `model/legacy/`
- Updated internal scripts to reference the versioned production artifact
- Historical documentation may still reference `model/classifier.pkl` for archival context
- **Current production model**: `disaster_lr_v26-09-21_prod_2026-09-21.pkl` (LogisticRegression, date-based versioning)

### 3. Hybrid Deployment Architecture
> **Superseded for active deploys** by the 2026-09-21 amendment (local tracked `model/*.pkl`). The bullets below describe the **original 2025 hybrid design**.

- **Production Environment (historical)**: Google Drive model storage
  - Lightweight deployments without large model files in repository
  - Model downloaded on first application startup
  - Environment variable (historical): `GDRIVE_MODEL_ID="1s_sBXnUdJ-rWm4-YEsDixHCbxBca-oXh"`
- **Development Environment (historical)**: Local model with Google Drive fallback
  - Local model for fast development cycles
  - Google Drive option for testing production behavior
  - Flexible environment variable configuration
- **Current Environment (2026-09-21+)**: Local git-tracked production pickle + mtime auto-discovery; no runtime Drive download

### 4. Professional Tooling
- **Model naming utility**: `scripts/model_naming_utility.py`
- **Deployment testing**: Comprehensive test scripts for all scenarios
- **Documentation**: Complete team guidelines and procedures

## Consequences

### Positive
- **Immediate Production Readiness**: System can be deployed today without model retraining
- **Professional ML Operations**: Industry-standard model versioning and lifecycle management
- **Deployment Flexibility**: Optimized for both production constraints and development efficiency
- **Team Collaboration**: Clear naming conventions and documentation enable better collaboration
- **Repository Performance**: Git operations remain fast without large binary files
- **Independent Model Updates**: Models can be updated without code deployments
- **Environment Isolation**: Clear separation between production, staging, and development models
- **Audit Trail**: Complete version history and artifact management

### Negative
- **Runtime Complexity**: Module compatibility layer adds complexity to model loading
- **Network Dependency**: Production deployments require internet connectivity for first startup
- **Google Drive Dependency**: Production relies on external service availability
- **Migration Path**: Future model updates still require compatibility considerations until full migration

### Neutral
- **Memory Usage**: Minimal impact on runtime memory footprint
- **Performance**: Sub-second local loading maintained, ~2-5 second Google Drive download on first run
- **Security**: Google Drive files are publicly readable (appropriate for non-sensitive ML models)

## Alternatives Considered

### Alternative 1: Retrain Model with Current Codebase
**Description**: Retrain the RandomForest model using `disasterproject` module structure

**Pros**: 
- Clean solution without runtime patches
- Eliminates module path complexity

**Cons**: 
- Requires significant retraining time and computational resources
- Risk of performance regression from current optimized model
- Delays production deployment by days/weeks
- Would need to recreate optimized thresholds and configurations

**Rejection Reason**: Production deployment urgency and risk of performance regression

### Alternative 2: Local Model Only Deployment
**Description**: Include 32MB model files directly in repository/deployment packages

**Pros**: 
- No network dependencies
- Instant model availability
- Simple deployment architecture

**Cons**: 
- 32MB+ repository size impacts all developers
- Slow git operations (clone, pull, push)
- GitHub file size limitations
- Coupled model updates with code deployments
- Multiple copies across environments increase storage costs

**Rejection Reason**: Repository performance impact and deployment inflexibility

### Alternative 3: Dedicated Model Storage Service
**Description**: Implement custom model artifact storage (S3, Azure Blob, etc.)

**Pros**: 
- Enterprise-grade model storage
- Advanced access controls and versioning
- Better integration with ML pipelines

**Cons**: 
- Additional infrastructure complexity and costs
- Requires authentication/credential management
- Over-engineering for current project scale
- Longer implementation timeline

**Rejection Reason**: Complexity overkill for current requirements and timeline constraints

### Alternative 4: Code Reversion to disaster_classifier
**Description**: Revert codebase naming back to original `disaster_classifier` structure

**Pros**: 
- Immediate compatibility with existing model
- No runtime patches required

**Cons**: 
- Undoes valuable refactoring work
- `disasterproject` naming better reflects current project scope
- Regression in codebase organization and clarity
- Sets precedent for avoiding necessary improvements

**Rejection Reason**: Counterproductive regression of codebase improvements

## References

- [Model Naming Convention Documentation](../standards/model-naming.md) - Updated to reflect date-based versioning
- [Model README](../../model/README.md) - Current production model details and promotion workflow
- [Deployment Configuration Guide](../runbooks/deployment.md) - Current local-artifact deployment (with historical GDrive notes)
- [Historical Google Drive Model Storage](https://drive.google.com/file/d/1s_sBXnUdJ-rWm4-YEsDixHCbxBca-oXh/view) - Archival link for the 2025 hybrid design
- [ADR-002: Tokenization Trade-offs](adr-002-tokenization-trade-offs.md)
- [ADR-006: Model Artifact Naming Standard](adr-006-model-artifact-naming-standard.md) - Detailed naming convention
- [ADR-009: Algorithm Selection](adr-009-algorithm-selection-logistic-regression-over-random-forest.md) - Current algorithm (LogisticRegression)

## Implementation Status

- ✅ **Module compatibility layer**: Implemented and tested
- ✅ **Standardized naming**: Applied to production model and artifacts
- ✅ **Google Drive integration (historical)**: Configured and validated for the 2025 hybrid path; superseded by local tracked artifacts
- ✅ **Local tracked production artifact**: `disaster_lr_v26-09-21_prod_2026-09-21.pkl` (2026-09-21)
- ✅ **Development tooling**: Created and documented
- ✅ **Testing framework**: All deployment scenarios validated
- ✅ **Documentation**: Complete team guidelines established
- ✅ **Legacy artifact archived**: `model/legacy/legacy_classifier.pkl` with scripts updated to use versioned artifact
- ✅ **Version format migration**: Switched from semantic versioning (`v1-2-0`) to date-based (`v25-11-06`) as of 2025-11-06
- ✅ **Algorithm detection**: Automatic algorithm detection implemented in promotion script

## Migration Path

**Immediate (Completed)**: Production system functional with hybrid approach

**Short-term (Next 1-2 months)**: 
- Monitor production stability and performance
- Collect deployment feedback from team
- Refine tooling based on usage patterns

**Long-term (3-6 months)**:
- Evaluate model retraining with current codebase
- Consider migration to dedicated model storage if scale demands
- Remove compatibility layer when all models trained with current structure

## Success Metrics

- ✅ **Zero production errors** related to model loading
- ✅ **Sub-100ms model loading** for cached models
- ✅ **<5 second startup time** for Google Drive downloads
- ✅ **100% deployment scenario coverage** in testing
- ✅ **Team adoption** of standardized naming conventions

---

**Decision Rationale**: This hybrid approach balances immediate production needs with long-term ML operations best practices, providing a sustainable foundation for model lifecycle management while solving the critical deployment blocker.
