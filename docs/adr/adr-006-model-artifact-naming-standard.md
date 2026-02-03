---
title: "Adopt Standardized Model Artifact Naming"
date: "2025-09-19"
status: "accepted"
tags: ["ml-operations", "governance", "artifacts", "naming-standard"]
author: "ML Engineering Team"
related: ["adr-003-hybrid-model-deployment-strategy.md", "../standards/model-naming.md"]
---

# Adopt Standardized Model Artifact Naming

**Date**: 2025-09-19  
**Status**: Accepted  
**Deciders**: ML Engineering Team, DevOps Team  
**Tags**: ml-operations, governance, artifacts

## Context

The project previously used ad-hoc names for serialized models (e.g., `classifier.pkl`). This complicated deployment, traceability, and rollback. As part of improving ML operations and aligning with the hybrid deployment strategy, we require a clear, machine- and human-readable naming scheme for model artifacts and their companion files.

## Decision

Adopt a standardized naming convention for all model artifacts:

```
{domain}_{algorithm}_{version}_prod_{training_date}.pkl
```

- Domain: business context (e.g., `disaster`)
- Algorithm: short code (e.g., `rf`, `lr`) - **auto-detected during promotion**
- Version: date-based format `v{YY}-{MM}-{DD}` derived from training date (e.g., `v25-11-06` from `2025-11-06`)
- Environment: `prod` for production models
- Training Date: `YYYY-MM-DD` format - **must match the version date**

**Critical**: Both the version (`v25-11-06`) and the date field (`2025-11-06`) refer to the **training date**, not the promotion date. The promotion date is stored separately in `MODEL_INFO.json`.

**Example**: `disaster_lr_v25-11-06_prod_2025-11-06.pkl`
- Algorithm: `lr` (LogisticRegression) - detected automatically
- Version: `v25-11-06` (derived from training date)
- Training Date: `2025-11-06`
- Promotion date: Stored in `MODEL_INFO.json` as `promotion_timestamp`

The convention also applies to companion files (thresholds, labels, metadata, performance metrics) using the same base name with descriptive suffixes (e.g., `{model_stem}_thresholds.json`, `{model_stem}_performance_metrics.csv`).

## Consequences

### Positive
- Clear provenance and environment separation for artifacts
- Safer rollbacks and audits through explicit versioning tied to training dates
- Consistent automation hooks in scripts and deployment pipelines
- Automatic algorithm detection prevents manual errors
- Training date in filename provides clear model lineage
- Version and date alignment ensures consistency

### Negative
- Requires renaming legacy artifacts to follow the standard
- Date-based versioning doesn't convey semantic meaning (major/minor/patch)
- Version format change from semantic (`v1-2-0`) to date-based (`v25-11-06`) requires migration

## Alternatives Considered
- Keep generic names (rejected: poor traceability)
- Embed metadata only inside artifacts (rejected: not visible to ops/tooling)
- Semantic versioning (`v1-2-0`) (rejected: switched to date-based for better traceability to training dates)
- Use promotion date in filename (rejected: training date provides better model lineage)

## References

- Standard details: [Model Naming Standard](../standards/model-naming.md)
- Related decision: [ADR-003: Hybrid Model Deployment Strategy](adr-003-hybrid-model-deployment-strategy.md)

## Status & Migration

- Status: Adopted for all new artifacts as of 2025-09-19
- **Current Implementation** (as of 2026-02-03):
  - Version format: Date-based `v{YY}-{MM}-{DD}` (e.g., `v25-11-06`)
  - Algorithm detection: Automatic during promotion via `scripts/07_operations/promote_model.py`
  - Current production model: `disaster_lr_v25-11-06_prod_2025-11-06.pkl`
  - Promotion workflow: See `model/README.md` for detailed promotion process
- Migration: Legacy files may be retained under `model/legacy/` or renamed following the standard

## Implementation Details

### Algorithm Detection
The promotion script (`scripts/07_operations/promote_model.py`) automatically detects the algorithm type by inspecting the model file structure:
- **RandomForestClassifier** → `rf`
- **LogisticRegression** → `lr`
- Unknown → defaults to `rf` with warning

This prevents manual errors and ensures consistency between the model file and filename.

### Version Derivation
Versions are derived from the training date using the following logic:
1. **Primary**: Extract from candidate directory name (e.g., `2025-11-06-vocab15k-promotion` → `v25-11-06`)
2. **Fallback 1**: Extract from `training_log.json` timestamp
3. **Fallback 2**: Use current date (promotion date) if training date unavailable

The version format `v{YY}-{MM}-{DD}` ensures the version always matches the training date in the filename.

### Promotion Workflow
Models are promoted using `scripts/07_operations/promote_model.py`:
1. Validates candidate model performance
2. Detects algorithm type automatically
3. Generates filename using training date
4. Copies model and metadata files
5. Verifies file integrity (hash check)
6. Updates `MODEL_INFO.json` with promotion metadata
7. Archives previous production model

See `model/README.md` for complete workflow documentation.

