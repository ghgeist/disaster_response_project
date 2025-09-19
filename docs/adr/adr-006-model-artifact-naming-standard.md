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
{domain}_{algorithm}_{version}_{environment}_{date}.pkl
```

- Domain: business context (e.g., `disaster`)
- Algorithm: short code (e.g., `rf`, `tfidf`, `bert`)
- Version: semantic (e.g., `v1-2-0`)
- Environment: `prod|stg|dev|exp`
- Date: `YYYY-MM-DD`

The convention also applies to companion files (thresholds, labels, metadata, params) using the same base name with descriptive suffixes.

## Consequences

### Positive
- Clear provenance and environment separation for artifacts
- Safer rollbacks and audits through explicit versioning
- Consistent automation hooks in scripts and deployment pipelines

### Negative
- Requires renaming legacy artifacts to follow the standard

## Alternatives Considered
- Keep generic names (rejected: poor traceability)
- Embed metadata only inside artifacts (rejected: not visible to ops/tooling)

## References

- Standard details: [Model Naming Standard](../standards/model-naming.md)
- Related decision: [ADR-003: Hybrid Model Deployment Strategy](adr-003-hybrid-model-deployment-strategy.md)

## Status & Migration

- Status: Adopted for all new artifacts as of 2025-09-19
- Migration: Legacy files may be retained under `model/legacy/` or renamed following the standard

