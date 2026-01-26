---
title: "Model Promotion Gating Policy"
date: "2025-09-19"
status: "accepted"
tags: ["ml-operations", "model-management", "promotion", "gating"]
author: "ML Engineering Team"
related: ["adr-006-model-artifact-naming-standard.md"]
---

# Model Promotion Gating Policy

**Date:** 2025-09-19  
**Status:** accepted  
**Deciders**: ML Engineering Team  
**Tags**: ml-operations, model-management, promotion, gating

## Context

- Prior gating treated per-label positive-class F1 as "micro-F1," causing false failures.
- Promotion now prefers true micro metrics when present; otherwise falls back to weighted F1.

## Decision

- Primary gate: use weighted F1 from training/evaluation artifacts.
- Secondary gate: use micro F1 if explicitly logged; else fall back to weighted F1.
- Thresholds live in src/disasterproject/utils/config.py:204 (PERFORMANCE_THRESHOLDS).

## Consequences

### Positive
- Prevents false negatives during promotion when micro isn't logged.
- Keeps promotion deterministic and aligned with reported metrics.

### Negative
- Requires consistent metric logging across all model training scripts.
- Fallback logic adds complexity to promotion validation.

### Neutral
- No impact on model performance or training process.

## Implementation Notes

- Promotion reads micro_f1/samples_f1 if available; otherwise uses overall_f1 (weighted).
- Experimental model script now writes micro_f1 and samples_f1 to training_log.json.

## Alternatives Considered

1. **Always use weighted F1**: Rejected - loses granularity of micro metrics when available
2. **Always use micro F1**: Rejected - causes false failures when micro metrics not logged
3. **Require micro F1**: Rejected - breaks backward compatibility with existing models

## References

- [Performance Thresholds Configuration](../../src/disasterproject/utils/config.py) - Line 204: PERFORMANCE_THRESHOLDS
- [ADR-006: Model Artifact Naming Standard](adr-006-model-artifact-naming-standard.md) - Related naming and artifact management

## Future Work

- Make gating fully configurable (select metrics and AND/OR logic in config).
- Centralize metric computation/serialization to ensure consistent keys across scripts.
