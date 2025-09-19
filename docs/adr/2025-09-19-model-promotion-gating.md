# Model Promotion Gating Policy

**Date:** 2025-09-19  
**Status:** accepted

## Context

- Prior gating treated per-label positive-class F1 as "micro-F1," causing false failures.
- Promotion now prefers true micro metrics when present; otherwise falls back to weighted F1.
- Promotion now prefers true micro metrics when present; otherwise falls back to weighted F1.

## Decision

- Primary gate: use weighted F1 from training/evaluation artifacts.
- Secondary gate: use micro F1 if explicitly logged; else fall back to weighted F1.
- Thresholds live in src/disasterproject/utils/config.py:204 (PERFORMANCE_THRESHOLDS).

## Consequences

- Prevents false negatives during promotion when micro isn't logged.
- Keeps promotion deterministic and aligned with reported metrics.

## Implementation Notes

- Promotion reads micro_f1/samples_f1 if available; otherwise uses overall_f1 (weighted).
- Experimental model script now writes micro_f1 and samples_f1 to training_log.json.

## Future Work

- Make gating fully configurable (select metrics and AND/OR logic in config).
- Centralize metric computation/serialization to ensure consistent keys across scripts.
