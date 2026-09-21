---
title: "Model Promotion Gating Policy"
date: "2025-09-19"
amended: "2026-09-21"
status: "accepted"
tags: ["ml-operations", "model-management", "promotion", "gating"]
author: "ML Engineering Team"
related: ["adr-006-model-artifact-naming-standard.md", "adr-009-algorithm-selection-logistic-regression-over-random-forest.md"]
---

# Model Promotion Gating Policy

**Date:** 2025-09-19  
**Amended:** 2026-09-21  
**Status:** accepted (amended)  
**Deciders**: ML Engineering Team  
**Tags**: ml-operations, model-management, promotion, gating

## Context

- Earlier gating treated weighted F1 as primary and allowed micro F1 to fall back to weighted/samples when missing.
- Soft floors (`min_f1_weighted=0.5`, `max_model_size_mb=1000`) did not enforce the train/cal/eval evaluation contract.
- Thresholds can now be tuned on a calibration split and reported on frozen eval. Promotion must require that provenance and deploy the exact threshold artifact that validation scored.
- The deployed production model may remain historically grandfathered; its **evaluation methodology** is not grandfathered for future promotions.

## Decision

Promotion enforces the evaluation contract via `PERFORMANCE_THRESHOLDS` in `src/disasterproject/utils/config.py` and `scripts/07_operations/promote_model.py`:

1. **Baseline model-quality gate:** frozen-eval micro F1 ≥ `min_baseline_f1_micro` (0.60), from explicit `training_log.json` `performance.micro_f1` / `f1_micro`. No weighted or samples substitution.
2. **Thresholded operating-point gate:** frozen-eval critical recall ≥ `min_eval_critical_recall` (0.55), from `{model_stem}_thresholds.json` → `performance.optimized.critical_recall`.
3. **Size gate:** model ≤ `max_model_size_mb` (50).
4. **Weighted-F1 damage guardrail (not the selection objective):**
   `(baseline_weighted_f1 - optimized_weighted_f1) / baseline_weighted_f1 <= max_weighted_f1_relative_drop` (0.05).
5. **Provenance gate:** thresholds metadata must include `optimization_split=calibration` and `reporting_split=frozen_eval`.
6. **Deploy invariant:** promotion copies the exact validated `{model_stem}_thresholds.json` to `{prod_model_stem}_thresholds.json` and verifies SHA256 identity.
7. **Artifact consistency:** `training_log` baseline micro F1 must agree with `thresholds.performance.baseline.f1_micro`, and `metadata.eval_critical_recall` must agree with `performance.optimized.critical_recall` (absolute tolerance `1e-6`).
8. **Single model file:** candidate directories must contain exactly one `.pkl` (no newest-by-mtime selection).

Missing or invalid evidence fails closed as `validation_errors`. `--force` may override **metric/provenance** gate failures only. Structural prerequisites (`model_path`, `model_hash`, `thresholds_path`, `thresholds_sha256`) are never bypassable — checked via `assert_force_promotion_prerequisites` before dry-run success (on force) and again before archiving production.
## Consequences

### Positive
- Future promotions cannot claim a calibrated operating point and then deploy a different threshold file.
- Baseline micro F1 and thresholded critical recall remain distinct operating points.
- Weighted F1 remains a damage guardrail rather than the model-selection objective.

### Negative
- Tune-on-eval candidates (for example the 2025-11-06 vocab15k promotion artifacts) fail the new contract until reworked with cal/eval provenance.
- Candidates must ship model-stem thresholds artifacts with the required metadata keys.

### Neutral
- Deployed production artifacts are not automatically re-promoted by this policy change.

## Implementation Notes

- Shared discovery: `discover_candidate_thresholds()` resolves `{model_stem}_thresholds.json` (non-f2) for both validation and promotion.
- Relative drop is unit-tested at the exact 0.05 boundary.
- Acceptance: `experiments/experimental_runs/2026-09-21` passes; `2025-11-06-vocab15k-promotion` fails provenance.

## Alternatives Considered

1. **Gate the old tune-on-eval operating point until calibration exists** — Rejected; calibration already exists on main.
2. **Use thresholded micro F1 as the micro gate** — Rejected; threshold policy intentionally trades precision for recall.
3. **Eliminate weighted F1 entirely** — Rejected; relative-drop guardrail prevents threshold policies from destroying global F1 unchecked.
4. **Soft floors only (weighted 0.5 / size 1000MB)** — Rejected; does not enforce the evaluation contract.

## References

- [Performance Thresholds Configuration](../../src/disasterproject/utils/config.py) — `PERFORMANCE_THRESHOLDS`
- [Promotion script](../../scripts/07_operations/promote_model.py)
- [ADR-006: Model Artifact Naming Standard](adr-006-model-artifact-naming-standard.md)
- [ADR-009: Algorithm Selection](adr-009-algorithm-selection-logistic-regression-over-random-forest.md)

## Future Work

- Optionally promote the 2026-09-21 three-way candidate once operators choose to replace the grandfathered production artifact.
- Keep metric key serialization consistent across training and threshold scripts.
