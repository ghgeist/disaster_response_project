title: "Model Improvement Agent: Boost Disaster Classifier Performance"
date: "2025-09-13"
status: "active"
priority: "high"
estimated duration: "≤ 4 hours"
tags: \["ML", "pipeline", "performance", "F1", "experiments"]

# Model Improvement Agent: Boost Disaster Classifier Performance

**Date**: 2025-09-13
**Status**: Active
**Priority**: High
**Estimated Duration**: ≤ 4 hours
**Tags**: ML, Pipeline, Performance, F1, Experiments

## 🎯 Objective

Increase model quality enough to impress a hiring manager. Deliver a reproducible experiment showing a measurable improvement in macro-F1 (≥5%), document findings, and produce a promotable model artifact for the Flask UI.

## 📋 Success Criteria

* [ ] Macro-F1 improves ≥5% on validation vs baseline.
* [ ] Per-class metrics exported to CSV and summarized in `experiments/summary.md`.
* [ ] New model artifact promoted and loadable in Flask UI.
* [ ] Rollback path (previous artifact) documented in case of regressions.

## 🔍 Context

Current pipeline uses sklearn models (LogisticRegression, RandomForest) with basic preprocessing. There is no threshold tuning, class weighting, or calibrated probabilities. Evaluation is limited to global F1 with no slice analysis. Quick modeling uplift will demonstrate ability to iterate thoughtfully under constraints.

## 📝 Requirements

### Functional Requirements

* Train baseline and improved models using same data split.
* Log precision, recall, F1 per class and macro averages.
* Save artifacts using `Config.MODEL_FILENAME` convention.

### Technical Requirements

* Use existing libraries (scikit-learn, pandas).
* Keep runtime < 20 min on laptop.
* No major dependency additions (no deep learning).

### Quality Requirements

* Deterministic split for reproducibility (set random seed).
* Versioned artifacts with timestamps or semantic names.
* Simple, readable experiment code with comments.

## 🛠️ Approach

1. **Understand Current Setup:** Review `pipeline.py`, training scripts, current hyperparameters.
2. **Identify Levers:**

   * Try `class_weight="balanced"` in LogisticRegression/RandomForest.
   * Adjust decision thresholds using precision-recall curves.
   * Optionally prune low-signal features.
3. **Experiment:**

   * Train baseline + improved model on same split.
   * Log metrics to CSV and generate macro-F1 comparison chart.
4. **Promote:**

   * Save improved model to `model/` with config-driven filename.
   * Update Flask to load new artifact; run smoke test.
5. **Document:**

   * Summarize results in `experiments/summary.md`.
   * Include before/after F1 table and chart.
6. **Rollback Plan:**

   * Keep old model artifact; revert `MODEL_FILENAME` to roll back.

## 📊 Acceptance Criteria

* `pytest -q` still passes after promoting model.
* Flask UI loads and returns predictions from new artifact.
* `experiments/summary.md` contains a clear F1 uplift chart.
* Old artifact remains available for comparison.

## 🔗 Related Work

* `src/disasterproject/models/pipeline.py` and `samplers.py`.
* `scripts/04_create_production_model.py` (training script).
* Evaluation utilities in `src/disasterproject/evaluation/metrics.py`.

## 📈 Metrics

* Macro-F1 uplift (target ≥5%).
* Per-class recall on critical categories (no regressions).
* Training runtime (target ≤ 20 minutes).

## 🚨 Risks & Mitigations

| Risk                          | Impact | Probability | Mitigation                                  |
| ----------------------------- | ------ | ----------- | ------------------------------------------- |
| Overfitting on validation set | High   | Medium      | Use fixed split; hold out 10% as final test |
| Model size increases too much | Low    | Low         | Prefer simple models, limit tree depth      |
| F1 drops for rare labels      | Med    | Medium      | Log per-class metrics, monitor regressions  |

## 📄 Deliverables

* [ ] Updated training script or notebook with reproducible run.
* [ ] Metrics CSV and F1 comparison chart in `experiments/`.
* [ ] Promoted model artifact in `model/`.
* [ ] `experiments/summary.md` write-up with before/after metrics.
* [ ] Rollback instructions (1 line in README).