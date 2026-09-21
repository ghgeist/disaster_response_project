---
title: "Use Class Weighting Over Multi-Label Sampling for Imbalanced Data"
date: "2026-01-26"
status: "accepted"
tags: ["ml-operations", "class-imbalance", "data-quality", "training-strategy"]
author: "ML Engineering Team"
related: ["../dev_notes/2025-09-16.md", "adr-004-filter-related-category-from-ui.md", "adr-009-algorithm-selection-logistic-regression-over-random-forest.md"]
---

# Use Class Weighting Over Multi-Label Sampling for Imbalanced Data

**Date**: 2026-01-26  
**Status**: Accepted (strategy); see Amendment for current production training fact  
**Deciders**: ML Engineering Team  
**Tags**: ml-operations, class-imbalance, data-quality, training-strategy

## Context

The disaster response classification system faces extreme class imbalance across 36 multi-label categories. During hyperparameter optimization work (2025-09-16), a critical data quality issue was discovered: the `child_alone` category has **0 positive examples** across all 26,027 messages in the dataset (0.000%). Additionally, several other categories have extremely rare positive examples:
- `tools`: 159 examples (0.61%)
- `shops`: 120 examples (0.46%)
- `offer`: 118 examples (0.45%)

Recent spot checks show the production model can miss labels on short, low-context inputs (for example, single-word prompts like "water" can fall below the `water` threshold). This reinforces the need to address data quality (better translations, richer context) before sampling strategies are likely to help.

The project initially considered multi-label sampling approaches (SMOTE, ADASYN) to address class imbalance, with validation scripts (`scripts/validate_multilabel_sampling.py`) implemented to test these methods. However, data quality limitations revealed fundamental incompatibilities with sampling-based approaches.

## Decision

Prefer **class weighting** (via `get_multilabel_class_weights()`) over data resampling techniques (SMOTE, ADASYN, or other oversampling methods) when the training pipeline enables imbalance mitigation.

The training pipeline (`scripts/02_training/03_create_experimental_model.py`, `scripts/02_training/04_create_production_model.py`) supports:
- `get_multilabel_class_weights(y_train, strategy='balanced')` to calculate per-label weights
- `create_pipeline_with_custom_weights()` or `create_pipeline_logistic_regression_weighted()` to apply weights
- No data resampling or synthetic sample generation when weights are enabled via config

Sampling validation scripts remain available for experimentation (`scripts/validate_multilabel_sampling.py`, `scripts/02_training/01_test_sampling_strategies.py`) but are not used in production model training.

**This ADR records the preferred strategy when weighting is turned on.** It does not assert that every promoted artifact was trained with weights enabled. See the Amendment below for the current production artifact.

## Amendment (2026-09-21): Current production training fact

The promoted production model `disaster_lr_v26-09-21_prod_2026-09-21.pkl` (three-way vocab15k LR) was trained with **class weighting disabled**:

- Training log: `experiments/experimental_runs/2026-09-21/training_log.json` records `class_weighting.enabled: false`
- Default candidate config: `experiments/model_candidates/class_weights.json` has `class_weights.enabled: false`

**Live imbalance handling for this artifact** relies primarily on **per-label optimized thresholds** at inference (`model/disaster_lr_v26-09-21_prod_2026-09-21_thresholds.json`), tuned on the calibration split under the train/cal/eval contract.

The prior production artifact `disaster_lr_v25-11-06_prod_2025-11-06.pkl` was likewise trained with weighting disabled (see `experiments/experimental_runs/2025-11-06-vocab15k-promotion/training_log.json`).

Re-enabling class weighting for a future promotion is treated as an **experiment**: do not claim it improves the current LR model until a side-by-side comparison is recorded.

## Consequences

### Positive
- **Robustness across all labels**: Class weighting gracefully handles labels with 0 positive examples (falls back to equal weights 1.0, 1.0) without crashing
- **Consistent behavior**: Works uniformly across all 36 labels regardless of class distribution
- **Preserves label relationships**: Maintains original multi-label combinations without creating synthetic label vectors that may not exist in real data
- **Faster training**: No synthetic data generation overhead; trains on original dataset size
- **Lower memory footprint**: No increase in training set size
- **Better for text data**: Avoids creating synthetic TF-IDF feature vectors that don't correspond to real text messages
- **Production reliability**: Eliminates risk of sampling failures on rare or zero-positive labels

### Negative
- **Potential performance trade-off**: Class weighting may be less effective than sampling for moderately imbalanced labels (though evidence suggests comparable performance)
- **Less intuitive**: Weights are less visible than resampled data distributions
- **Validation scripts unused**: Sampling validation infrastructure exists but isn't leveraged in production (though still useful for experimentation)
- **Docs/artifact drift risk**: Preferring weighting in policy can be misread as "current prod uses weights" unless training logs are checked

### Neutral
- **Experimental flexibility**: Sampling scripts remain available for future experimentation if data quality improves
- **No change to model architecture**: Decision affects training strategy, not model structure
- **Compatible with existing models**: Both RandomForest and LogisticRegression support class weighting
- **OOD / placeholder text**: Class weighting does not address out-of-distribution or non-message input (e.g. Lorem Ipsum, gibberish, non-English). The "related" head has ~76% positive examples in training, so for neutral or OOD input the model can output "related" near that prior (e.g. ~66%). We mitigate by excluding "related" from classification API and UI display (see [ADR-004](adr-004-filter-related-category-from-ui.md); API: `_build_simplified_classification` in `app/routes/api.py`).

## Alternatives Considered

1. **SMOTE (Synthetic Minority Oversampling Technique)**: 
   - **Rejected**: Requires minimum k_neighbors (5-7) positive examples per label. Fails on `child_alone` (0 positives) and struggles with very rare labels (118-159 positives). Creates synthetic feature vectors that don't correspond to real text.

2. **ADASYN (Adaptive Synthetic Sampling)**:
   - **Rejected**: Same fundamental limitation as SMOTE - requires sufficient positive examples for neighbor-based interpolation. Focuses on harder-to-learn examples but still fails on zero-positive labels.

3. **Random Oversampling**:
   - **Rejected**: Would duplicate existing rare examples without addressing the fundamental data quality issue. Doesn't help with zero-positive labels.

4. **Label Powerset Approach**:
   - **Rejected**: Treats each unique label combination as a class, but with 36 labels this creates an exponential number of combinations. Most combinations have zero examples, making sampling impossible.

5. **Hybrid Approach (Sampling + Weighting)**:
   - **Rejected**: Adds complexity without clear benefit. Sampling failures would still need to be handled, and class weighting alone achieves comparable results.

6. **Remove Zero-Positive Labels**:
   - **Rejected**: `child_alone` is excluded from hierarchy constraints but kept in model outputs for potential future use. Removing labels entirely loses structural completeness.

The chosen approach (class weighting when enabled) provides the best balance of robustness, performance, and production reliability given the data quality constraints—without requiring synthetic multi-label samples.

## Implementation Details

### Class Weighting Implementation
- **Location**: `src/disasterproject/models/samplers.py::get_multilabel_class_weights()`
- **Strategy**: Balanced weights calculated as `w_c = N / (n_classes * n_c)` where N is total samples and n_c is count of class c
- **Fallback**: Labels with missing classes (0 positives) receive equal weights (1.0, 1.0) to prevent undefined behavior

### Production Usage
- **Experimental models**: `scripts/02_training/03_create_experimental_model.py` uses class weighting when `class_weights.enabled=true` in config
- **Production models**: `scripts/02_training/04_create_production_model.py` applies class weighting based on config file
- **Current production model**: `disaster_lr_v26-09-21_prod_2026-09-21.pkl` — LogisticRegression trained with class weighting **disabled**; inference uses cal-tuned per-label thresholds

### Sampling Infrastructure (Retained for Experimentation)
- **Validation script**: `scripts/validate_multilabel_sampling.py` - Tests SMOTE, ADASYN, and other methods
- **Experiment scripts**: `scripts/02_training/01_test_sampling_strategies.py` - Compares sampling strategies
- **Status**: Available for experimentation but not used in production training

## References

- **Data Quality Discovery**: [Dev Note 2025-09-16](../dev_notes/2025-09-16.md) - Hyperparameter optimization work where `child_alone` issue was discovered
- **Class Weighting Implementation**: `src/disasterproject/models/samplers.py::get_multilabel_class_weights()`
- **Sampling Implementation**: `src/disasterproject/models/samplers.py::apply_proper_multilabel_sampling()`
- **Production Model Training**: `scripts/02_training/03_create_experimental_model.py`, `scripts/02_training/04_create_production_model.py`
- **Promotion training log**: `experiments/experimental_runs/2026-09-21/training_log.json` (prior: `2025-11-06-vocab15k-promotion/training_log.json`)
- **Sampling Validation**: `scripts/validate_multilabel_sampling.py`
- **Data Quality Analysis**: `notebooks/02_data_quality_analysis.ipynb` - Documents zero-positive and rare categories
- **README Documentation**: `README.md` - Documents `child_alone` exclusion from hierarchy constraints
- **Configuration**: `src/disasterproject/utils/config.py` - Defines `EXCLUDE_FROM_CONSTRAINTS = {"child_alone"}`

## Status & Migration

- **Status**: Strategy accepted; production training currently has weights off (see Amendment 2026-09-21)
- **Current Production Model**: `disaster_lr_v26-09-21_prod_2026-09-21.pkl` (class weighting disabled; cal-tuned thresholds at inference)
- **Prior Production Model (historical)**: `disaster_lr_v25-11-06_prod_2025-11-06.pkl`
- **Future Consideration**: If comparative eval shows balanced weights improve the current LR setup, enable via config and re-promote. If data quality improves (e.g., `child_alone` gains positive examples), sampling approaches could be re-evaluated for specific labels
