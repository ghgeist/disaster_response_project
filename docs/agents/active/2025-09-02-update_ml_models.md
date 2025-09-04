````markdown
---

## Frozen Eval Set and Reproducible Evaluation (New)

### What and Why

- A frozen eval set is a fixed, immutable subset of the data used for evaluation only. It guarantees apples-to-apples comparisons across models and time.
- We now create and persist the exact membership to `data/04_fct/eval_ids.csv` as stable UIDs.

### How UIDs are defined

- UID = SHA-1 of `<message>|<row_index>` from the loaded dataset. This remains stable so long as message text and dataset ordering are unchanged.

### Scripts and Defaults

- `scripts/create_frozen_eval_ids.py` — creates the eval IDs CSV once.
- `scripts/04_create_production_model.py` — defaults to using the frozen eval set if `data/04_fct/eval_ids.csv` exists.
  - Flags:
    - `--eval-ids PATH` to explicitly provide a file
    - `--no-frozen-eval` to force a random split
  - Logs split mode and sizes to `model/training_log.json`.
- `scripts/03_create_experimental_model.py` — mirrors the same default and flags.
  - Snapshots `eval_ids_used.csv` into `experiments/results/` for traceability.

### One-time setup

```bash
python scripts/create_frozen_eval_ids.py \
  --db data/02_stg/stg_disaster_response.db \
  --out data/04_fct/eval_ids.csv \
  --test-size 0.2 --seed 42
```

### Train with the frozen eval set (default behavior)

```bash
# Production model (outputs to model/)
python scripts/04_create_production_model.py \
  --db data/02_stg/stg_disaster_response.db \
  --params model/parameters.json \
  --class-weights model/class_weights.json \
  --output model/classifier.pkl

# Experimental model (outputs to experiments/results/)
python scripts/03_create_experimental_model.py \
  --db data/02_stg/stg_disaster_response.db \
  --params experiments/model_candidates/parameters.json \
  --class-weights experiments/model_candidates/class_weights.json \
  --output experiments/results/experimental_classifier.pkl
```

---

## Housekeeping (New)

- Legacy/broken-tokenizer results moved to `experiments/results/legacy_tokenizer/original_prediction_results.csv` and treated as “non-comparable.”

---

## Sampling Experiments Runbook (Updated)

Use the interactive runner to execute baseline and imbalance-mitigation variants:

```bash
python scripts/01_test_sampling_strategies.py data/02_stg/stg_disaster_response.db
```

- Run: baseline, smote, conservative; then choose “Compare all experiments.”
- Review per-label recall/F1 (especially labels that were formerly zero-recall), macro/weighted F1, training time, and model size.
- Note: These experiments use a consistent random seed. Head-to-head against production should be performed on the frozen eval set using the production/experimental creation scripts above.

---

## Why ADASYN is disabled (multi-label constraint)

**What happens**
- imbalanced-learn raises: “Imbalanced-learn currently supports binary, multiclass and binarized encoded multiclass targets. Multilabel and multioutput targets are not supported.”

**Root cause**
- ADASYN expects a 1D target vector \(y\). Our setup is multi-label with \(Y \in \mathbb{R}^{n\times L}\) (one column per disaster category), so calling ADASYN on the full \(Y\) fails.

**Current decision**
- We removed ADASYN from the experiment menu. Prior attempts safely fell back to baseline behavior (no oversampling).

### How we could enable ADASYN in the future

- One-vs-Rest with per-label samplers
  - Train a binary pipeline per label and place ADASYN inside each binary pipeline so that it sees a 1D \(y\) per fit.
  - Sketch:
  ```python
  from imblearn.pipeline import Pipeline
  from imblearn.over_sampling import ADASYN
  from sklearn.multiclass import OneVsRestClassifier

  binary_pipeline = Pipeline([
      ("vect", vectorizer),
      ("adasyn", ADASYN()),
      ("clf", base_classifier),
  ])
  model = OneVsRestClassifier(binary_pipeline)
  ```
  - Caveats: vectorization may be refit per label; memory/time costs can be high. A custom loop that fits the vectorizer once, then applies ADASYN on the transformed space per label, can reduce overhead.

- Label Powerset transformation
  - Convert multi-label targets to a single multiclass target (unique label combinations), then apply ADASYN.
  - Caveats: potentially huge class space, many rare classes, and poorer generalization for unseen combinations.

- Prefer multi-label–friendly alternatives
  - Class weighting in the classifiers (already supported and simple).
  - Conservative/heuristic resampling that respects label co-occurrence patterns.
  - Threshold tuning and iterative stratification for fair evaluation splits.


## Comparison and Promotion Gates (New)

### Head-to-head comparison

- Compare production (`model/`) vs candidate (`experiments/results/`) using the frozen eval set only.
- Report deltas for: per-label precision/recall/F1, macro/micro/weighted aggregates, training time, model size.
- Optional statistical confidence: bootstrap CIs for macro F1 and per-label recall; paired tests for per-label classification changes where applicable.

### Threshold tuning (optional)

- Optimize per-label thresholds for F-beta (β=2) on a validation split (not the frozen eval set) to favor recall; freeze thresholds in the saved model metadata.

### Promotion gates

- Positive-class targets: recall ≥ 25%, F1 ≥ 20%.
- Fewer zero-recall labels than production.
- No >2% drop in weighted F1; latency and model size remain reasonable.
- Must pass “critical phrases” regression: "Help me!", "Save us", "We need help".

---

## Current Status (Today)

- Tokenizer fixed (disaster-aware stopword filtering) and verified.
- Frozen eval set created at `data/04_fct/eval_ids.csv`.
- Production and experimental creation scripts updated to default to the frozen eval set with safety flags and logging.
- Legacy results quarantined to `experiments/results/legacy_tokenizer/`.
- Production retraining on the frozen eval set: in progress/completed during this session.

---

## Next Steps

1. Run the sampling experiments via `scripts/01_test_sampling_strategies.py` (baseline, smote, adasyn, conservative) and review outputs in `results/`.
2. Train an experimental candidate with the frozen eval set using `scripts/03_create_experimental_model.py`.
3. Head-to-head comparison vs production on the frozen eval set; produce a brief delta report and verify promotion gates.
4. (Optional) Apply per-label threshold tuning (F2) if recall targets are narrowly missed, then re-evaluate on the frozen set.
5. Promote the winner by keeping `app/config.py` pointing at `model/classifier.pkl` and replacing that artifact; smoke test the Flask app.

---
title: "Update ML Models"
date: "2025-09-02"
status: "active"
tags: ["documentation", "instruction"]
author: "runner"
related: []
---

# ML Model Development Strategy and Implementation Plan

## Executive Summary

This document outlines a systematic approach to **diagnosing, improving, and validating** machine learning models for disaster message classification. The plan addresses a critical issue: the current model fails on basic cases like `"Help me!"` not being classified as disaster-related. We will establish proper baselines, systematically test hypotheses, and demonstrate both working code and professional methodology for portfolio presentation.

---

## Current State Assessment

### System Assets
- Established ML infrastructure and training pipeline
- Four experimental model configurations completed
- Functional Flask application framework
- Comprehensive evaluation metrics collection

### Critical Issues Identified
- **Model Failure on Basic Cases**: `"Help me!"` not classified as disaster-related  
- **Root Cause Identified**: Preprocessing is too aggressive — `"Help me!"` → `['help']` (loses "me" and punctuation)
- **Missing Experimental Framework**: No systematic hypothesis testing or validation
- **Portfolio Readiness**: Need both working code and professional methodology demonstration

### Important Context
- **Baseline Model Uses Optimized Parameters**: The “baseline” model actually uses previously optimized parameters from original GridSearchCV (`n_estimators=100`, `ngram_range=(1,1)`).
- **⚠️ Optimization Compromised**: The original GridSearchCV was optimized on **incorrectly preprocessed data** (`"Help me!"` → `['help']`), so these hyperparameters may not be optimal for correctly preprocessed data.
- **This Strengthens Analysis**: We’re diagnosing why an **already-optimized model** fails, and discovering the optimization itself was compromised.
- **Portfolio Advantage**: Shows systematic problem diagnosis beyond just hyperparameter tuning, including data quality issues.

---

## Quick-Start Runbook

### 🚀 Phase 1: Automated Experiment Execution
```bash
python run_all_experiments.py
````

**Queue status:**

* 🧪 `baseline_no_sampling` — running
* ⏳ `smote_conservative` — next
* ⏳ `adasyn_moderate` — third
* ⏳ `conservative_sampling` — final

> Timeline hint: \~20 minutes total (good window for a quick workout).

---

### 🧭 Phase 2: Post-Run Checks (15–20 minutes)

#### Step 1: Verify outputs

```bash
ls -la experiments/
ls -la models/*.pkl
cat experiment_report_*.json
```

#### Step 2: Compare results

```bash
python scripts/compare_models.py
```

#### Step 3: Inspect detailed metrics

```bash
ls -la data/04_fct/
cat data/04_fct/fct_*_prediction_results.csv
```

---

## Implementation Strategy

### Phase 1: Preprocessing Fix (Root Cause)

**Duration:** 30 minutes
**Objective:** Fix aggressive preprocessing that removes critical signal (`"Help me!"` → `['help']`)

#### Steps

1. **Confirm root cause**

   ```bash
   source .venv/Scripts/activate
   python -c "import sys; sys.path.append('src'); \
   ```

from disaster\_classifier.data.preprocessor import tokenize;&#x20;
print(tokenize('Help me!'))"

````

2. **Fix `tokenize()`**
- Edit `src/disaster_classifier/data/preprocessor.py` to retain disaster-critical tokens.

3. **Test fixed preprocessing**
```bash
python -c "import sys; sys.path.append('src'); \
from disaster_classifier.data.preprocessor import tokenize; \
print(tokenize('Help me!')); print(tokenize('Save us')); print(tokenize('We need help'))"
````

4. **Retrain with fixed preprocessing**

   ```bash
   python scripts/create_baseline_model.py --out models/fixed_preprocessing.pkl
   ```

5. **Sanity check**

   ```bash
   python -c "import sys; sys.path.append('src'); \
   ```

from disaster\_classifier.models.pipeline import load\_model;&#x20;
m=load\_model('models/fixed\_preprocessing.pkl');&#x20;
print(m.predict(\['Help me!']))"

````

6. **Re-optimize (if needed)**
```bash
python scripts/test_hyperparameters.py --preprocessing fixed
````

#### Success Criteria

* `"Help me!"` tokenizes to include `me` and punctuation context where helpful.
* Model trained on fixed preprocessing correctly flags critical cases.
* Measurable lift on targeted labels.

#### Implementation Results

**✅ Completed:** Disaster-aware stopword filtering in `preprocessor.py`.

**Code sketch:**

```python
# DISASTER-AWARE stopword removal
disaster_critical = {'me', 'us', 'we', 'i', 'my', 'our', 'help', 'please', 'save', 'rescue'}
tokens = [t for t in tokens if (t.lower() not in STOPWORDS_SET) or (t.lower() in disaster_critical)]
```

**Tests:**

* `"Help me"` → `['help', 'me']` ✅ (was `['help']`)
* `"Save us"` → `['save', 'us']` ✅ (was `['save']`)
* `"We need help"` → `['we', 'need', 'help']` ✅ (was `['need', 'help']`)

**Status:** Fix implemented and unit-tested. Ready for retraining.

---

### Phase 2: Class Imbalance Testing (If Needed)

**Duration:** 30 minutes
**Objective:** Only proceed if Phase 1 doesn’t fully resolve the issue.

#### Hypotheses & Tests

* **H1 (Imbalance root cause):** Compare optimized baseline vs class-weighted model.
  Metrics: per-label recall, F1, precision.
* **H2 (Preprocessing too aggressive):** **Confirmed**, compare before vs after.
  Metric: critical-case accuracy.
* **H3 (Architecture mismatch):** RF vs SVM/linear models comparison.
* **H4 (Features insufficient):** Add message length, urgency lexicons, etc.

#### Experiment Driver

```bash
python scripts/establish_baseline.py --model baseline --output results/baseline_results.json
python scripts/test_hypothesis.py --hypothesis class_imbalance --method class_weighting
python scripts/test_hypothesis.py --hypothesis preprocessing --method conservative_preprocessing
python scripts/test_hypothesis.py --hypothesis architecture --method svm_classifier
python scripts/test_hypothesis.py --hypothesis features --method enhanced_features
python scripts/validate_results.py --results results/ --significance_level 0.05
```

---

### Phase 3: Model Selection & Validation (10–15 minutes)

#### Step 1: Pick the winner

Prioritize:

* **Recall (positive cases)**: >25% target
* **F1 (positive cases)**: >20% target
* **Macro averages**: >60% target

#### Step 2: Integrate into web app

```bash
# Point to best model
# Edit app/config.py

# Smoke test
cd app && python app.py
# Visit http://127.0.0.1:3000
```

---

### Phase 4: Advanced Experimentation (Optional, 30–45 minutes)

If further lift needed:

```bash
python scripts/systematic_testing_framework.py
python scripts/train_model.py data/02_stg/stg_disaster_response.db models/custom_model.pkl
# choose custom experiment
```

Tune:

* `ngram_range`: (1,1), (1,2), (2,2)
* RandomForest params
* Sampling ratios
* Per-label decision thresholds

---

## Expected Results (based on legacy baseline)

### Legacy Baseline

* Recall (positive): 4–8% ❌
* F1 (positive): 7–14% ❌
* Train time: \~7 min ⚠️
* Model size: \~561 MB ⚠️

### With Improved Sampling / Weighting

* Recall: 20–30% ✅
* Balanced precision/recall ✅
* Train time: <5 min ✅
* Model size: manageable ✅

---

## Success Criteria (Next Work Session)

### Minimum

* Recall >15% (positive) ✅
* F1 >12% (positive) ✅
* Train time <5 min ✅
* Web app functional ✅

### Excellent

* Recall >25% (positive) 🎯
* F1 >20% (positive) 🎯
* Train time <3 min 🎯
* Improvements across most of the 36 categories 🎯

---

## Pro Tips for Analysis

**Key metrics**

1. Recall for class `1` (positives)
2. Macro F1
3. Labels with <10% recall
4. Training time and model size

**Red flags**

* Recall <10% after interventions
* Precision <70%
* Train time >10 min
* Model size >1 GB

**Success tells**

* Recall >20%, F1 >15%
* Train time <5 min
* Predictions make sense on critical phrases

---

## Implementation Timeline

### Immediate (105 minutes total)

* **Phase 1 – Preprocessing Fix:** 30 min
* **Phase 2 – Imbalance Testing (if needed):** 30 min
* **Phase 3 – Analysis & Portfolio Demo:** 45 min

### Completed Implementation

* ✅ Multi-label sampling solutions scaffolded
* ✅ `samplers.py` updated for multilabel awareness
* ✅ Documentation and validation scripts created
* ✅ Noted that “baseline” used prior optimized params on misprocessed data

### Next Priorities

* ✅ Root cause identified (preprocessing)
* 🎯 Apply fix (done), retrain, and validate
* 🎯 Portfolio demo with clear before/after narrative

---

## Latest Implementation Progress (Session Update)

### 🎯 Class Weighting — **COMPLETED**

**Highlights**

* Pipeline updated (`src/disaster_classifier/models/pipeline.py`)

  * `create_pipeline(use_class_weights=True)`
  * `create_pipeline_with_custom_weights()`
* Production scripts:

  * `scripts/validate_multilabel_sampling.py`
  * `scripts/create_model_with_weighting.py`
* System validation:

  * Weights computed for all 36 labels in \~0.02s
  * Extreme imbalance confirmed (e.g., `offer` \~218:1, `shops` \~216:1)
  * Trained weighted model in \~199s
  * Results saved to `data/04_fct/fct_class_weighted_validation_prediction_results.csv`

**Notes**

* ✅ Class weighting: stable and production-ready
* ⚠️ ML-SMOTE: refine to operate on vectorized space, not raw text
* ⚠️ Random oversampling: fix dimensional mismatches
* ✅ Label Powerset: works but no lift on current data snapshot

**Immediate next steps**

1. Build **baseline** and **weighted** production models
2. Compare per-label recall/F1; document gains
3. Swap into Flask app; smoke test end-to-end

**Expected**

* Better recall on severely imbalanced labels (`offer`, `shops`, `tools`, `fire`)
* Maintain overall accuracy; low overhead

---

## Phase 3: Results Analysis and Portfolio Demonstration

### 1) Problem–Solution Narrative

* **Problem**: `"Help me!"` failed despite tuned hyperparameters.
* **Analysis**: Aggressive preprocessing removed critical signal.
* **Solution**: Disaster-aware tokenization; retrain; optional class weighting.
* **Result**: Critical cases now correctly classified; measurable lift.

### 2) Skills Demonstrated

* Data science: hypothesis-driven diagnosis and validation
* ML engineering: pipelines, experiment tracking, deployment
* Code quality: modular, documented, tested
* Business impact: better detection of urgent messages

### 3) Working System Showcase

```bash
python scripts/demonstrate_system.py --showcase critical_cases
# Shows: "Help me!" correctly classified
# Before/after comparison
# Real-time classification in Flask app
```

### 4) Documentation Artifacts

* Experimental design and stats validation
* Per-label improvements with CIs
* Operational runbook and failure modes

**Portfolio Checklist**

* [ ] Clear problem–solution narrative
* [ ] End-to-end demo
* [ ] Experimental methodology
* [ ] Quantified improvements
* [ ] Clean, production-ready code
* [ ] Comprehensive docs

---