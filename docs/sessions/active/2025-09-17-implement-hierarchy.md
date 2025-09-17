# Hierarchy Fix Agent: Enforce Parent-Child Consistency and Boost Safety Recall

**Date**: 2025-09-17
**Status**: Active
**Priority**: High
**Estimated Duration**: 2–3 hours
**Tags**: multi-label, hierarchy, evaluation, safety

## 🎯 Objective

Implement a lightweight hierarchy post-processor that enforces parent ≥ child probabilities and child ⇒ parent activation, with gentle threshold softening for life-critical labels. Produce a before/after report that highlights Safety Recall improvements with minimal macro-F1 impact. Ground choices in current label set and findings. &#x20;

## 📋 Success Criteria

* [ ] Shipping `apply_hierarchy(...)` utility wired into eval/prediction path
* [ ] Configured taxonomy and critical-label set checked into `config.py`
* [ ] Before/after eval showing:

  * [ ] Safety Recall (avg recall on critical labels) improves meaningfully
  * [ ] Parent≥Child violations drop to 0 per 1k predictions
  * [ ] Macro F1 change within −0 to −2 points
* [ ] README note explaining exclusion and rationale for `child_alone` given 0 positives.&#x20;

## 🔍 Context

Your validation showed catastrophic recall on rare but urgent categories despite strong macro metrics. We will enforce hierarchical consistency and nudge thresholds on life-critical leaves to lift recall with minimal retraining. `TARGET_COLUMNS` are centralized in `config.py`. `child_alone` has 0 positives in source and training data, so constraints should not involve it. &#x20;

## 📝 Requirements

### Functional Requirements

* Add taxonomy: parent → children for aid, infrastructure, weather, and related groups
* Add critical label set used for threshold softening
* Apply hierarchy post-processing to probabilities and binary decisions
* Exclude `child_alone` from constraints but keep it visible in outputs and docs&#x20;

### Technical Requirements

* Single module: `src/disasterproject/hierarchy.py`
* Config keys in `config.py`: `TAXONOMY`, `CRITICAL_LABELS`, `EXCLUDE_FROM_CONSTRAINTS`&#x20;
* Integration point: evaluation and batch inference path only
* Unit tests in `tests/test_hierarchy.py`

### Quality Requirements

* Zero parent≥child probability violations post-fix
* Deterministic behavior, no external dependencies
* Clear logging of adjustments and any forced parent activations

## 🛠️ Approach

1. **Define taxonomy and critical labels**
   Add to `config.py` based on your label list:

* Parents: `aid_related`, `infrastructure_related`, `weather_related`, `related`
* Children: medical, water, food, shelter, infra, weather subclasses, request/offer/direct\_report
* Critical: `medical_help`, `medical_products`, `search_and_rescue`, `water`, `food`, `security`
  Exclude `child_alone`. &#x20;

2. **Implement `apply_hierarchy(...)`**
   Rules:

* Monotone probs: `p(parent) = max(p(parent), max_children)` and `p(child) ≤ p(parent)`
* Decision logic: if any child predicts 1 ⇒ force parent 1
* Threshold softening: reduce thresholds by \~0.10 for critical labels only

3. **Wire into evaluation**

* Run baseline eval and record metrics
* Run with hierarchy fixer and record after metrics
* Log “violations per 1k preds” before and after

4. **Document `child_alone`**
   Short README section: zero positives in dataset and training, excluded from constraints to avoid spurious flips. Keep predictions visible.&#x20;

## 📊 Acceptance Criteria

* `apply_hierarchy` is unit-tested and integrated
* Before/after table included in report with:

  * Safety Recall ↑
  * Violations = 0
  * Macro F1 within target band
* README updated to explain hierarchy logic and `child_alone` rationale with dataset counts.&#x20;

## 🔗 Related Work

* Data Quality Agent write-up identifying zero-positive `child_alone` and systemic rare-class failures.&#x20;
* Centralized label config and paths in `config.py`.&#x20;

## 📈 Metrics

* **Safety Recall**: mean recall over critical labels
* **Parent≥Child Violations**: count per 1k predictions
* **Macro F1 Δ**: after − baseline
* **Critical FN Count**: number of false negatives for critical labels at chosen thresholds

## 🚨 Risks & Mitigations

| Risk                                  | Impact | Prob.  | Mitigation                                                            |                               |
| ------------------------------------- | ------ | ------ | --------------------------------------------------------------------- | ----------------------------- |
| Precision drop on critical labels     | Medium | Medium | Tune per-class thresholds via PR curves to maintain a precision floor |                               |
| Taxonomy edges mismatch data          | Medium | Low    | Compute P(child=1                                                     | parent=1); adjust edges < 0.5 |
| Over-activation of parents            | Low    | Medium | Cap parent lift to max(child) and monitor confusion slices            |                               |
| Confusion from `child_alone` handling | Low    | Low    | Explicit README note with counts and exclusion rationale              |                               |

## 📄 Deliverables

* [ ] `src/disasterproject/hierarchy.py` with `apply_hierarchy(...)`
* [ ] `config.py` additions: `TAXONOMY`, `CRITICAL_LABELS`, `EXCLUDE_FROM_CONSTRAINTS`&#x20;
* [ ] `tests/test_hierarchy.py` unit tests
* [ ] Eval script update and **before/after** metrics table in report
* [ ] README update including `child_alone` note with dataset counts.&#x20;

## hierarchy for config.py

# Label taxonomy (parent -> children)
TAXONOMY = {
    "aid_related": [
        "medical_help","medical_products","search_and_rescue","water","food",
        "shelter","clothing","money","other_aid"
    ],
    "infrastructure_related": [
        "transport","buildings","electricity","tools","hospitals","shops",
        "aid_centers","other_infrastructure"
    ],
    "weather_related": ["floods","storm","fire","earthquake","cold","other_weather"],
    # Treat "related" as a root with siblings that can co-occur
    "related": ["request","offer","direct_report"],
}

# Critical leaves (use softer thresholds in the fixer)
CRITICAL_LABELS = {
    "medical_help","medical_products","search_and_rescue","water","food","security"
}

# Labels excluded from hierarchy constraints (documented data limitations)
EXCLUDE_FROM_CONSTRAINTS = {"child_alone"}  # 0 positives in source + train data
