# Codebase Research Findings — Legacy Code & AI-Thrash Analysis

**Date**: 2026-01-26  
**Purpose**: Systematic analysis to inform cleanup decisions  
**Status**: In Progress

---

## Executive Summary

This document contains findings from systematic analysis of the codebase to identify:
- Legacy code that can be safely archived or removed
- AI-generated code that adds surface area without value
- Duplicate functionality
- Unused code
- Code that's critical to keep

**Decision Matrix**: See [Decision Matrix](#decision-matrix) section below.

---

## Phase 1: Entry Points & Execution Paths

### Critical Path (Production)

**Data Processing → Model Training → Application**

1. **Data Processing**
   - `scripts/process_data.py` - ✅ **CRITICAL** - Referenced in README, CLAUDE.md, AGENTS.md
   - Entry point: `python scripts/process_data.py <args>`

2. **Model Training**
   - `scripts/04_create_production_model.py` - ✅ **CRITICAL** - Referenced in README, CLAUDE.md, AGENTS.md
   - `scripts/03_create_experimental_model.py` - ✅ **ACTIVE** - Referenced in README
   - Entry points: Direct execution

3. **Application**
   - `run.py` - ✅ **CRITICAL** - Main entry point, referenced everywhere
   - `app/app.py` - ✅ **CRITICAL** - Flask factory, imported by run.py

### Experimental/Testing Entry Points

- `scripts/01_test_sampling_strategies.py` - ✅ **ACTIVE** - Referenced in README, CLAUDE.md
- `scripts/02_test_hyperparameters.py` - ✅ **ACTIVE** - Referenced in scripts/README.md
- `scripts/compare_models.py` - ✅ **ACTIVE** - Referenced in README, CLAUDE.md
- `scripts/test_experimental_model.py` - ⚠️ **UNCERTAIN** - Not in README, may be utility

### Analysis Scripts (Entry Points)

- `scripts/evaluate_hierarchy.py` - ⚠️ **UNCERTAIN** - Not in main README
- `scripts/optimize_thresholds.py` - ⚠️ **UNCERTAIN** - Not in main README
- `scripts/validate_production_model.py` - ⚠️ **UNCERTAIN** - Not in main README
- `scripts/system_validation.py` - ✅ **ACTIVE** - Referenced in CLAUDE.md, AGENTS.md

---

## Phase 2: Import Dependency Analysis

### App Dependencies (Flask Application)

**What the Flask app actually imports from `disasterproject`**:
- `disasterproject.utils.config` - Configuration constants
- `disasterproject.utils.metrics_io` - Metrics CSV reading
- `disasterproject.hierarchy` - Hierarchy post-processing (`apply_hierarchy`, `count_violations`)

**Status**: ✅ **MINIMAL DEPENDENCIES** - App only uses core utilities and hierarchy processing.

### Scripts That Import disasterproject

**26 scripts import from `disasterproject` package**:
- All training scripts (01-04)
- All comparison scripts
- All optimization scripts
- All validation scripts
- Analysis scripts

**Status**: ✅ **EXPECTED** - Scripts use the core package as intended.

### Scripts That Import Other Scripts

**No cross-script imports found** - Scripts are independent entry points, which is good architecture.

---

## Phase 3: Duplicate Functionality

### Comparison Scripts (4 found)

1. **`compare_models.py`** - ✅ **KEEP** - Referenced in README, CLAUDE.md
   - Purpose: Compare experiment results
   - Status: Active, documented

2. **`compare_csv_models.py`** - ⚠️ **REVIEW** - Only in scripts/README.md
   - Purpose: "Enhanced model comparison tool for CSV prediction results"
   - Status: May be duplicate of compare_models.py

3. **`compare_vocabulary_models.py`** - ❓ **UNKNOWN** - Not documented
   - Purpose: Unknown
   - Status: Needs investigation

4. **`compare_child_alone.py`** - ⚠️ **REVIEW** - Not in main README
   - Purpose: "Child alone label analysis" (per README architecture section)
   - Status: Specialized analysis tool

**Recommendation**: Investigate if compare_csv_models.py and compare_vocabulary_models.py are truly needed or can be consolidated.

### Optimization Scripts (3 found)

1. **`optimize_thresholds.py`** - ⚠️ **REVIEW** - Not in main README
   - Purpose: Threshold optimization
   - Status: Referenced in README architecture section

2. **`optimize_all_thresholds.py`** - ❓ **UNKNOWN** - Not documented
   - Purpose: Unknown
   - Status: Needs investigation

3. **`optimize_critical_thresholds_inc1.py`** - ❓ **UNKNOWN** - Not documented
   - Purpose: Unknown (name suggests incremental version)
   - Status: Likely legacy/incremental work

**Recommendation**: Consolidate threshold optimization scripts. The "inc1" suffix suggests incremental work that may be obsolete.

### Validation Scripts (6 found)

1. **`validate_production_model.py`** - ⚠️ **REVIEW** - Not in main README
2. **`validate_multilabel_sampling.py`** - ⚠️ **REVIEW** - In scripts/README.md
3. **`validate_threshold_optimization_results.py`** - ❓ **UNKNOWN** - Not documented
4. **`validate_ml_execution_environment.py`** - ❓ **UNKNOWN** - Not documented
5. **`system_validation.py`** - ✅ **KEEP** - Referenced in CLAUDE.md, AGENTS.md
6. **`test_deployment_scenarios.py`** - ❓ **UNKNOWN** - Not documented

**Recommendation**: Many validation scripts may be one-off utilities. Review which are actually used.

### Data Preparation Scripts (2 found)

1. **`process_data.py`** - ✅ **CRITICAL** - Main data processing, referenced everywhere
   - **Implementation**: Direct CSV loading, cleaning, and DB saving
   - **Status**: Active, used in all documentation

2. **`prepare_data.py`** - ⚠️ **ALTERNATIVE INTERFACE** - Wrapper around ETL pipeline
   - **Implementation**: Calls `disasterproject.data.loader.prepare_data()` which uses `run_etl_pipeline()`
   - **Status**: Not referenced in main README, may be newer alternative interface
   - **Note**: Has a bug (line 76 references undefined `log_level` variable)

**Recommendation**: 
- `process_data.py` is the active one (referenced everywhere)
- `prepare_data.py` appears to be an alternative/newer interface that's not actively used
- **Action**: Archive `prepare_data.py` or fix the bug and update documentation if it's meant to replace `process_data.py`

---

## Phase 4: Explicit Legacy Indicators

### Archive Directory

**Location**: `scripts/archive/`

**Files Found**:
- `compare_results.py` - Legacy result comparison
- `run_all_experiments.py` - Legacy experiment runner
- `systematic_testing_framework.py` - Legacy testing framework
- `train_classifier_original.py` - Original training script
- `train_classifier.py` - Legacy training script
- `validate_structure.py` - Legacy structure validation

**Status**: ✅ **ALREADY ARCHIVED** - These are properly quarantined.

### Legacy Path References

**Found**: `src/disasterproject/utils/experimental_paths.py` contains `ExperimentalPathManager` that handles both legacy and new path structures.

**Status**: ⚠️ **TRANSITION CODE** - This is transition infrastructure. Once migration is complete, this can be simplified.

### Legacy Model References

**Found**: `model/README.md` mentions:
- Legacy fallback filename: `optimized_critical_thresholds.json`
- Removed files: `model/parameters.json` and `model/class_weights.json` (removed 2026-01-22)

**Status**: ✅ **DOCUMENTED** - Legacy references are documented.

---

## Phase 5: Unused Code Detection

### Scripts Not Referenced in Documentation

**Not in README.md, CLAUDE.md, AGENTS.md, or scripts/README.md**:

1. `analyze_vocabulary_distribution.py` - ❓ **UNKNOWN**
2. `compare_csv_models.py` - ⚠️ (in scripts/README.md only)
3. `compare_vocabulary_models.py` - ❓ **UNKNOWN**
4. `create_frozen_eval_ids.py` - ❓ **UNKNOWN**
5. `deployment_health_check.py` - ❓ **UNKNOWN**
6. `eda_functions.py` - ⚠️ (in scripts/README.md only, but as utility)
7. `ensure_venv.py` - ❓ **UNKNOWN**
8. `estimate_search_time.py` - ❓ **UNKNOWN**
9. `migrate_experimental_paths.py` - ⚠️ **TRANSITION** - Migration utility
10. `model_naming_utility.py` - ❓ **UNKNOWN**
11. `optimize_all_thresholds.py` - ❓ **UNKNOWN**
12. `optimize_critical_thresholds_inc1.py` - ❓ **UNKNOWN**
13. `prepare_15k_model_for_promotion.py` - ❓ **UNKNOWN** (name suggests one-off task)
14. `prepare_data.py` - ❓ **UNKNOWN**
15. `promote_model.py` - ⚠️ **ACTIVE** - In scripts/README.md promotion workflow
16. `test_deployment_scenarios.py` - ❓ **UNKNOWN**
17. `test_experimental_model.py` - ⚠️ **UNCERTAIN**
18. `validate_ml_execution_environment.py` - ❓ **UNKNOWN**
19. `validate_threshold_optimization_results.py` - ❓ **UNKNOWN**
20. `visualize_performance.py` - ⚠️ (in scripts/README.md only)

**Note**: Scripts marked with ⚠️ may be utilities that are useful but not part of main workflow.

---

## Phase 6: Test Coverage Analysis

### Test Files Found

**12 test files in `tests/` directory**:
- `test_smoke.py` - Smoke tests
- `test_app_smoke.py` - App smoke tests
- `test_csrf_smoke.py` - CSRF tests
- `test_flask_standardized.py` - Flask standardization
- `test_gdrive_deployment.py` - Google Drive deployment
- `test_hierarchy.py` - Hierarchy processing
- `test_metrics_io.py` - Metrics I/O
- `test_optimization.py` - Optimization tests
- `test_perf.py` - Performance tests
- `test_request_logging_utils.py` - Request logging
- `test_security.py` - Security tests
- `test_compare_models_paths.py` - Compare models paths
- `test_thresholds_alignment.py` - Threshold alignment

**Disabled tests**:
- `test_security.py.disabled` - Disabled security test
- `test_train_classifier.py.disabled` - Disabled training test

### Script Test Coverage

**No direct tests found for individual scripts** - Tests focus on:
- Core package functionality (`src/disasterproject/`)
- Flask application
- Integration scenarios

**Status**: ⚠️ **SCRIPTS NOT DIRECTLY TESTED** - Scripts are tested indirectly through package tests, but individual script logic is not unit tested.

---

## Phase 7: Documentation Consistency

### Scripts Mentioned in README but Wrong Names

**scripts/README.md** mentions:
- `create_production_model.py` - ❌ **WRONG NAME** - Actual file is `04_create_production_model.py`
- `test_sampling_strategies.py` - ❌ **WRONG NAME** - Actual file is `01_test_sampling_strategies.py`

**Status**: ⚠️ **DOCUMENTATION OUT OF SYNC** - scripts/README.md needs updating.

### Missing Documentation

Many scripts exist but are not documented in any README:
- See "Unused Code Detection" section above

### Documentation Inconsistencies

**Script name mismatches in scripts/README.md**:
- Mentions `create_production_model.py` but actual file is `04_create_production_model.py`
- Mentions `test_sampling_strategies.py` but actual file is `01_test_sampling_strategies.py`

**References to non-existent scripts**:
- `AGENTS.md` and `CLAUDE.md` reference `06_create_lightweight_model.py` which **does not exist**
- According to dev notes, this script was removed on 2025-09-17
- **Status**: ⚠️ **DOCUMENTATION OUTDATED** - Needs cleanup

---

## Phase 8: AI-Thrash Indicators

### Rapid Generation Patterns

*Analysis in progress - checking git history for rapid file creation*

### Low Understanding Signals

**Scripts with minimal docstrings or unclear purpose**:
- `optimize_critical_thresholds_inc1.py` - Name suggests incremental work
- `prepare_15k_model_for_promotion.py` - Name suggests one-off task
- `compare_vocabulary_models.py` - No documentation
- `analyze_vocabulary_distribution.py` - No documentation

### Surface Area Without Value

**Potential candidates**:
- Multiple threshold optimization scripts (3 variants)
- Multiple comparison scripts (4 variants)
- Multiple validation scripts (6 variants)
- Utility scripts that may have been one-off tasks

---

## Decision Matrix

### ✅ KEEP (Critical/Active)

| File | Reason | Notes |
|------|--------|-------|
| `run.py` | Main app entry point | Referenced everywhere |
| `app/app.py` | Flask factory | Core application |
| `scripts/process_data.py` | Data processing | Critical path |
| `scripts/04_create_production_model.py` | Production model | Critical path |
| `scripts/03_create_experimental_model.py` | Experimental model | Active workflow |
| `scripts/01_test_sampling_strategies.py` | Testing | Referenced in docs |
| `scripts/02_test_hyperparameters.py` | Testing | Referenced in docs |
| `scripts/compare_models.py` | Analysis | Referenced in docs |
| `scripts/system_validation.py` | Validation | Referenced in docs |
| `scripts/promote_model.py` | Workflow | Promotion workflow documented |

### ⚠️ REVIEW (Needs Investigation)

| File | Reason | Action Needed |
|------|--------|---------------|
| `scripts/compare_csv_models.py` | May duplicate compare_models.py | Check if truly needed |
| `scripts/compare_vocabulary_models.py` | No documentation | Understand purpose |
| `scripts/compare_child_alone.py` | Specialized tool | Verify if needed |
| `scripts/optimize_thresholds.py` | Not in main README | Verify if active |
| `scripts/optimize_all_thresholds.py` | No documentation | Check vs optimize_thresholds.py |
| `scripts/optimize_critical_thresholds_inc1.py` | "inc1" suggests incremental | Likely obsolete |
| `scripts/validate_production_model.py` | Not in main README | Verify usage |
| `scripts/validate_multilabel_sampling.py` | In scripts/README only | Verify usage |
| `scripts/test_experimental_model.py` | Not in main README | Verify usage |
| `scripts/evaluate_hierarchy.py` | Not in main README | Verify usage |
| `scripts/prepare_data.py` | May duplicate process_data.py | Check if needed |
| `scripts/visualize_performance.py` | In scripts/README only | Verify usage |
| `scripts/eda_functions.py` | Utility functions | Check if imported anywhere |

### ❓ INVESTIGATE (Unknown Purpose)

| File | Reason | Findings |
|------|--------|----------|
| `scripts/analyze_vocabulary_distribution.py` | No documentation | **Purpose**: Analyzes vocabulary from trained model, extracts vocabulary, document frequencies, provides recommendations for max_features/min_df/max_df. **Status**: Utility script, may be useful for model optimization. |
| `scripts/create_frozen_eval_ids.py` | No documentation | **Purpose**: Unknown, but name suggests creating frozen evaluation IDs for consistent train/test splits. **Status**: May be used by other scripts. |
| `scripts/deployment_health_check.py` | No documentation | **Purpose**: Unknown. **Status**: May be deployment utility. |
| `scripts/ensure_venv.py` | No documentation | **Purpose**: Likely ensures virtual environment is set up. **Status**: Utility script. |
| `scripts/estimate_search_time.py` | No documentation | **Purpose**: Unknown. **Status**: May be for hyperparameter search time estimation. |
| `scripts/model_naming_utility.py` | No documentation | **Purpose**: Likely utility for model naming conventions. **Status**: May be used by promotion workflow. |
| `scripts/prepare_15k_model_for_promotion.py` | Name suggests one-off | **Purpose**: One-off script for 2025-11-06 model promotion. Hardcoded paths to specific experiment date. **Status**: ✅ **ONE-OFF COMPLETED** - Safe to archive. |
| `scripts/test_deployment_scenarios.py` | No documentation | **Purpose**: Unknown. **Status**: May be deployment testing utility. |
| `scripts/validate_ml_execution_environment.py` | No documentation | **Purpose**: Likely validates ML environment setup. **Status**: Utility script. |
| `scripts/validate_threshold_optimization_results.py` | No documentation | **Purpose**: Validates threshold optimization results. **Status**: May be used after threshold optimization. |
| `scripts/compare_vocabulary_models.py` | No documentation | **Purpose**: Compares vocabulary-optimized models, generates comparison report. Analyzes vocabulary-limited models. **Status**: Specialized analysis tool, may be useful. |

### 🗑️ ARCHIVE/DELETE (Likely Safe)

| File | Reason | Action |
|------|--------|--------|
| `scripts/optimize_critical_thresholds_inc1.py` | Incremental work for "Increment 1" model, likely obsolete | **Archive** - Specific to old model version |
| `scripts/prepare_15k_model_for_promotion.py` | One-off task for 2025-11-06, hardcoded paths, completed | **Archive** - One-off completed task |
| `scripts/migrate_experimental_paths.py` | Transition utility for path migration | **Archive** - If migration is complete (check if still needed) |
| `scripts/prepare_data.py` | Alternative interface, not used, has bug | **Archive or Fix** - If not replacing process_data.py, archive it |
| `scripts/optimize_all_thresholds.py` | May duplicate optimize_thresholds.py | **Review** - Check if truly different from optimize_thresholds.py |

### ✅ ALREADY ARCHIVED

| Location | Status |
|----------|--------|
| `scripts/archive/` | Properly quarantined |

---

## Key Findings Summary

### High-Confidence Actions

1. **Archive one-off scripts**:
   - `prepare_15k_model_for_promotion.py` - Completed one-off task
   - `optimize_critical_thresholds_inc1.py` - Incremental work for obsolete model

2. **Fix documentation**:
   - Remove references to non-existent `06_create_lightweight_model.py` from AGENTS.md and CLAUDE.md
   - Fix script name mismatches in scripts/README.md

3. **Review duplicate functionality**:
   - Consolidate threshold optimization scripts (3 variants)
   - Review comparison scripts (4 variants) - may serve different purposes
   - Decide on `prepare_data.py` vs `process_data.py`

### Medium-Confidence Actions

4. **Archive transition code** (if migration complete):
   - `migrate_experimental_paths.py` - Check if experimental path migration is done
   - Simplify `ExperimentalPathManager` if legacy paths no longer needed

5. **Review utility scripts**:
   - Many utility scripts may be one-off tools
   - Consider archiving if not actively used

### Low-Confidence Actions (Need More Investigation)

6. **Investigate specialized scripts**:
   - Vocabulary analysis scripts may be useful for model optimization
   - Validation scripts may be part of quality gates
   - Need to understand actual usage patterns

## Next Steps

1. ✅ **Dependency analysis** - Completed (scripts are independent entry points)
2. ✅ **Test coverage check** - Completed (scripts not directly tested)
3. ⏳ **Git history analysis** - Find rapid generation patterns (optional)
4. ⏳ **Code review of "INVESTIGATE" scripts** - Understand their purpose (partially done)
5. ⏳ **Consolidation planning** - Plan how to merge duplicate functionality
6. ✅ **Documentation audit** - Identified issues (needs fixing)

---

## Research Methodology

This analysis used:
- File system enumeration
- Documentation cross-referencing (README.md, CLAUDE.md, AGENTS.md, scripts/README.md)
- Pattern matching for duplicate functionality
- Archive directory inspection
- Entry point identification

**Still needed**:
- Import dependency graph
- Test coverage mapping
- Git history analysis
- Code review of uncertain scripts
