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
- `scripts/optimize_hierarchy_threshold_reduction.py` - ✅ **KEEP** - Hierarchy parameter optimization
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

2. **`compare_csv_models.py`** - ✅ **ARCHIVED** 2026-01-26 - Only in scripts/README.md
   - Purpose: "Enhanced model comparison tool for CSV prediction results" (arbitrary CSV file comparison)
   - Context: Designed to help with UI/UX issues in the portfolio app (portfolio project requirement)
   - Status: Not actively used in main workflow, archived as part of cleanup
   - **Action**: ✅ Archived - More flexible than compare_models.py (can compare arbitrary CSV files), but not actively used. compare_models.py handles the standard comparison use case. Archived despite UI/UX purpose as part of cleanup pass.

3. **`compare_vocabulary_models.py`** - ✅ **KEEP** - Specialized tool for vocabulary experiments
   - Purpose: Compare vocabulary-optimized models, generates markdown reports
   - Status: Specialized analysis tool for vocabulary size optimization experiments
   - **Action**: ✅ KEEP - Distinct purpose, used in vocabulary optimization workflow

4. **`compare_child_alone.py`** - ✅ **KEEP** - Specialized diagnostic tool
   - Purpose: "Child alone label analysis" (per README architecture section)
   - Status: Specialized diagnostic tool for child_alone category behavior
   - **Action**: ✅ KEEP - Referenced in main README, specialized diagnostic tool

**Recommendation**: ✅ **COMPLETED** 2026-01-26 - Archived compare_csv_models.py as unused one-off tool. Other comparison scripts serve distinct purposes and are kept.

### Optimization Scripts (3 found)

1. **`optimize_hierarchy_threshold_reduction.py`** (renamed from `optimize_thresholds.py`) - ✅ **KEEP**
   - Purpose: Optimizes critical threshold reduction parameter for hierarchy post-processing
   - Status: Referenced in README, distinct from per-category optimization

2. **`optimize_per_category_thresholds.py`** (renamed from `optimize_all_thresholds.py`) - ✅ **KEEP**
   - Purpose: Optimizes individual thresholds for all 36 categories using precision-recall curves
   - Status: Referenced in threshold-file-naming.md, distinct from hierarchy optimization

3. **`optimize_critical_thresholds_inc1.py`** - ✅ **ARCHIVED** 2026-01-26
   - Purpose: Incremental threshold optimization for obsolete "Increment 1" model version
   - Status: Archived to `scripts/archive/`

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

2. **`prepare_data.py`** - ✅ **ARCHIVED** 2026-01-26 - Wrapper around ETL pipeline
   - **Implementation**: Calls `disasterproject.data.loader.prepare_data()` which uses `run_etl_pipeline()`
   - **Status**: Not referenced in main README, unused alternative interface
   - **Investigation**: No bugs found (false positive), not imported or called anywhere
   - **Action**: ✅ Archived - Redundant with `process_data.py` which is the active, documented script

**Recommendation**: 
- `process_data.py` is the active one (referenced everywhere)
- `prepare_data.py` was an alternative/newer interface that was never adopted
- ✅ **COMPLETED**: Archived `prepare_data.py` as unused duplicate functionality

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
- `prepare_15k_model_for_promotion.py` - ✅ Added 2026-01-26 (one-off completed task)
- `optimize_critical_thresholds_inc1.py` - ✅ Added 2026-01-26 (incremental work for obsolete model)
- `prepare_data.py` - ✅ Added 2026-01-26 (unused alternative interface, duplicate of process_data.py)

**Status**: ✅ **PROPERLY QUARANTINED** - All legacy scripts archived.

### Legacy Path References

**Found**: `src/disasterproject/utils/experimental_paths.py` contains `ExperimentalPathManager` that handles both legacy and new path structures.

**Status**: ✅ **SIMPLIFIED** 2026-01-26 - Migration complete, legacy support removed. Now only uses `experiments/experimental_runs/<date>/` structure.

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
7. `ensure_venv.py` - ✅ **KEEP** - DX utility for AI agents (local vs Replit environment detection)
8. `estimate_search_time.py` - ❓ **UNKNOWN**
9. ~~`migrate_experimental_paths.py`~~ - ✅ **ARCHIVED** 2026-01-26 - Migration utility (migration completed)
10. `model_naming_utility.py` - ❓ **UNKNOWN**
11. ~~`optimize_all_thresholds.py`~~ - ✅ **RENAMED** to `optimize_per_category_thresholds.py`
12. ~~`optimize_critical_thresholds_inc1.py`~~ - ✅ **ARCHIVED** 2026-01-26
13. ~~`prepare_15k_model_for_promotion.py`~~ - ✅ **ARCHIVED** 2026-01-26 (one-off task completed)
14. ~~`prepare_data.py`~~ - ✅ **ARCHIVED** 2026-01-26
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
- ~~`create_production_model.py` - ❌ **WRONG NAME** - Actual file is `04_create_production_model.py`~~ ✅ **FIXED** 2026-01-26
- ~~`test_sampling_strategies.py` - ❌ **WRONG NAME** - Actual file is `01_test_sampling_strategies.py`~~ ✅ **FIXED** 2026-01-26

**Status**: ✅ **FIXED** - All script names now match actual filenames.

### Missing Documentation

Many scripts exist but are not documented in any README:
- See "Unused Code Detection" section above

### Documentation Inconsistencies

**Script name mismatches in scripts/README.md**:
- ~~Mentions `create_production_model.py` but actual file is `04_create_production_model.py`~~ ✅ **FIXED** 2026-01-26
- ~~Mentions `test_sampling_strategies.py` but actual file is `01_test_sampling_strategies.py`~~ ✅ **FIXED** 2026-01-26

**References to non-existent scripts**:
- ~~`AGENTS.md` and `CLAUDE.md` reference `06_create_lightweight_model.py` which **does not exist**~~ ✅ **FIXED** 2026-01-26
- According to dev notes, this script was removed on 2025-09-17
- **Status**: ✅ **CLEANED UP** - All references removed

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
| ~~`scripts/compare_csv_models.py`~~ | UI/UX tool for portfolio app | ✅ **ARCHIVED** 2026-01-26 |
| `scripts/compare_vocabulary_models.py` | Specialized tool for vocabulary experiments | ✅ **KEEP** - Distinct purpose |
| `scripts/compare_child_alone.py` | Specialized diagnostic tool | ✅ **KEEP** - Referenced in README |
| `scripts/optimize_hierarchy_threshold_reduction.py` | Hierarchy parameter optimization | ✅ **KEEP** - Distinct purpose |
| `scripts/optimize_per_category_thresholds.py` | Per-category threshold optimization | ✅ **KEEP** - Distinct purpose |
| ~~`scripts/optimize_critical_thresholds_inc1.py`~~ | "inc1" suggests incremental | ✅ **ARCHIVED** 2026-01-26 |
| `scripts/validate_production_model.py` | Not in main README | Verify usage |
| `scripts/validate_multilabel_sampling.py` | In scripts/README only | Verify usage |
| `scripts/test_experimental_model.py` | Not in main README | Verify usage |
| `scripts/evaluate_hierarchy.py` | Not in main README | Verify usage |
| ~~`scripts/prepare_data.py`~~ | Duplicated process_data.py | ✅ **ARCHIVED** 2026-01-26 |
| `scripts/visualize_performance.py` | In scripts/README only | Verify usage |
| `scripts/eda_functions.py` | Utility functions | Check if imported anywhere |

### ❓ INVESTIGATE (Unknown Purpose)

| File | Reason | Findings |
|------|--------|----------|
| `scripts/analyze_vocabulary_distribution.py` | No documentation | **Purpose**: Analyzes vocabulary from trained model, extracts vocabulary, document frequencies, provides recommendations for max_features/min_df/max_df. **Status**: Utility script, may be useful for model optimization. |
| `scripts/create_frozen_eval_ids.py` | No documentation | **Purpose**: Unknown, but name suggests creating frozen evaluation IDs for consistent train/test splits. **Status**: May be used by other scripts. |
| `scripts/deployment_health_check.py` | No documentation | **Purpose**: Unknown. **Status**: May be deployment utility. |
| `scripts/ensure_venv.py` | DX utility for AI agents | **Purpose**: Helps AI agents in Cursor distinguish between local dev (requires venv) vs Replit (no venv needed). **Status**: ✅ **KEEP** - DX quality-of-life improvement for dual-environment workflows. Original model was ~900 MB, requiring local training. |
| `scripts/estimate_search_time.py` | No documentation | **Purpose**: Unknown. **Status**: May be for hyperparameter search time estimation. |
| `scripts/model_naming_utility.py` | No documentation | **Purpose**: Likely utility for model naming conventions. **Status**: May be used by promotion workflow. |
| `scripts/prepare_15k_model_for_promotion.py` | Name suggests one-off | **Purpose**: One-off script for 2025-11-06 model promotion. Hardcoded paths to specific experiment date. **Status**: ✅ **ONE-OFF COMPLETED** - Safe to archive. |
| `scripts/test_deployment_scenarios.py` | No documentation | **Purpose**: Unknown. **Status**: May be deployment testing utility. |
| `scripts/validate_ml_execution_environment.py` | No documentation | **Purpose**: Likely validates ML environment setup. **Status**: Utility script. |
| `scripts/validate_threshold_optimization_results.py` | No documentation | **Purpose**: Validates threshold optimization results. **Status**: May be used after threshold optimization. |
| `scripts/compare_vocabulary_models.py` | Specialized tool | **Purpose**: Compares vocabulary-optimized models, generates comparison report. Analyzes vocabulary-limited models. **Status**: ✅ **KEEP** - Specialized analysis tool for vocabulary experiments. |

### 🗑️ ARCHIVE/DELETE (Likely Safe)

| File | Reason | Action | Status |
|------|--------|--------|--------|
| `scripts/optimize_critical_thresholds_inc1.py` | Incremental work for "Increment 1" model, likely obsolete | **Archive** - Specific to old model version | ✅ **ARCHIVED** 2026-01-26 |
| `scripts/prepare_15k_model_for_promotion.py` | One-off task for 2025-11-06, hardcoded paths, completed | **Archive** - One-off completed task | ✅ **ARCHIVED** 2026-01-26 |
| ~~`scripts/migrate_experimental_paths.py`~~ | Transition utility for path migration | ✅ **ARCHIVED** 2026-01-26 - Migration completed | ✅ **COMPLETED** |
| ~~`scripts/prepare_data.py`~~ | Alternative interface, not used | ✅ **ARCHIVED** 2026-01-26 - Unused duplicate of process_data.py |
| ~~`scripts/optimize_all_thresholds.py`~~ | Renamed to `optimize_per_category_thresholds.py` | ✅ **RENAMED** 2026-01-26 - Clarifies per-category purpose |
| ~~`scripts/compare_csv_models.py`~~ | UI/UX tool for portfolio app | ✅ **ARCHIVED** 2026-01-26 - Designed for UI/UX purposes in portfolio app, but not actively used in main workflow. compare_models.py handles standard use case. Archived as part of cleanup pass. |

### ✅ ALREADY ARCHIVED

| Location | Status |
|----------|--------|
| `scripts/archive/` | Properly quarantined |
| `scripts/archive/prepare_15k_model_for_promotion.py` | ✅ Archived 2026-01-26 |
| `scripts/archive/optimize_critical_thresholds_inc1.py` | ✅ Archived 2026-01-26 |
| `scripts/archive/compare_csv_models.py` | ✅ Archived 2026-01-26 |

---

## Key Findings Summary

### High-Confidence Actions

1. ✅ **Archive one-off scripts** (COMPLETED 2026-01-26):
   - `prepare_15k_model_for_promotion.py` - ✅ Archived
   - `optimize_critical_thresholds_inc1.py` - ✅ Archived

2. ✅ **Fix documentation** (COMPLETED 2026-01-26):
   - ✅ Removed references to non-existent `06_create_lightweight_model.py` from AGENTS.md and CLAUDE.md
   - ✅ Fixed script name mismatches in scripts/README.md

3. **Review duplicate functionality**:
   - Consolidate threshold optimization scripts (3 variants)
   - Review comparison scripts (4 variants) - may serve different purposes
   - Decide on `prepare_data.py` vs `process_data.py`

### Medium-Confidence Actions

4. ✅ **Archive transition code** (COMPLETED 2026-01-26):
   - ✅ `migrate_experimental_paths.py` - Archived (migration completed, 0 legacy artifacts)
   - ✅ `ExperimentalPathManager` - Simplified (removed legacy support, now only uses new structure)

5. **Review utility scripts**:
   - Many utility scripts may be one-off tools
   - Consider archiving if not actively used

### Low-Confidence Actions (Need More Investigation)

6. **Investigate specialized scripts**:
   - Vocabulary analysis scripts may be useful for model optimization
   - Validation scripts may be part of quality gates
   - Need to understand actual usage patterns

## Cleanup Actions Completed

**Date**: 2026-01-26  
**Branch**: `cleanup/documentation-and-one-off-scripts`

### Tier 1: Documentation Fixes ✅

1. **Removed references to non-existent script**:
   - `AGENTS.md`: Removed `python scripts/06_create_lightweight_model.py` from Model Training & Evaluation section
   - `CLAUDE.md`: Removed "Lightweight model" section referencing the non-existent script
   - Script was removed on 2025-09-17 per dev notes

2. **Fixed script name mismatches in `scripts/README.md`**:
   - Updated `create_production_model.py` → `04_create_production_model.py` (header and usage examples)
   - Updated `test_sampling_strategies.py` → `01_test_sampling_strategies.py` (header and usage examples)
   - All documentation now matches actual filenames

### Tier 2: Archived One-Off Scripts ✅

1. **Archived `scripts/prepare_15k_model_for_promotion.py`**:
   - Moved to `scripts/archive/prepare_15k_model_for_promotion.py`
   - One-off script for 2025-11-06 model promotion (completed task)
   - Hardcoded paths to specific experiment date

2. **Archived `scripts/optimize_critical_thresholds_inc1.py`**:
   - Moved to `scripts/archive/optimize_critical_thresholds_inc1.py`
   - Incremental threshold optimization for obsolete "Increment 1" model version
   - Not referenced in any documentation

3. **Updated `scripts/README.md`**:
   - Added entries for newly archived scripts in Archive Directory section

**Status**: All changes staged on branch `cleanup/documentation-and-one-off-scripts`, ready for commit.

### Tier 3: Archive Unused Alternative Interface ✅

**Date**: 2026-01-26  
**Branch**: `cleanup/tier3-prepare-data-investigation`

1. **Investigated `scripts/prepare_data.py`**:
   - ✅ **Bug check**: False positive - no bugs found (code is correct)
   - ✅ **Usage check**: Not imported or called anywhere in codebase
   - ✅ **Functionality comparison**: Duplicate of `process_data.py` but unused
   - **Findings**: Wrapper around ETL pipeline with argparse interface, but `process_data.py` is the active, documented script

2. **Archived `scripts/prepare_data.py`**:
   - Moved to `scripts/archive/prepare_data.py`
   - Unused alternative interface that duplicates `process_data.py`
   - Not referenced in main documentation (only in scripts/README.md)

3. **Updated documentation**:
   - Removed `prepare_data.py` entry from `scripts/README.md`
   - Updated research findings to reflect investigation results and archiving

**Status**: All changes staged on branch `cleanup/tier3-prepare-data-investigation`, ready for commit.

### Tier 4: Utility Script Investigation ✅

**Date**: 2026-01-26  
**Branch**: `cleanup/utility-scripts-investigation`

1. **Investigated utility scripts**:
   - ✅ **`eda_functions.py`**: KEEP - Used in notebooks (active and archived)
   - ✅ **`ensure_venv.py`**: KEEP - DX utility for AI agents in Cursor to distinguish between local dev (requires venv) vs Replit (no venv needed). Important context: Original model was ~900 MB, requiring local training instead of Replit, necessitating dual-environment support.
   - ✅ **`model_naming_utility.py`**: KEEP - Referenced in multiple docs (standards, ADRs, dev notes), part of promotion workflow
   - ✅ **`estimate_search_time.py`**: KEEP - Referenced in documentation (hyperparameter tuning plan, ADRs), active utility

2. **Updated documentation**:
   - Added comprehensive docstring to `ensure_venv.py` explaining its DX purpose and dual-environment context
   - Updated research findings to mark all utility scripts with proper status and explanations

**Status**: All utility scripts are active and serve specific purposes. No archiving needed.

### Tier 5: Threshold Optimization Scripts Renaming ✅

**Date**: 2026-01-26  
**Branch**: `cleanup/threshold-optimization-consolidation`

1. **Investigated threshold optimization scripts**:
   - ✅ **`optimize_thresholds.py`**: Optimizes hierarchy post-processing parameter (critical threshold reduction)
   - ✅ **`optimize_all_thresholds.py`**: Optimizes individual thresholds for all 36 categories
   - **Finding**: Both scripts serve distinct purposes - NOT duplicates

2. **Renamed scripts for clarity**:
   - `optimize_thresholds.py` → `optimize_hierarchy_threshold_reduction.py`
   - `optimize_all_thresholds.py` → `optimize_per_category_thresholds.py`
   - Updated docstrings to clarify distinct purposes and cross-reference each other

3. **Updated documentation**:
   - Updated README.md with new names and descriptions
   - Updated threshold-file-naming.md
   - Updated research findings to reflect renaming and distinct purposes

**Status**: Scripts renamed to clarify distinct purposes. Both scripts are kept as they serve different optimization needs.

---

## Next Steps

1. ✅ **Dependency analysis** - Completed (scripts are independent entry points)
2. ✅ **Test coverage check** - Completed (scripts not directly tested)
3. ✅ **Documentation audit** - Identified and fixed issues
4. ✅ **Tier 1 & 2 cleanup** - Documentation fixes and one-off script archiving completed
5. ⏳ **Git history analysis** - Find rapid generation patterns (optional)
6. ⏳ **Code review of "INVESTIGATE" scripts** - Understand their purpose (partially done)
7. ⏳ **Consolidation planning** - Plan how to merge duplicate functionality
8. ✅ **Tier 3 cleanup** - ✅ **COMPLETED** 2026-01-26 - Archived `prepare_data.py` (unused alternative interface, no bugs found)
9. ✅ **Tier 4 utility investigation** - ✅ **COMPLETED** 2026-01-26 - All utility scripts verified as active and documented

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
