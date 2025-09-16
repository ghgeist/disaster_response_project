---
title: "Planning Agent: Hyperparameter Tuning for Production Model"
date: "2025-09-16"
status: "active"
tags: ["hyperparameter-tuning", "optimization", "planning", "RandomForest"]
author: "Gemini"
related: ["docs/research/2025-09-16-analyzing-and-improving-the-classifier.md"]
---


# Planning Agent: Hyperparameter Tuning for Production Model

**Date**: 2025-09-16  
**Status**: Active  
**Priority**: High  
**Estimated Duration**: 1-2 hours (with Randomized Search)
**Tags**: `hyperparameter-tuning`, `optimization`, `planning`, `RandomForest`

## Progress Log
- **2025-09-16**: Initial plan created to build a new script for hyperparameter tuning.
- **2025-09-16**: **Revised plan** to leverage the existing `scripts/02_test_hyperparameters.py`.
- **2025-09-16**: Created a new configuration file with a descriptive name (`2025-09-16_comprehensive-grid_active.json`) and renamed the legacy config to preserve it.
- **2025-09-16**: **Improved script** by adding a `--config` argument to allow for flexible selection of parameter grids.
- **2025-09-16**: **Improved script** by changing the scoring metric from `accuracy` to `f1_weighted` for more meaningful optimization.
- **2025-09-16**: **Improved script** by switching from `GridSearchCV` to the faster `RandomizedSearchCV` to reduce execution time.
- **2025-09-16**: **Fixed** an `IndentationError` in the script caused by previous modifications.
- **2025-09-16**: **Refactored codebase** to improve separation of concerns. Moved duplicated functions for parameter searching and config loading from scripts into the `src/disasterproject` package. Created a dedicated, non-interactive `estimate_search_time.py` script.
- **2025-09-16**: **Performance review identified critical issues** requiring immediate attention before execution: duplicate `run_grid_search()` function in hyperparameter script creates maintenance burden and potential inconsistencies.
- **2025-09-16**: **Output path analysis revealed** hyperparameter script outputs to `model/` directory instead of proper `experiments/` structure, inconsistent with project organization.
- **2025-09-16**: **Comprehensive script analysis identified CRITICAL reliability and performance issues**: inappropriate CV strategy for multi-label data, no resume capability, unbounded resource usage, missing error recovery, and runtime dependency risks that could cause search failure.
- **2025-09-16**: **Selected CV Strategy**: Chose to use `MultilabelStratifiedKFold` from the `iterative-stratification` library to ensure a robust and reliable multi-label CV strategy.
- **2025-09-16**: **Discovered Critical Split Inconsistency**: Found that the hyperparameter script uses a naive, non-reproducible train/test split, inconsistent with the production script's robust "frozen test set" logic.
- **2025-09-16**: **Revised Increment 2 Strategy**: Replaced the initial plan for recovery and monitoring with a more robust **Custom Search Loop** implementation to provide superior resilience, error handling, and progress tracking.

## 🎯 Objective

To systematically search for and identify a more optimal set of hyperparameters for the existing `RandomForestClassifier` pipeline by leveraging and improving the project's existing tuning scripts and shared code.

## 📋 Success Criteria

- [x] A new, descriptively named hyperparameter grid (`2025-09-16_comprehensive-grid_active.json`) is created.
- [x] Shared utility functions are refactored into the `src/disasterproject` package.
- [ ] **CRITICAL**: Remove duplicate `run_grid_search()` function and fix code quality issues.
- [ ] **CRITICAL**: Fix output paths to use proper `experiments/` directory structure.
- [ ] **CRITICAL**: Implement proper cross-validation strategy for multi-label classification.
- [ ] **CRITICAL**: Implement careful resource management to prevent system crashes during long search.
- [ ] The improved `scripts/02_test_hyperparameters.py` script is executed successfully.
- [ ] The script generates an `optimized_parameters.json` file with the best-performing parameter set.
- [ ] The contents of `optimized_parameters.json` are copied to `model/parameters.json`.
- [ ] A new production model, trained with the optimized parameters, shows a measurable improvement in F1-score or recall.

## 🔍 Context

The current production model uses minimal hyperparameters. The project's scripts contained duplicated code, which we have now refactored into the main `src/disasterproject` package for better code quality and reuse. However, **comprehensive analysis identified critical reliability and performance issues that must be resolved before execution**:

### **🔥 CRITICAL BLOCKERS**
1. **Code Quality**: Duplicate `run_grid_search()` function in `scripts/02_test_hyperparameters.py` (lines 465-517) conflicts with centralized package function
2. **Output Structure**: Script outputs to `model/` directory instead of proper `experiments/` structure, breaking project organization
3. **Cross-Validation Strategy**: Uses default 5-fold CV inappropriate for imbalanced multi-label classification. **Decision**: Will implement `MultilabelStratifiedKFold` from the `iterative-stratification` library.
4. **Resource Management**: No memory validation, timeout protection, or CPU usage limits - could crash or overwhelm system
5. **Inconsistent Data Splitting**: Hyperparameter script uses a simple random split with no fixed seed, while the production script uses a robust frozen test set. This makes results unreliable and not comparable.

### **🔥 HIGH-RISK RELIABILITY ISSUES**
6. **No Resume Capability**: Multi-hour search that crashes loses all progress and must restart completely
7. **Missing Error Recovery**: Minimal handling of parameter validation failures or runtime errors
8. **Runtime Dependencies**: NLTK downloads during execution could timeout or fail, breaking the search
9. **No Progress Monitoring**: Long-running process provides minimal feedback on completion status

## 🛠️ Conservative Sequential Implementation Strategy

**APPROACH**: One change at a time with validation gates. No parallelization until we're confident each step works.

**TESTING STRATEGY**: Portfolio project focus - prevent 4-hour time loss, not enterprise-grade validation.
- **SMOKE TESTS**: Quick validation that basic functionality works
- **FOCUSED TESTS**: Crash-prevention for high-risk components
- **MINIMAL TESTS**: Just verify no obvious errors
- **SKIP**: Performance optimization, edge cases, comprehensive validation

### **Phase 1: Complete Validation First** ⚠️ *CRITICAL FOUNDATION*
1. **Validate Environment**:
   - Check `data/02_stg/stg_disaster_response.db` exists and can be queried
   - Verify critical packages import (sklearn, pandas, nltk, iterative-stratification)
2. **Test Parameter Combinations**: Validate hyperparameter grid for invalid combinations
3. **Verify Data Loading**: Ensure dataset loads without errors and fits in memory
4. **Check Resource Limits**: Verify system can handle expected CPU/memory usage
5. **Confirm Baseline**: Test current script works before making any changes

**🚪 GATE 1**: Don't proceed until environment is ready and baseline script confirmed working

### **Phase 2: Sequential Fixes with Individual Testing** ⚠️ *CRITICAL BLOCKERS*

**Fix 1**: Add iterative-stratification dependency
- Add `iterative-stratification` to `requirements.txt`
- **SMOKE TEST**: `python -c "from iterstrat.ml_stratifiers import MultilabelStratifiedKFold; print('Import OK')"`

**Fix 2**: Remove duplicate function
- Delete `run_grid_search()` function from `scripts/02_test_hyperparameters.py` (lines 465-517)
- Add import: `from disasterproject.models.pipeline import run_parameter_search`
- **SMOKE TEST**: `python scripts/02_test_hyperparameters.py --help` (should show help without errors)

**Fix 3**: Fix cross-validation strategy
- Add import: `from iterstrat.ml_stratifiers import MultilabelStratifiedKFold`
- Implement `MultilabelStratifiedKFold` in place of default cross-validation
- **FOCUSED TEST**: Load 100 sample subset, test 3-fold CV, verify reasonable class distribution vs old CV

**Fix 4**: Standardize data splitting
- Investigate how `scripts/04_create_production_model.py` handles train/test splitting
- Replace naive `train_test_split` with the production script's robust data splitting logic
- **MINIMAL TEST**: Verify both frozen/random modes run without errors

**Fix 5**: Fix output paths
- Update script to output to `experiments/` structure:
  - Results → `experiments/results/2025-09-16_hyperparameter_search_results.json`
  - Parameters → `experiments/model_candidates/2025-09-16_optimized_parameters.json`
  - Models → `experiments/models/2025-09-16_hyperparameter_tuned_model.pkl`
  - Logs → `experiments/logs/2025-09-16_hyperparameter_search.log`
  - Manifest → `experiments/2025-09-16_hyperparameter_search_manifest.json`
- **Current problematic paths**: `scripts/*.json` outputs (move to experiments structure)
- **Automation**: Add function to automatically generate manifest JSON at script completion
- **SMOKE TEST**: Run with minimal params to test directory creation

**Fix 6**: Add resource management
- Add import: `import psutil`
- **Memory monitoring**: Use `psutil` to check RAM usage every 10 iterations, warn at 10GB, abort at 12GB
- **CPU**: Set `n_jobs=2` instead of `-1` to keep system responsive
- **Time**: Overall timeout 4-6 hours, per-trial timeout 30 minutes
- **Implementation**: Add memory check function with `psutil.Process().memory_info().rss`
- **FOCUSED TEST**: Test with artificial limits to ensure monitoring works (crash prevention)

**🚪 GATE 2**: All fixes implemented and individually tested before proceeding

### **Phase 3: Execution & Deployment** ⚠️ *FOCUS ON RESOURCE MANAGEMENT*
**Strategy**: Use RandomizedSearchCV as-is with careful resource management instead of custom loop complexity.

7. **Estimate Runtime**: Use `scripts/estimate_search_time.py` to set realistic expectations
8. **Configure Resource Limits**:
   - Memory cap: ~12GB (75% of 16GB system)
   - CPU: `n_jobs=2` for system responsiveness
   - Search size: `n_iter=100` (reasonable vs 500+)
   - Timeouts: 4-6 hours overall, 30 min per trial
9. **Execute Search**: Run improved `scripts/02_test_hyperparameters.py` with monitoring
10. **Monitor Progress**:
    - **Console**: Print progress every 10 iterations (completion %, current best score, RAM usage, ETA)
    - **Log file**: Write detailed progress to `experiments/logs/` with timestamps and system stats
    - **Git ignore**: Add `experiments/logs/` to `.gitignore`
11. **Validate Results**: Verify search completed successfully with valid outputs
12. **Create Experiment Manifest**: Add function to hyperparameter script to automatically generate manifest JSON at completion with all output file paths
13. **Promote Parameters**: Copy optimized parameters to `model/parameters.json`
14. **Train Model**: Execute `scripts/04_create_production_model.py`
15. **Validate Improvement**: Compare new vs old `performance_metrics.csv`

## 📊 Acceptance Criteria

The task is complete when:
- A new production model (`.pkl`) is generated using hyperparameters found by the improved script.
- The `model/parameters.json` file is updated with the new parameters.
- The performance improvement is confirmed by comparing the old and new evaluation files.

## 🔗 Related Work

- This plan is a direct follow-up to the analysis in [docs/research/2025-09-16-analyzing-and-improving-the-classifier.md](docs/research/2025-09-16-analyzing-and-improving-the-classifier.md).

## 📈 Metrics

- **Primary Metric**: Improvement in the weighted average F1-score.
- **Secondary Metric**: Improvement in recall for critical, rare categories.
- **Constraint Metric**: Randomized search execution time.

## 🚨 Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Code duplication causes maintenance issues** | **High** | **Confirmed** | **CRITICAL**: Remove duplicate `run_grid_search()` function before execution. Use centralized package functions only. |
| **Wrong output paths break project organization** | **High** | **Confirmed** | **CRITICAL**: Update script to output to proper `experiments/` directory structure instead of `model/`. |
| **Inappropriate CV strategy for multi-label data** | **High** | **Confirmed** | **CRITICAL**: Replace default 5-fold CV with `MultilabelStratifiedKFold` from the `iterative-stratification` library. This provides a robust, purpose-built solution for this exact problem. |
| **No resource management causes system crashes** | **High** | **High** | **CRITICAL**: Add memory limits, CPU bounds, and timeout protection before execution. |
| **Search crashes lose all progress (no resume)** | **High** | **Medium** | **HIGH**: Implement a custom search loop with a results log. On restart, the script reads the log and resumes from the last completed trial, ensuring zero progress is lost. |
| **Runtime dependency failures break execution** | **High** | **Medium** | **HIGH**: The custom search loop will wrap each trial in a try/except block, allowing the search to continue even if one parameter set causes a runtime error. |
| **No progress monitoring in long-running process** | **Medium** | **High** | **HIGH**: The custom loop will provide structured, real-time logging for each trial, showing parameters, score, and current best score. |
| Refactoring breaks existing functionality | Medium | Low | Test with `estimate_search_time.py` after each change. Use existing working import patterns. |

## 📄 Deliverables

- [x] A new configuration file: `experiments/experimental_configs/hyperparameters/2025-09-16_comprehensive-grid_active.json`.
- [x] Refactored, centralized utility functions in the `src/disasterproject` package.
- [ ] **Pre-execution validation suite** ensuring environment readiness and parameter validity.
- [ ] **Hardened hyperparameter script** with proper CV strategy, resource management, and error recovery.
- [ ] A robust **Custom Search Loop** implementation replacing the default `fit` method.
- [ ] A `search_results.csv` log file for tracking progress and enabling resume capability.
- [ ] **Clean script architecture** with no duplicate functions and proper package imports.
- [ ] **Validated functionality** confirmed through comprehensive testing before execution.
- [ ] **Organized experiment outputs** in proper `experiments/` directory structure.
- [ ] An updated `model/parameters.json` file.
- [ ] A new production model file (`.pkl`) and its corresponding `performance_metrics.csv`.
- [ ] A brief results summary comparing the new and old model performance.

## 🎯 Immediate Next Steps

**CONSERVATIVE APPROACH**: Based on past experience creating messes, we're taking a systematic, one-change-at-a-time approach with validation gates.

### **START HERE: Phase 1 - Complete Validation**
1. **Validate environment** - Confirm database, memory, dependencies ready
2. **Test current script baseline** - Ensure it runs before we change anything
3. **Validate parameter grid** - Check for invalid combinations
4. **🚪 GATE 1**: Don't proceed until baseline confirmed working

### **Phase 2: Sequential Fixes (One at a Time)**
1. **Fix 1**: Add dependency → Test import works
2. **Fix 2**: Remove duplicate function → Test script loads
3. **Fix 3**: Fix cross-validation → Test CV works in isolation
4. **Fix 4**: Fix data splitting → Test data loading works
5. **Fix 5**: Fix output paths → Test directory creation works
6. **Fix 6**: Add resource limits → Test limits work
7. **🚪 GATE 2**: All fixes tested individually before proceeding

### **Phase 3: Custom Search Loop (Single Focused Effort)**
8. **Implement custom search loop** as one cohesive change
9. **🧪 COMPREHENSIVE TESTING**: Test thoroughly with small parameter set
10. **🚪 GATE 3**: Prove reliability before full execution

### **Phase 4: Execute Only When Everything Proven**
11. **Execute full hyperparameter search** with monitoring
12. **Deploy optimized model** and validate improvement

**This transforms the approach from "change everything and hope" to "change one thing, prove it works, move to next" - essential for avoiding the mess that happened last time.**

## 🔄 Confirmation Required

**This comprehensive plan addresses all identified reliability and performance issues. The hyperparameter search is currently unsuitable for production execution without these critical fixes.**

**Please review this updated plan and confirm if you'd like me to proceed with the systematic reliability improvements before attempting hyperparameter tuning execution.**
