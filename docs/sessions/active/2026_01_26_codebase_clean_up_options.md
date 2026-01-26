## Option 2: Duplicate consolidation — threshold optimization scripts ✅ COMPLETED

Three threshold optimization scripts:
- `scripts/optimize_hierarchy_threshold_reduction.py` (renamed from `optimize_thresholds.py`) — Hierarchy parameter optimization
- `scripts/optimize_per_category_thresholds.py` (renamed from `optimize_all_thresholds.py`) — Per-category threshold optimization
- `scripts/optimize_critical_thresholds_inc1.py` — Already archived

Action: ✅ **COMPLETED** - Renamed scripts to clarify distinct purposes. Both scripts serve different functions and are kept.

## Option 3: Duplicate consolidation — comparison scripts ✅ COMPLETED

Four comparison scripts:
- `scripts/compare_models.py` — ✅ **KEEP** - Active, documented, main comparison tool
- `scripts/compare_csv_models.py` — ✅ **ARCHIVED** 2026-01-26 - Designed for UI/UX purposes in portfolio app, not used in main workflow
- `scripts/compare_vocabulary_models.py` — ✅ **KEEP** - Specialized tool for vocabulary experiments
- `scripts/compare_child_alone.py` — ✅ **KEEP** - Specialized diagnostic tool, referenced in README

Action: ✅ **COMPLETED** - Archived `compare_csv_models.py` as unused one-off tool. Other comparison scripts serve distinct purposes and are kept.

## Option 4: Transition code cleanup ✅ COMPLETED

- `scripts/migrate_experimental_paths.py` — ✅ **ARCHIVED** 2026-01-26 - Migration utility (migration completed)
- `src/disasterproject/utils/experimental_paths.py` — ✅ **SIMPLIFIED** 2026-01-26 - Removed legacy support, now only uses new structure

Action: ✅ **COMPLETED** - Verified migration is complete (0 legacy artifacts), archived migration script, and simplified ExperimentalPathManager to remove legacy support.

## Option 5: Validation script audit (medium risk, medium effort)

Six validation scripts:
- `scripts/system_validation.py` — Active, documented
- `scripts/validate_production_model.py` — Not in main README
- `scripts/validate_multilabel_sampling.py` — In scripts/README only
- `scripts/validate_threshold_optimization_results.py` — No documentation
- `scripts/validate_ml_execution_environment.py` — No documentation
- `scripts/test_deployment_scenarios.py` — No documentation

Action: Check which are actually used vs one-off utilities.
