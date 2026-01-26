## Option 2: Duplicate consolidation — threshold optimization scripts (medium risk, medium effort)

Three threshold optimization scripts:
- `scripts/optimize_thresholds.py` — Referenced in README architecture section
- `scripts/optimize_all_thresholds.py` — No documentation
- `scripts/optimize_critical_thresholds_inc1.py` — Already archived

Action: Compare `optimize_thresholds.py` vs `optimize_all_thresholds.py` to see if one can be archived.

## Option 3: Duplicate consolidation — comparison scripts (low risk, medium effort)

Four comparison scripts:
- `scripts/compare_models.py` — Active, documented
- `scripts/compare_csv_models.py` — Only in scripts/README.md
- `scripts/compare_vocabulary_models.py` — Purpose known (vocabulary-optimized models)
- `scripts/compare_child_alone.py` — Specialized tool

Action: Verify if `compare_csv_models.py` duplicates `compare_models.py` or serves a different purpose.

## Option 4: Transition code cleanup (low risk, low effort)

- `scripts/migrate_experimental_paths.py` — Migration utility
- `src/disasterproject/utils/experimental_paths.py` — Transition infrastructure

Action: Check if experimental path migration is complete; if so, archive the migration script and simplify the path manager.

## Option 5: Validation script audit (medium risk, medium effort)

Six validation scripts:
- `scripts/system_validation.py` — Active, documented
- `scripts/validate_production_model.py` — Not in main README
- `scripts/validate_multilabel_sampling.py` — In scripts/README only
- `scripts/validate_threshold_optimization_results.py` — No documentation
- `scripts/validate_ml_execution_environment.py` — No documentation
- `scripts/test_deployment_scenarios.py` — No documentation

Action: Check which are actually used vs one-off utilities.
