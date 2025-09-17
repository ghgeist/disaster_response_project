---
title: "Fix DEFAULT_N_JOBS Constant Redefinition Issue"
date: "2025-09-17"
status: "accepted"
tags: ["configuration", "performance", "constants"]
author: "Claude Code"
related: []
---


# Fix DEFAULT_N_JOBS Constant Redefinition Issue

**Date**: 2025-09-17
**Status**: Accepted
**Deciders**: Claude Code, Grant (User)
**Tags**: configuration, performance, constants

## Context

The `src/disasterproject/utils/config.py` file contained a critical issue where the `DEFAULT_N_JOBS` constant was defined twice:

1. Line 31: `DEFAULT_N_JOBS = 2` (intended for hyperparameter search parallelism)
2. Line 44: `DEFAULT_N_JOBS = 1` (intended for RandomForest estimators, but overwrote the previous value)

This caused hyperparameter search operations to silently use only 1 CPU core instead of the intended 2 cores, potentially dramatically slowing down experimentation and model optimization workflows.

The issue was identified through code review highlighting that importing modules would receive the final assigned value (1 core) rather than the intended values for their specific use cases.

## Decision

Replace the single overloaded `DEFAULT_N_JOBS` constant with two distinct, purpose-specific constants:

1. `SEARCH_N_JOBS = 2` - For hyperparameter search operations that benefit from parallelism
2. `RF_N_JOBS = 1` - Conservative default for RandomForest estimators to prevent CPU oversubscription

Update all importing modules to use the appropriate constant for their specific use case:
- Hyperparameter search modules use `SEARCH_N_JOBS`
- RandomForest estimator modules use `RF_N_JOBS`

## Consequences

### Positive
- Hyperparameter search now properly utilizes 2 CPU cores, improving experimentation speed
- Clear separation of concerns with purpose-specific constants
- Eliminates silent performance degradation
- Better code maintainability with explicit naming
- Prevents future confusion about intended parallelism levels

### Negative
- Requires updating import statements in affected modules
- Slightly increases the number of configuration constants
- Breaking change for any external code referencing `DEFAULT_N_JOBS`

### Neutral
- No change in actual runtime behavior for RandomForest estimators (still uses 1 core)
- Maintains conservative CPU usage patterns for individual estimators

## Alternatives Considered

1. **Keep single constant with conditional logic**: Would require passing context about the use case, adding complexity
2. **Use environment variables**: Would make configuration less explicit and harder to track
3. **Remove parallelism entirely**: Would slow down all operations unnecessarily
4. **Use higher default for all operations**: Could cause CPU oversubscription and system responsiveness issues

The chosen approach provides the best balance of performance optimization and system stability.

## References

- Code review comment identifying the issue in `src/disasterproject/utils/config.py:40-44`
- Updated files:
  - `src/disasterproject/utils/config.py`
  - `src/disasterproject/models/hyperparameter_search.py`
  - `scripts/02_test_hyperparameters.py`
  - `scripts/estimate_search_time.py`