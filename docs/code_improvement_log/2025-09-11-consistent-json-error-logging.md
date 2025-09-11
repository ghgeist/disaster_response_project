# Code Improvement Log - 2025-09-11

## Improvement: Consistent JSON Error Logging and Type Hints

**Files Modified:** `src/disaster_classifier/utils/io.py`

**Change Type:** Error Handling & Maintainability

**Description:**
- Replaced `print` statements with structured logging in `load_json`
- Added type hints across all utility functions in `io.py`
- Added error handling for general `OSError` cases during file reads

**Impact:**
- Improves observability by routing failures through the application's logging system
- Enhances code readability and static analysis through explicit type hints
- Prevents silent failures and aids debugging when JSON loading issues occur

**Validation:**
- `pytest -q`
