# Code Improvement Log - 2025-09-15

## Improvement: Defensive NLTK Handling and Idempotent Logging Setup

**Files Modified:**
- `src/disasterproject/utils/config.py`
- `src/disasterproject/utils/json_io.py`

**Change Type:** Resilience, Logging Quality, Maintainability

**Description:**
- Made `setup_logging()` idempotent to prevent duplicate handlers when called multiple times from scripts and the app.
- Replaced import-time logging with a quiet, deterministic random seed initialization.
- Loaded `STOPWORDS_SET` defensively: falls back to an empty set with a warning if NLTK resources are unavailable, avoiding crashes before `app/nltk_setup.py` runs.
- Standardized logging style: replaced f-string logging in `json_io.load_model_parameters()` with lazy, structured logging via module logger.

**Impact:**
- Prevents log duplication and noisy output across repeated runs.
- Improves startup robustness in fresh environments lacking NLTK data.
- Aligns logging with established patterns from prior improvements (structured, lazy formatting).
- Reduces unexpected side effects at import time.

**Validation:**
- Import `disasterproject.utils.config` without NLTK data present no longer raises; warning is logged and code proceeds.
- Running scripts that call `setup_logging()` multiple times does not duplicate log lines.
- Existing APIs remain unchanged; no functional regressions expected.

**Related Work:**
- Builds on earlier logging and error-handling improvements (2025-09-11 logs).

