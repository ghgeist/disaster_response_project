# Code Improvement Log - 2025-09-15

## Improvement: Standardize Parameterized Logging (Lazy Formatting)

**Files Modified:**
- `app/services.py`
- `app/routes.py`
- `app/utils.py`

**Change Type:** Logging Consistency & Performance

**Description:**
Replaced f-string and concatenated log messages with parameterized logging across services, routes, and utils. This aligns with best practices and earlier logging improvements, ensuring lazy evaluation and consistent structure.

**Specific Changes:**
- Converted `logger.info/warning/error/exception` calls to use `%s` placeholders with arguments.
- Preserved existing log levels and messages; no behavioral changes.
- Standardized exception logs to avoid unnecessary string interpolation.

**Impact:**
- Avoids constructing log strings when a log level is disabled (small perf wins).
- Produces consistent, parseable logs across modules.
- Complements previous improvements to logging configuration and error handling.

**Validation:**
- Imported and exercised affected modules without runtime errors.
- Verified representative code paths render logs correctly with placeholders.

**Related Work:**
- Builds on 2025-09-11 improvements around error handling and import organization.
- Complements 2025-09-15 defensive NLTK/idempotent logging setup.

