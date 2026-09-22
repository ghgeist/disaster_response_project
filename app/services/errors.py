"""
Service-layer exception types.
"""
from __future__ import annotations


class DataServiceError(RuntimeError):
    """Raised when the data service cannot fulfill a request."""


class ModelServiceError(RuntimeError):
    """Raised when the model service encounters an unrecoverable issue."""


class DemoFeedError(RuntimeError):
    """Raised when the cached demo feed is missing, invalid, or stale."""
