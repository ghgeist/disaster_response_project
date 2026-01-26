"""
Service-layer exception types.
"""
from __future__ import annotations


class ModelDownloadSkipped(Exception):
    """Exception raised when model download should be skipped."""


class DataServiceError(RuntimeError):
    """Raised when the data service cannot fulfill a request."""


class ModelServiceError(RuntimeError):
    """Raised when the model service encounters an unrecoverable issue."""
