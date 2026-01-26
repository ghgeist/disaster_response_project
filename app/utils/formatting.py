"""
Formatting utilities for request context and logging.
"""
from flask import g, has_request_context, request


def format_request_context() -> str:
    """Return a formatted suffix containing request metadata for logging."""
    if not has_request_context():
        return ""

    request_id = getattr(g, "request_id", None) or request.headers.get("X-Request-ID") or "-"
    return " [request_id=%s method=%s path=%s]" % (request_id, request.method, request.path)
