"""Tests for request logging helpers."""

from flask import Flask, g

from app.utils import format_request_context


def test_format_request_context_without_request() -> None:
    """When no request context is active an empty suffix is returned."""
    assert format_request_context() == ""


def test_format_request_context_with_header_request_id() -> None:
    """The helper includes request metadata when headers provide an ID."""
    app = Flask(__name__)
    with app.test_request_context("/health", method="GET", headers={"X-Request-ID": "abc-123"}):
        assert format_request_context() == " [request_id=abc-123 method=GET path=/health]"


def test_format_request_context_prefers_g_request_id() -> None:
    """The Flask ``g`` request_id overrides header values when available."""
    app = Flask(__name__)
    with app.test_request_context("/go", method="POST", headers={"X-Request-ID": "abc-123"}):
        g.request_id = "req-999"
        assert format_request_context() == " [request_id=req-999 method=POST path=/go]"
