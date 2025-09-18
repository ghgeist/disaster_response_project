"""Shared pytest fixtures and helpers for the test suite."""
from __future__ import annotations

from pathlib import Path
from typing import Type

import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from flask import Flask

from app.app import create_app
from app.config import Config, TestConfig


def create_test_app(config_cls: Type[Config]) -> Flask:
    """Create a Flask application for the provided configuration class."""
    app = create_app(config_cls)
    # Flask uses the TESTING flag to disable error handlers; ensure it is set for test configs.
    if hasattr(config_cls, "TESTING") and getattr(config_cls, "TESTING", False):
        app.testing = True
    return app


@pytest.fixture(scope="session")
def app() -> Flask:
    """Session-wide application instance configured for fast smoke tests."""
    return create_test_app(TestConfig)


@pytest.fixture
def client(app: Flask):
    """Provide a test client bound to the shared test application."""
    with app.test_client() as client:
        yield client


def has_model(config_cls: Type[Config] = Config) -> bool:
    """Return True when the configured model artifact exists on disk."""
    model_path = getattr(config_cls, "MODEL_PATH", None)
    if not model_path:
        return False
    path = Path(model_path)
    return path.exists()


def skip_if_no_model(config_cls: Type[Config] = Config, *, reason: str | None = None) -> None:
    """Skip the calling test when the configured model artifact is missing."""
    if not has_model(config_cls):
        pytest.skip(reason or "Model artifact required for this test is not present.")
