"""Performance tests for model warm and reload timings."""
from __future__ import annotations

import logging
from time import perf_counter

import pytest

from app.config import Config
from app.services import ModelService
from tests.conftest import skip_if_no_model

pytestmark = pytest.mark.perf

PERF_THRESHOLD_MS = 150


def test_model_reload_under_threshold(caplog: pytest.LogCaptureFixture) -> None:
    """Reloading an already cached model should remain within the SLA."""
    skip_if_no_model(
        Config,
        reason="Model artifact required for performance tests is not present.",
    )

    service = ModelService(Config.MODEL_PATH, Config.GDRIVE_MODEL_ID)
    service.load_model()

    with caplog.at_level(logging.INFO):
        start = perf_counter()
        service.load_model()
        duration_ms = (perf_counter() - start) * 1000
        logging.getLogger("tests.perf").info("Model reload completed in %.2fms", duration_ms)

    assert (
        duration_ms < PERF_THRESHOLD_MS
    ), f"Model reload took {duration_ms:.2f}ms, exceeding {PERF_THRESHOLD_MS}ms threshold"
