from pathlib import Path

import pytest

from app.config import Config


def test_thresholds_file_matches_current_model() -> None:
    """Thresholds metadata should match the active production model."""
    model_path = Config.MODEL_PATH
    thresholds_path = model_path.with_name(f"{model_path.stem}_thresholds.json")

    if not model_path.exists() or not thresholds_path.exists():
        pytest.skip("Model or thresholds file not available in this environment")

    data = thresholds_path.read_text(encoding="utf-8")
    # Light metadata check: ensure the thresholds file references the current model
    # without requiring hash computation.
    assert model_path.name in data, (
        f"Thresholds metadata does not reference active model {model_path.name}"
    )