"""Tests for deterministic cached demo feed builder (no pickle required)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services.demo_feed import (
    build_demo_feed,
    hash_input_rows,
    load_message_ids,
    select_initial_message_ids,
    serialize_demo_feed,
)
from app.services.errors import ModelServiceError


@dataclass(frozen=True)
class _FakePaths:
    model_path: Path
    thresholds_path: Path
    labels_path: Path
    model_info_path: Path


@dataclass(frozen=True)
class _FakeArtifacts:
    paths: _FakePaths
    thresholds: Dict[str, float]
    label_order: List[str]
    model_sha256: str
    thresholds_sha256: str
    labels_sha256: str
    model_info: Mapping[str, Any]


class StubModelService:
    """Minimal ModelService stub for demo-feed builder tests."""

    def __init__(
        self,
        *,
        predictions: Dict[str, Dict[str, Any]] | None = None,
        thresholds: Dict[str, float] | None = None,
        artifacts: _FakeArtifacts | None = None,
    ):
        self._predictions = predictions or {}
        self._thresholds = thresholds or {"water": 0.1, "food": 0.5, "related": 0.5}
        self._artifacts = artifacts or _FakeArtifacts(
            paths=_FakePaths(
                model_path=Path("model/disaster_lr_v_test_prod_2026-01-01.pkl"),
                thresholds_path=Path("model/disaster_lr_v_test_prod_2026-01-01_thresholds.json"),
                labels_path=Path("model/disaster_lr_v_test_prod_2026-01-01_labels.json"),
                model_info_path=Path("model/MODEL_INFO.json"),
            ),
            thresholds=dict(self._thresholds),
            label_order=list(self._thresholds.keys()),
            model_sha256="a" * 64,
            thresholds_sha256="b" * 64,
            labels_sha256="c" * 64,
            model_info={"version": "v_test"},
        )
        self.predict_calls: list[str] = []

    def get_production_artifacts(self) -> _FakeArtifacts:
        return self._artifacts

    def get_thresholds_map(self) -> Dict[str, float]:
        return dict(self._thresholds)

    def predict(self, text: str) -> dict:
        self.predict_calls.append(text)
        if text in self._predictions:
            return self._predictions[text]
        # Default: low-confidence water positive after hierarchy (threshold 0.1)
        return {
            "labels": {"related": 1, "water": 1, "food": 0},
            "probabilities": {"related": 0.9, "water": 0.12, "food": 0.05},
        }


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [10, 2, 5, 7],
            "message": ["alpha", "beta", "", "gamma"],
            "original": ["alpha", "beta-orig", "", "gamma"],
            "genre": ["direct", "news", "direct", "social"],
            "water": [1, 0, 1, 1],
            "food": [0, 1, 0, 0],
            "related": [1, 1, 1, 1],
        }
    )


def test_select_initial_message_ids_deterministic_and_ignores_categories():
    df = _sample_frame()
    selected = select_initial_message_ids(df, n=2)
    assert selected == [2, 7]

    # Category columns must not affect selection
    flipped = df.copy()
    flipped["water"] = 1 - flipped["water"]
    flipped["food"] = 1 - flipped["food"]
    assert select_initial_message_ids(flipped, n=2) == selected


def test_build_demo_feed_never_calls_simulation():
    service = StubModelService(
        predictions={
            "Need water": {
                "labels": {"related": 1, "water": 1},
                "probabilities": {"related": 0.9, "water": 0.8},
            }
        },
        thresholds={"related": 0.5, "water": 0.5},
    )
    rows = {
        1: {"id": 1, "message": "Need water", "original": "", "genre": "direct"},
    }

    with (
        patch("app.routes.api._improved_simulated_probabilities") as sim,
        patch("app.routes.api._row_to_feed_item") as row_to_item,
    ):
        payload = build_demo_feed(
            model_service=service,
            rows_by_id=rows,
            message_ids=[1],
            generated_at="2026-09-22T00:00:00Z",
        )
        sim.assert_not_called()
        row_to_item.assert_not_called()

    assert payload["schema_version"] == 1
    assert payload["items"][0]["classifications"]


def test_schema_provenance_and_item_order():
    service = StubModelService(
        predictions={
            "first": {
                "labels": {"related": 1, "water": 1},
                "probabilities": {"related": 0.9, "water": 0.7},
            },
            "second": {
                "labels": {"related": 1, "food": 1},
                "probabilities": {"related": 0.9, "food": 0.6},
            },
        },
        thresholds={"related": 0.5, "water": 0.5, "food": 0.5},
    )
    rows = {
        9: {"id": 9, "message": "second", "original": "", "genre": "news"},
        3: {"id": 3, "message": "first", "original": "primero", "genre": "direct"},
    }
    payload = build_demo_feed(
        model_service=service,
        rows_by_id=rows,
        message_ids=[3, 9],
        generated_at="2026-09-22T01:02:03Z",
    )

    for key in (
        "schema_version",
        "generated_at",
        "model_version",
        "model_stem",
        "model_sha256",
        "thresholds_sha256",
        "labels_sha256",
        "input_message_ids",
        "input_rows_sha256",
        "items",
    ):
        assert key in payload

    assert payload["input_message_ids"] == [3, 9]
    assert [item["id"] for item in payload["items"]] == ["SIG-3", "SIG-9"]
    assert payload["items"][0]["content"] == "first"
    assert payload["items"][0]["isTranslated"] is True
    assert "timestamp" not in payload["items"][0]
    assert "raw" in payload["items"][0] and "fixed" in payload["items"][0]


def test_display_from_fixed_labels_not_ground_truth_columns():
    # Ground-truth column would say food=1, but fixed labels mark only water.
    service = StubModelService(
        predictions={
            "msg": {
                "labels": {"related": 1, "water": 1, "food": 0},
                "probabilities": {"related": 0.95, "water": 0.8, "food": 0.01},
            }
        },
        thresholds={"related": 0.5, "water": 0.5, "food": 0.5},
    )
    rows = {
        1: {
            "id": 1,
            "message": "msg",
            "original": "",
            "genre": "direct",
            "water": 0,
            "food": 1,
        }
    }
    payload = build_demo_feed(
        model_service=service,
        rows_by_id=rows,
        message_ids=[1],
        generated_at="2026-09-22T00:00:00Z",
    )
    names = [entry["category"] for entry in payload["items"][0]["classifications"]]
    assert "Water" in names
    assert "Food" not in names
    assert payload["items"][0]["categories"] == ["Water"]


def test_serialize_identical_bytes_for_same_stub_and_generated_at():
    service = StubModelService()
    rows = {1: {"id": 1, "message": "Need water now", "original": "", "genre": "direct"}}
    kwargs = dict(
        model_service=service,
        rows_by_id=rows,
        message_ids=[1],
        generated_at="2026-09-22T12:00:00Z",
    )
    first = serialize_demo_feed(build_demo_feed(**kwargs))
    second = serialize_demo_feed(build_demo_feed(**kwargs))
    assert first == second
    assert first.startswith(b"{")


def test_low_confidence_hierarchy_positive_preserved():
    service = StubModelService(
        predictions={
            "low": {
                "labels": {"related": 1, "water": 1},
                "probabilities": {"related": 0.9, "water": 0.12},
            }
        },
        thresholds={"related": 0.5, "water": 0.10},
    )
    rows = {4: {"id": 4, "message": "low", "original": "", "genre": "direct"}}
    payload = build_demo_feed(
        model_service=service,
        rows_by_id=rows,
        message_ids=[4],
        generated_at="2026-09-22T00:00:00Z",
    )
    classifications = payload["items"][0]["classifications"]
    water = next(item for item in classifications if item["category"] == "Water")
    assert water["confidence"] == 0.12


def test_changing_message_text_changes_input_rows_sha256():
    rows_a = [{"id": 1, "message": "hello", "original": "", "genre": "direct"}]
    rows_b = [{"id": 1, "message": "hello!", "original": "", "genre": "direct"}]
    assert hash_input_rows(rows_a) != hash_input_rows(rows_b)

    service = StubModelService()
    base_rows = {1: {"id": 1, "message": "hello", "original": "", "genre": "direct"}}
    changed_rows = {1: {"id": 1, "message": "hello!", "original": "", "genre": "direct"}}
    payload_a = build_demo_feed(
        model_service=service,
        rows_by_id=base_rows,
        message_ids=[1],
        generated_at="2026-09-22T00:00:00Z",
    )
    payload_b = build_demo_feed(
        model_service=service,
        rows_by_id=changed_rows,
        message_ids=[1],
        generated_at="2026-09-22T00:00:00Z",
    )
    assert payload_a["input_rows_sha256"] != payload_b["input_rows_sha256"]
    assert payload_a["input_message_ids"] == payload_b["input_message_ids"]


def test_missing_ids_file_fails(tmp_path: Path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        load_message_ids(missing)


def test_build_demo_feed_fails_closed_on_model_service_error():
    service = MagicMock()
    service.get_production_artifacts.side_effect = ModelServiceError("unavailable")
    with pytest.raises(ModelServiceError):
        build_demo_feed(
            model_service=service,
            rows_by_id={1: {"id": 1, "message": "x", "original": "", "genre": "direct"}},
            message_ids=[1],
            generated_at="2026-09-22T00:00:00Z",
        )
