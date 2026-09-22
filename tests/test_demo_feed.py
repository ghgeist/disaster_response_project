"""Tests for deterministic cached demo feed builder (no pickle required)."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pandas as pd
import pytest

from app.services import demo_feed as demo_feed_module
from app.services.demo_feed import (
    assert_demo_feed_matches_production,
    build_demo_feed,
    hash_input_rows,
    load_demo_feed,
    load_message_ids,
    select_initial_message_ids,
    serialize_demo_feed,
    validate_generated_at,
)
from app.services.errors import DemoFeedError, ModelServiceError


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
                thresholds_path=Path(
                    "model/disaster_lr_v_test_prod_2026-01-01_thresholds.json"
                ),
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


def test_builder_uses_hierarchy_helper_and_display_from_fixed():
    """Predict output must pass through real hierarchy; display from fixed result."""
    service = StubModelService(
        predictions={
            "Need water": {
                "labels": {"related": 1, "water": 1, "aid_related": 0},
                "probabilities": {
                    "related": 0.9,
                    "water": 0.40,
                    "aid_related": 0.40,
                },
            }
        },
        thresholds={"related": 0.5, "water": 0.30, "aid_related": 0.5},
    )
    rows = {
        1: {"id": 1, "message": "Need water", "original": "", "genre": "direct"},
    }

    assert demo_feed_module.run_hierarchy_correction is not None
    payload = build_demo_feed(
        model_service=service,
        rows_by_id=rows,
        message_ids=[1],
        generated_at="2026-09-22T00:00:00Z",
    )
    item = payload["items"][0]
    assert service.predict_calls == ["Need water"]
    # Hierarchy activates parent; display must not use ground-truth columns.
    assert item["fixed"]["labels"]["water"] == 1
    assert item["fixed"]["labels"]["aid_related"] == 1
    assert item["raw"]["labels"]["aid_related"] == 0
    names = [entry["category"] for entry in item["classifications"]]
    assert "Water" in names
    assert "Aid Related" in names


def test_demo_feed_module_does_not_import_routes_or_simulation():
    """Architectural boundary: builder must not import Flask routes / simulation."""
    source = Path(demo_feed_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            for alias in node.names:
                imported.add(f"{node.module}.{alias.name}")

    forbidden_prefixes = (
        "app.routes",
        "app.routes.api",
    )
    forbidden_names = (
        "_improved_simulated_probabilities",
        "_row_to_feed_item",
        "_simulated_probabilities",
    )
    for module_name in imported:
        assert not any(
            module_name == prefix or module_name.startswith(prefix + ".")
            for prefix in forbidden_prefixes
        ), f"Unexpected routes import: {module_name}"
        assert not any(
            module_name.endswith(name) for name in forbidden_names
        ), f"Unexpected simulation import: {module_name}"


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

    assert payload["schema_version"] == 1
    assert payload["generated_at"] == "2026-09-22T01:02:03Z"
    assert "provenance" in payload
    provenance = payload["provenance"]
    for key in (
        "model_version",
        "model_stem",
        "model_sha256",
        "thresholds_sha256",
        "labels_sha256",
        "input_message_ids",
        "input_rows_sha256",
    ):
        assert key in provenance
    # Provenance fields must not be flattened at the top level.
    assert "model_sha256" not in payload
    assert "input_rows_sha256" not in payload

    assert provenance["input_message_ids"] == [3, 9]
    assert [item["id"] for item in payload["items"]] == ["SIG-3", "SIG-9"]
    assert [item["message_id"] for item in payload["items"]] == [3, 9]
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


def test_risk_level_ignores_sub_threshold_critical_probability():
    """Critical prob above severity cutoffs but below deployed threshold → LOW.

    Future models may set critical thresholds above 0.5; severity must still
    follow hierarchy-corrected production-positive decisions only.
    """
    service = StubModelService(
        predictions={
            "subthreshold critical": {
                "labels": {"related": 1, "water": 0, "food": 0},
                "probabilities": {
                    "related": 0.95,
                    "water": 0.80,
                    "food": 0.01,
                },
            }
        },
        # Deployed water threshold above the probability → fixed label stays 0.
        thresholds={"related": 0.5, "water": 0.90, "food": 0.5},
    )
    rows = {
        8: {
            "id": 8,
            "message": "subthreshold critical",
            "original": "",
            "genre": "direct",
        }
    }
    payload = build_demo_feed(
        model_service=service,
        rows_by_id=rows,
        message_ids=[8],
        generated_at="2026-09-22T00:00:00Z",
    )
    item = payload["items"][0]
    assert item["fixed"]["labels"]["water"] == 0
    assert item["fixed"]["probabilities"]["water"] == 0.8
    assert "Water" not in [
        entry["category"] for entry in item["classifications"]
    ]
    assert item["riskLevel"] == "LOW"


def test_hierarchy_activates_parent_when_child_clears_threshold():
    """Raw water=1 / aid_related=0 must become aid_related=1 after hierarchy."""
    service = StubModelService(
        predictions={
            "Need water urgently": {
                "labels": {
                    "related": 1,
                    "aid_related": 0,
                    "water": 1,
                    "food": 0,
                },
                "probabilities": {
                    "related": 0.9,
                    "aid_related": 0.40,
                    "water": 0.40,
                    "food": 0.01,
                },
            }
        },
        thresholds={
            "related": 0.5,
            "aid_related": 0.5,
            "water": 0.30,
            "food": 0.5,
        },
    )
    rows = {
        42: {
            "id": 42,
            "message": "Need water urgently",
            "original": "",
            "genre": "direct",
        }
    }
    payload = build_demo_feed(
        model_service=service,
        rows_by_id=rows,
        message_ids=[42],
        generated_at="2026-09-22T00:00:00Z",
    )
    item = payload["items"][0]
    assert item["message_id"] == 42
    assert item["raw"]["labels"]["water"] == 1
    assert item["raw"]["labels"]["aid_related"] == 0
    assert item["fixed"]["labels"]["water"] == 1
    assert item["fixed"]["labels"]["aid_related"] == 1
    names = [entry["category"] for entry in item["classifications"]]
    assert "Water" in names
    assert "Aid Related" in names


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
    assert (
        payload_a["provenance"]["input_rows_sha256"]
        != payload_b["provenance"]["input_rows_sha256"]
    )
    assert (
        payload_a["provenance"]["input_message_ids"]
        == payload_b["provenance"]["input_message_ids"]
    )


def test_missing_ids_file_fails(tmp_path: Path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        load_message_ids(missing)


def test_generated_at_rejects_non_iso8601():
    assert validate_generated_at("2026-09-22T00:00:00Z") == "2026-09-22T00:00:00Z"
    with pytest.raises(ValueError, match="ISO-8601"):
        validate_generated_at("banana")
    with pytest.raises(ValueError, match="ISO-8601"):
        build_demo_feed(
            model_service=StubModelService(),
            rows_by_id={1: {"id": 1, "message": "x", "original": "", "genre": "direct"}},
            message_ids=[1],
            generated_at="banana",
        )


def test_build_demo_feed_fails_closed_on_model_service_error():
    class BrokenService:
        def get_production_artifacts(self):
            raise ModelServiceError("unavailable")

        def get_thresholds_map(self):
            raise AssertionError("should not be reached")

        def predict(self, text: str):
            raise AssertionError("should not be reached")

    with pytest.raises(ModelServiceError):
        build_demo_feed(
            model_service=BrokenService(),
            rows_by_id={1: {"id": 1, "message": "x", "original": "", "genre": "direct"}},
            message_ids=[1],
            generated_at="2026-09-22T00:00:00Z",
        )


def _minimal_cached_payload(**provenance_overrides) -> dict:
    provenance = {
        "model_version": "v_test",
        "model_stem": "disaster_lr_v_test_prod_2026-01-01",
        "model_sha256": "a" * 64,
        "thresholds_sha256": "b" * 64,
        "labels_sha256": "c" * 64,
        "input_message_ids": [1],
        "input_rows_sha256": "d" * 64,
    }
    provenance.update(provenance_overrides)
    return {
        "schema_version": 1,
        "generated_at": "2026-09-22T00:00:00Z",
        "provenance": provenance,
        "items": [
            {
                "id": "SIG-1",
                "message_id": 1,
                "source": "Direct Report",
                "content": "Need water",
                "originalContent": None,
                "language": "en",
                "riskLevel": "MEDIUM",
                "categories": ["Water"],
                "classifications": [{"category": "Water", "confidence": 0.8}],
                "isTranslated": False,
                "raw": {"probabilities": {"water": 0.8}, "labels": {"water": 1}},
                "fixed": {"probabilities": {"water": 0.8}, "labels": {"water": 1}},
            }
        ],
    }


def test_load_demo_feed_accepts_valid_cache(tmp_path: Path):
    path = tmp_path / "demo_feed.json"
    path.write_text(json.dumps(_minimal_cached_payload()), encoding="utf-8")
    payload = load_demo_feed(path)
    assert payload["schema_version"] == 1
    assert len(payload["items"]) == 1
    assert payload["provenance"]["model_sha256"] == "a" * 64


def test_load_demo_feed_rejects_bad_schema_and_empty_items(tmp_path: Path):
    bad_schema = tmp_path / "bad_schema.json"
    bad_schema.write_text(
        json.dumps({**_minimal_cached_payload(), "schema_version": 99}),
        encoding="utf-8",
    )
    with pytest.raises(DemoFeedError, match="schema_version"):
        load_demo_feed(bad_schema)

    empty_items = tmp_path / "empty_items.json"
    empty_payload = _minimal_cached_payload()
    empty_payload["items"] = []
    empty_items.write_text(json.dumps(empty_payload), encoding="utf-8")
    with pytest.raises(DemoFeedError, match="non-empty items"):
        load_demo_feed(empty_items)

    missing = tmp_path / "missing.json"
    with pytest.raises(DemoFeedError, match="not found"):
        load_demo_feed(missing)


def test_assert_demo_feed_matches_production_ok_and_mismatch():
    artifacts = StubModelService()._artifacts
    payload = _minimal_cached_payload()
    assert_demo_feed_matches_production(payload, artifacts)

    mismatched = _minimal_cached_payload(model_sha256="f" * 64)
    with pytest.raises(DemoFeedError, match="model_sha256"):
        assert_demo_feed_matches_production(mismatched, artifacts)

    missing_field = _minimal_cached_payload()
    del missing_field["provenance"]["labels_sha256"]
    with pytest.raises(DemoFeedError, match="labels_sha256"):
        assert_demo_feed_matches_production(missing_field, artifacts)
