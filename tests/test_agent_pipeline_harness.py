"""Tests for scripts/agent_pipeline harness helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "scripts" / "agent_pipeline"


def _load(name: str) -> ModuleType:
    module_path = PIPELINE_DIR / f"{name}.py"
    # Ensure sibling imports (common) resolve when loading by file path.
    if str(PIPELINE_DIR) not in sys.path:
        sys.path.insert(0, str(PIPELINE_DIR))
    spec = importlib.util.spec_from_file_location(f"agent_pipeline_{name}", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def common_mod() -> ModuleType:
    return _load("common")


@pytest.fixture(scope="module")
def parse_mod() -> ModuleType:
    return _load("parse_failures")


@pytest.fixture(scope="module")
def shard_mod() -> ModuleType:
    return _load("shard_tests")


def test_sanitize_task_id_accepts_safe_ids(common_mod: ModuleType) -> None:
    assert common_mod.sanitize_task_id("ruff-foo-abc123") == "ruff-foo-abc123"


def test_sanitize_task_id_rejects_unsafe_ids(common_mod: ModuleType) -> None:
    with pytest.raises(ValueError):
        common_mod.sanitize_task_id("../etc/passwd")
    with pytest.raises(ValueError):
        common_mod.sanitize_task_id("has spaces")


def test_parse_ruff_json_null_location_does_not_crash(parse_mod: ModuleType) -> None:
    payload = json.dumps(
        [
            {
                "filename": "src/disasterproject/utils/foo.py",
                "code": "F401",
                "message": "unused import",
                "location": None,
            }
        ]
    )
    tasks = parse_mod.parse_ruff_json(payload)
    assert len(tasks) == 1
    assert tasks[0]["line"] is None
    assert tasks[0]["message"] == "F401: unused import"


def test_parse_ruff_json_preserves_message_punctuation(parse_mod: ModuleType) -> None:
    payload = json.dumps(
        [
            {
                "filename": "src/disasterproject/utils/foo.py",
                "code": "E501",
                "message": "line too long:",
                "location": {"row": 12, "column": 80},
            },
            {
                "filename": "src/disasterproject/utils/bar.py",
                "code": "F401",
                "message": "",
                "location": {"row": 1, "column": 1},
            },
        ]
    )
    tasks = parse_mod.parse_ruff_json(payload)
    assert tasks[0]["message"] == "E501: line too long:"
    assert tasks[0]["line"] == 12
    assert tasks[1]["message"] == "F401"


def test_parse_ruff_json_builds_stable_tasks(parse_mod: ModuleType) -> None:
    payload = json.dumps(
        [
            {
                "filename": "src/disasterproject/utils/foo.py",
                "code": "F401",
                "message": "unused import",
                "location": {"row": 3, "column": 1},
            }
        ]
    )
    tasks = parse_mod.parse_ruff_json(payload)
    assert len(tasks) == 1
    assert tasks[0]["kind"] == "ruff"
    assert tasks[0]["path"].endswith("foo.py")
    assert tasks[0]["id"].startswith("ruff-")
    assert "tests/test_foo.py" in tasks[0]["suggested_tests"]


def test_parse_ruff_text_and_pytest_line(parse_mod: ModuleType) -> None:
    ruff_text = "app/app.py:10:1: E501 line too long\n"
    ruff_tasks = parse_mod.parse_ruff_text(ruff_text)
    assert ruff_tasks[0]["code"] == "E501"

    pytest_text = (
        "FAILED tests/test_smoke.py::test_index - AssertionError: boom\n"
        "tests/test_smoke.py:42: AssertionError: boom\n"
    )
    pytest_tasks = parse_mod.parse_pytest_tb_line(pytest_text)
    kinds = {t["kind"] for t in pytest_tasks}
    assert kinds == {"pytest"}
    assert any(t.get("nodeid") == "tests/test_smoke.py::test_index" for t in pytest_tasks)


def test_parse_pytest_junit(parse_mod: ModuleType) -> None:
    xml = """<?xml version="1.0" ?>
    <testsuite>
      <testcase classname="tests.test_smoke" name="test_index" file="tests/test_smoke.py">
        <failure message="AssertionError: boom">trace</failure>
      </testcase>
    </testsuite>
    """
    tasks = parse_mod.parse_pytest_junit(xml)
    assert len(tasks) == 1
    assert tasks[0]["suggested_tests"] == ["tests/test_smoke.py::test_index"]


def test_partition_shards_round_robin(shard_mod: ModuleType) -> None:
    items = [f"tests/test_{i}.py::test_a" for i in range(5)]
    shards = shard_mod.partition_shards(items, 3)
    assert len(shards) == 3
    assert sum(len(s) for s in shards) == 5
    assert shards[0][0].endswith("test_0.py::test_a")
    assert shards[1][0].endswith("test_1.py::test_a")
    assert shards[2][0].endswith("test_2.py::test_a")


def test_write_shard_manifests(tmp_path: Path, shard_mod: ModuleType) -> None:
    shards = [["a::t1", "a::t2"], ["b::t1"]]
    written = shard_mod.write_shard_manifests(shards, tmp_path)
    assert (tmp_path / "shard-00.json").exists()
    assert (tmp_path / "shard-01.txt").read_text(encoding="utf-8").strip() == "b::t1"
    assert (tmp_path / "shards-index.json").exists()
    assert len(written) >= 5


def test_worktree_path_for_uses_sanitized_id(common_mod: ModuleType) -> None:
    path = common_mod.worktree_path_for("task-1")
    assert path.name == "task-1"
    assert path.parent.name == ".worktrees"
