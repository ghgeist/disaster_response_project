"""Parse Ruff and pytest failure output into agent task queue JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

PYTEST_LINE_RE = re.compile(
    r"^(?P<path>[^:]+):(?P<line>\d+):\s*(?P<message>.+)$"
)
RUFF_LINE_RE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+):\d+:\s*(?P<code>[A-Z0-9]+)\s+(?P<message>.+)$"
)


def _task_id(kind: str, path: str, message: str) -> str:
    digest = hashlib.sha1(f"{kind}|{path}|{message}".encode("utf-8")).hexdigest()[:10]
    stem = Path(path).stem.replace(" ", "_")[:24] or "item"
    return f"{kind}-{stem}-{digest}"


def parse_ruff_json(payload: str) -> list[dict[str, Any]]:
    """Parse `ruff check --output-format json` into tasks."""
    data = json.loads(payload) if payload.strip() else []
    if not isinstance(data, list):
        raise ValueError("Ruff JSON must be a list of diagnostics")

    tasks: list[dict[str, Any]] = []
    for item in data:
        path = str(item.get("filename") or item.get("file") or "")
        code = str(item.get("code") or "RUFF")
        message = str(item.get("message") or "")
        location = item.get("location") or {}
        line = location.get("row") or item.get("location", {}).get("row")
        tasks.append(
            {
                "id": _task_id("ruff", path, f"{code}:{message}"),
                "kind": "ruff",
                "path": path,
                "line": line,
                "code": code,
                "message": f"{code}: {message}".strip(": "),
                "suggested_tests": _suggest_tests(path),
            }
        )
    return tasks


def parse_ruff_text(payload: str) -> list[dict[str, Any]]:
    """Parse default Ruff text diagnostics into tasks."""
    tasks: list[dict[str, Any]] = []
    for raw in payload.splitlines():
        line = raw.strip()
        match = RUFF_LINE_RE.match(line)
        if not match:
            continue
        path = match.group("path")
        code = match.group("code")
        message = match.group("message")
        tasks.append(
            {
                "id": _task_id("ruff", path, f"{code}:{message}"),
                "kind": "ruff",
                "path": path,
                "line": int(match.group("line")),
                "code": code,
                "message": f"{code}: {message}",
                "suggested_tests": _suggest_tests(path),
            }
        )
    return tasks


def parse_pytest_tb_line(payload: str) -> list[dict[str, Any]]:
    """Parse pytest `--tb=line` style failure lines into tasks."""
    tasks: list[dict[str, Any]] = []
    for raw in payload.splitlines():
        line = raw.strip()
        if "FAILED" in line and "::" in line:
            # e.g. FAILED tests/test_foo.py::test_bar - AssertionError
            node = line.split("FAILED", 1)[1].strip()
            node_id, _, msg = node.partition(" - ")
            path = node_id.split("::", 1)[0]
            tasks.append(
                {
                    "id": _task_id("pytest", node_id, msg or "FAILED"),
                    "kind": "pytest",
                    "path": path,
                    "line": None,
                    "code": None,
                    "message": msg or "FAILED",
                    "nodeid": node_id,
                    "suggested_tests": [node_id],
                }
            )
            continue

        match = PYTEST_LINE_RE.match(line)
        if not match:
            continue
        path = match.group("path")
        if not path.endswith(".py"):
            continue
        message = match.group("message")
        tasks.append(
            {
                "id": _task_id("pytest", path, message),
                "kind": "pytest",
                "path": path,
                "line": int(match.group("line")),
                "code": None,
                "message": message,
                "suggested_tests": _suggest_tests(path),
            }
        )
    return tasks


def parse_pytest_junit(payload: str) -> list[dict[str, Any]]:
    """Parse a minimal JUnit XML pytest report into tasks."""
    root = ET.fromstring(payload)
    tasks: list[dict[str, Any]] = []
    for case in root.iter("testcase"):
        failure = case.find("failure")
        error = case.find("error")
        node = failure if failure is not None else error
        if node is None:
            continue
        classname = case.attrib.get("classname", "")
        name = case.attrib.get("name", "")
        file_attr = case.attrib.get("file") or classname.replace(".", "/") + ".py"
        nodeid = f"{file_attr}::{name}" if name else file_attr
        message = node.attrib.get("message") or (node.text or "failed").strip().splitlines()[0]
        tasks.append(
            {
                "id": _task_id("pytest", nodeid, message),
                "kind": "pytest",
                "path": file_attr,
                "line": None,
                "code": None,
                "message": message,
                "nodeid": nodeid,
                "suggested_tests": [nodeid],
            }
        )
    return tasks


def _suggest_tests(path: str) -> list[str]:
    p = Path(path.replace("\\", "/"))
    name = p.name
    if name.startswith("test_") and name.endswith(".py"):
        return [str(p).replace("\\", "/")]
    stem = p.stem
    return [f"tests/test_{stem}.py"]


def parse_failures(
    *,
    kind: str,
    payload: str,
) -> list[dict[str, Any]]:
    """Dispatch to the appropriate parser."""
    if kind == "ruff-json":
        return parse_ruff_json(payload)
    if kind == "ruff-text":
        return parse_ruff_text(payload)
    if kind == "pytest-line":
        return parse_pytest_tb_line(payload)
    if kind == "pytest-junit":
        return parse_pytest_junit(payload)
    raise ValueError(f"Unknown kind: {kind}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        required=True,
        choices=["ruff-json", "ruff-text", "pytest-line", "pytest-junit"],
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Input file (default: stdin)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON path (default: stdout)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.input:
        payload = Path(args.input).read_text(encoding="utf-8")
    else:
        payload = sys.stdin.read()

    try:
        tasks = parse_failures(kind=args.kind, payload=payload)
    except (ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    queue = {"tasks": tasks, "count": len(tasks)}
    text = json.dumps(queue, indent=2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
