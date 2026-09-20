"""Partition pytest node ids into shard manifests for parallel agents."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from common import PROJECT_ROOT


def collect_nodeids(*, pytest_args: list[str] | None = None) -> list[str]:
    """Collect pytest node ids via `--collect-only -q`."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_tests.py"),
        "--collect-only",
        "-q",
        *(pytest_args or ["tests"]),
    ]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode not in (0,):
        # pytest collect-only still prints node ids; surface stderr if empty
        if not result.stdout.strip():
            raise RuntimeError(
                f"pytest collect failed ({result.returncode}): {result.stderr}"
            )

    nodeids: list[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("=") or " " in line:
            # skip summary lines like "123 tests collected in 0.1s"
            if "::" not in line:
                continue
        if "::" in line or (line.endswith(".py") and "/" in line.replace("\\", "/")):
            if line[0].isdigit() and " test" in line:
                continue
            nodeids.append(line)
    return nodeids


def partition_shards(items: list[str], shard_count: int) -> list[list[str]]:
    """Split items into shard_count nearly even contiguous shards."""
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if not items:
        return [[] for _ in range(shard_count)]

    shards: list[list[str]] = [[] for _ in range(shard_count)]
    for index, item in enumerate(items):
        shards[index % shard_count].append(item)
    return shards


def write_shard_manifests(
    shards: list[list[str]],
    output_dir: Path,
) -> list[Path]:
    """Write shard-NN.json and shard-NN.txt manifests; return paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    index_payload = {"shards": [], "created": datetime.now().strftime("%Y-%m-%d")}
    for i, items in enumerate(shards):
        stem = f"shard-{i:02d}"
        json_path = output_dir / f"{stem}.json"
        txt_path = output_dir / f"{stem}.txt"
        payload = {"shard": i, "count": len(items), "nodeids": items}
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        txt_path.write_text("\n".join(items) + ("\n" if items else ""), encoding="utf-8")
        written.extend([json_path, txt_path])
        index_payload["shards"].append(
            {"shard": i, "json": str(json_path), "txt": str(txt_path), "count": len(items)}
        )
    index_path = output_dir / "shards-index.json"
    index_path.write_text(json.dumps(index_payload, indent=2) + "\n", encoding="utf-8")
    written.append(index_path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, required=True, help="Number of shards")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output directory for manifests",
    )
    parser.add_argument(
        "--nodeids-file",
        type=Path,
        help="Optional file of node ids (skip live collect)",
    )
    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="Extra args forwarded to pytest collect (default: tests)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.nodeids_file:
            items = [
                line.strip()
                for line in args.nodeids_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            items = collect_nodeids(pytest_args=args.pytest_args or None)
        shards = partition_shards(items, args.shards)
    except (ValueError, RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.output:
        out_dir = args.output
    else:
        day = datetime.now().strftime("%Y-%m-%d")
        out_dir = PROJECT_ROOT / "experiments" / "agent_pipeline" / day

    paths = write_shard_manifests(shards, out_dir)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "shard_count": args.shards,
                "total_nodeids": len(items),
                "files": [str(p) for p in paths],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
