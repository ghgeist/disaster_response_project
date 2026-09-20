"""Git worktree pool for isolated agent-pipeline workers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import (
    PROJECT_ROOT,
    WORKTREES_DIR,
    branch_name_for,
    ensure_worktrees_dir,
    run_git,
    sanitize_task_id,
    worktree_path_for,
)


def add_worktree(task_id: str, *, base_ref: str = "HEAD") -> dict[str, str]:
    """Create a linked worktree and branch for the task."""
    task_id = sanitize_task_id(task_id)
    path = worktree_path_for(task_id)
    branch = branch_name_for(task_id)
    ensure_worktrees_dir()

    if path.exists():
        raise FileExistsError(f"Worktree already exists: {path}")

    run_git(["worktree", "add", "-b", branch, str(path), base_ref])
    return {
        "task_id": task_id,
        "path": str(path),
        "branch": branch,
        "base_ref": base_ref,
    }


def list_worktrees() -> list[dict[str, str]]:
    """List agent-pipeline worktrees under .worktrees/."""
    result = run_git(["worktree", "list", "--porcelain"])
    entries: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            if current:
                entries.append(current)
                current = {}
            continue
        if line.startswith("worktree "):
            current = {"path": line[len("worktree ") :]}
        elif line.startswith("branch "):
            current["branch"] = line[len("branch ") :].removeprefix("refs/heads/")
        elif line.startswith("HEAD "):
            current["head"] = line[len("HEAD ") :]
    if current:
        entries.append(current)

    root = WORKTREES_DIR.resolve()
    return [
        entry
        for entry in entries
        if Path(entry.get("path", "")).resolve().is_relative_to(root)
    ]


def remove_worktree(task_id: str, *, force: bool = False) -> dict[str, str]:
    """Remove a task worktree and optionally prune stale metadata."""
    task_id = sanitize_task_id(task_id)
    path = worktree_path_for(task_id)
    branch = branch_name_for(task_id)
    if not path.exists():
        raise FileNotFoundError(f"Worktree not found: {path}")

    args = ["worktree", "remove", str(path)]
    if force:
        args.append("--force")
    run_git(args)
    run_git(["worktree", "prune"], check=False)
    return {"task_id": task_id, "path": str(path), "branch": branch, "removed": "true"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    add_p = sub.add_parser("add", help="Create a worktree for a task id")
    add_p.add_argument("--task-id", required=True)
    add_p.add_argument("--base-ref", default="HEAD")

    sub.add_parser("list", help="List agent-pipeline worktrees")

    rm_p = sub.add_parser("remove", help="Remove a task worktree")
    rm_p.add_argument("--task-id", required=True)
    rm_p.add_argument("--force", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "add":
            payload = add_worktree(args.task_id, base_ref=args.base_ref)
        elif args.command == "list":
            payload = {"worktrees": list_worktrees(), "root": str(PROJECT_ROOT)}
        elif args.command == "remove":
            payload = remove_worktree(args.task_id, force=args.force)
        else:
            parser.error(f"Unknown command: {args.command}")
            return 2
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # git failures
        print(f"git error: {exc}", file=sys.stderr)
        if hasattr(exc, "stderr") and exc.stderr:
            print(exc.stderr, file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
