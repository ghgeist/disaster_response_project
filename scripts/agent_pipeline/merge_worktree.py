"""Merge a clean agent-pipeline worktree branch back into the current branch."""

from __future__ import annotations

import argparse
import json
import sys

from common import (
    PROJECT_ROOT,
    branch_name_for,
    run_git,
    sanitize_task_id,
    worktree_path_for,
)


def worktree_is_clean(worktree_path) -> bool:
    """Return True if the worktree has no uncommitted changes."""
    result = run_git(["status", "--porcelain"], cwd=worktree_path, check=True)
    return result.stdout.strip() == ""


def merge_worktree(
    task_id: str,
    *,
    strategy: str = "merge",
    allow_dirty: bool = False,
) -> dict[str, str]:
    """Merge or cherry-pick the task branch into the current checkout."""
    task_id = sanitize_task_id(task_id)
    path = worktree_path_for(task_id)
    branch = branch_name_for(task_id)

    if not path.exists():
        raise FileNotFoundError(f"Worktree not found: {path}")

    if not allow_dirty and not worktree_is_clean(path):
        raise RuntimeError(
            f"Worktree is dirty: {path}. Commit/stash there or pass --allow-dirty."
        )

    main_status = run_git(["status", "--porcelain"], cwd=PROJECT_ROOT)
    if main_status.stdout.strip():
        raise RuntimeError(
            "Current branch working tree is dirty; commit or stash before merging."
        )

    if strategy == "merge":
        run_git(["merge", "--no-ff", "-m", f"merge agent-pipeline task {task_id}", branch])
    elif strategy == "cherry-pick":
        tip = run_git(["rev-parse", branch]).stdout.strip()
        run_git(["cherry-pick", tip])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return {
        "task_id": task_id,
        "branch": branch,
        "path": str(path),
        "strategy": strategy,
        "status": "merged",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument(
        "--strategy",
        choices=["merge", "cherry-pick"],
        default="merge",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow merging even if the worktree has uncommitted changes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = merge_worktree(
            args.task_id,
            strategy=args.strategy,
            allow_dirty=args.allow_dirty,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"git error: {exc}", file=sys.stderr)
        if hasattr(exc, "stderr") and getattr(exc, "stderr"):
            print(exc.stderr, file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
