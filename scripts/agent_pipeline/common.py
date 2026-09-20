"""Shared helpers for agent pipeline scripts."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKTREES_DIR = PROJECT_ROOT / ".worktrees"
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def sanitize_task_id(task_id: str) -> str:
    """Return a filesystem-safe task id or raise ValueError."""
    cleaned = task_id.strip()
    if not TASK_ID_PATTERN.fullmatch(cleaned):
        raise ValueError(
            f"Invalid task id {task_id!r}. Use 1-64 chars: letters, digits, . _ -"
        )
    return cleaned


def worktree_path_for(task_id: str) -> Path:
    """Resolve the worktree directory for a sanitized task id."""
    return WORKTREES_DIR / sanitize_task_id(task_id)


def branch_name_for(task_id: str) -> str:
    """Return the dedicated branch name for a task worktree."""
    return f"agent-pipeline/{sanitize_task_id(task_id)}"


def run_git(args: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a git command and return the completed process."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd or PROJECT_ROOT,
        check=check,
        text=True,
        capture_output=True,
    )


def ensure_worktrees_dir() -> Path:
    """Create and return the .worktrees directory."""
    WORKTREES_DIR.mkdir(parents=True, exist_ok=True)
    return WORKTREES_DIR
