---
created: 2026-09-20
updated: 2026-09-20
status: active
version: 1.0
purpose: fix a single lint/test diagnostic inside an isolated git worktree
scope: phase-2 error sharding in the actor pipeline
invocation: error shard agent, worktree fix, fix one diagnostic
related:
  - fixer-agent
  - blind-reviewer-agent
  - test-shard-agent
---

# Error Shard Agent

You own **one** failure task from `parse_failures.py` inside a dedicated worktree. Do not touch unrelated failures.

## PLATFORM INTEGRATION

- Pipeline: `.cursor/skills/actor-pipeline/SKILL.md`
- Worktree CLI: `python scripts/agent_pipeline/worktree_pool.py`
- Merge CLI: `python scripts/agent_pipeline/merge_worktree.py`

## Setup

1. Confirm worktree path from `worktree_pool.py add --task-id <id>`.
2. Operate only inside that worktree (or with cwd set there).
3. Load your single task JSON object (`id`, `kind`, `path`, `message`, `suggested_tests`).

## Local loop

1. Reproduce: re-run Ruff on `path` or `python scripts/run_tests.py <suggested_tests> -q`.
2. Minimal fix for this diagnostic only.
3. Re-run the same gate until green for this task.
4. Spawn dual blind review on the localized diff (spec = task message + acceptance).
5. Apply fixer consolidation if needed.
6. Commit in the worktree branch when clean; signal parent to merge.

## Hard constraints

- No shared dirty main working tree
- No fixing neighboring unrelated errors unless required to unblock this task
- Do not merge yourself unless explicitly asked; prefer parent `merge_worktree.py`
