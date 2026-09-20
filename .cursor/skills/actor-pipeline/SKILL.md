---
name: actor-pipeline
description: >-
  Runs the three-phase Producer-Reviewer-Fixer actor pipeline with worktree
  error sharding and pytest test sharding. Use for large refactors, mass
  lint/test failures, multi-file migrations, or when the user asks for actor
  pipeline, blind peer review, worktree shards, or parallel fix agents.
---

# Actor Pipeline

Orchestrate non-trivial Python work with context-isolated actors and deterministic gates (Ruff + pytest). Prefer this over a single mega-prompt session.

## Deterministic gates

- Lint: `ruff check <paths>`
- Tests: `python scripts/run_tests.py <paths-or-nodeids> -q`
- Do not treat LLM judgment as done until gates pass for the assigned scope.

## When to use which phase

| Phase | Enter when |
|-------|------------|
| 1 File change + blind review | Scoped produce/fix of one file or tight change set |
| 2 Worktree error sharding | Many independent Ruff/pytest failures; need parallel fixes |
| 3 Test suite sharding | Suite fails broadly after compile/lint is clean; partition tests |

## Phase 1 — Producer → Blind reviewers → Fixer

1. **Producer** implements the task on the assigned file(s). Keep the change minimal.
2. Capture `git diff` for the change (and optionally the pre-change file contents).
3. Spawn **two fresh** `Task` subagents (`generalPurpose`, no `resume`, no shared chat history). Each gets only:
   - Original task spec / acceptance criteria
   - The git diff (and optional original file)
   - Instructions from `.cursor/skills/blind-peer-review/SKILL.md` and `docs/agents/blind-reviewer-agent.md`
4. **Fixer** consolidates both review logs (dedupe, blockers first) and patches.
5. Re-run Ruff + targeted tests. Loop review only if material changes remain.
6. Stop when gates are green and both reviewers would accept (or only non-blocking nits remain).

Producer never self-reviews. Reviewers must not receive prior conversation or tool traces.

## Phase 2 — Worktree-sharded error resolution

1. Collect failures: Ruff JSON and/or pytest output.
2. `python scripts/agent_pipeline/parse_failures.py … -o queue.json`
3. For each task (or batch of non-overlapping paths):
   - `python scripts/agent_pipeline/worktree_pool.py add --task-id <id>`
   - Assign one error-shard agent to that worktree (`docs/agents/error-shard-agent.md`)
   - Agent loop: parse diagnostic → edit → ruff → targeted tests → blind dual review on the localized diff → stop when clean
4. Merge with `python scripts/agent_pipeline/merge_worktree.py …` only from a clean worktree.
5. Never let parallel agents share one dirty working tree.

## Phase 3 — Test sharding

1. `python scripts/agent_pipeline/shard_tests.py --shards N -o experiments/agent_pipeline/<YYYY-MM-DD>/`
2. Each test-shard agent owns one manifest (`docs/agents/test-shard-agent.md`).
3. Run `python scripts/run_tests.py @manifest` or pass listed node ids until the shard is green.
4. Convergence: all shard manifests pass.

## Parent agent / human role

You are the merge controller: create tasks, spawn isolated agents, merge clean results, refuse dirty merges. Scripts isolate and decompose; they do not spawn LLM workers.

## Artifact location

Write queue/shard artifacts under `experiments/agent_pipeline/{YYYY-MM-DD}/` (date from the system clock, not guessed).
