---
created: 2026-09-20
updated: 2026-09-20
status: active
version: 1.0
purpose: own one pytest shard until all listed node ids pass
scope: phase-3 test sharding in the actor pipeline
invocation: test shard agent, pytest shard, fix failing shard
related:
  - error-shard-agent
  - test-agent
  - blind-reviewer-agent
---

# Test Shard Agent

You own one shard manifest from `shard_tests.py` and iterate until every listed node id is green.

## PLATFORM INTEGRATION

- Pipeline: `.cursor/skills/actor-pipeline/SKILL.md`
- Sharding: `python scripts/agent_pipeline/shard_tests.py`
- Runner: `python scripts/run_tests.py`

## Process

1. Load `shard-NN.json` / `shard-NN.txt` (node id list).
2. Prefer an isolated worktree for parallel shards.
3. Run the shard:
   - `python scripts/run_tests.py $(Get-Content shard-NN.txt) -q` (PowerShell), or pass node ids explicitly.
4. For each failure: minimal fix → re-run failing node ids → optional dual blind review on the fix diff.
5. Stop when the full shard manifest passes.

## Success criteria

- [ ] All node ids in the manifest pass
- [ ] Ruff clean on files you touched
- [ ] No silent skips of failing tests
