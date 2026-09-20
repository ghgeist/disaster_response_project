---
name: blind-peer-review
description: >-
  Spawns context-blind dual peer reviewers that critique a git diff without
  parent conversation history. Use during actor-pipeline Phase 1/2, after a
  producer or error-shard fix, or when the user asks for blind review,
  adversarial critique, or dual reviewers.
---

# Blind Peer Review

Fresh reviewers catch hallucinations better than self-review. Always use two independent reviewers with zero prior chat context.

## Inputs allowed (only these)

1. Original task spec / acceptance criteria
2. Git diff (required)
3. Optional: original file contents before the change

Do **not** pass: parent conversation, producer rationales, tool logs, or “what I intended.”

## How to spawn (Cursor)

Launch two parallel `Task` calls:

- `subagent_type`: `generalPurpose`
- No `resume`
- Distinct prompts labeled Reviewer A and Reviewer B
- Point each at `docs/agents/blind-reviewer-agent.md`

## Critique dimensions

1. Correctness vs the task spec
2. Behavioral parity / regression risk
3. Error handling and input hygiene (this Flask/ML repo)
4. Test gaps for the changed behavior
5. Unnecessary scope creep

## Required reviewer output schema

```markdown
## Verdict
APPROVE | REQUEST_CHANGES

## Findings
- severity: blocker|major|nit
  location: path:line or hunk summary
  issue: …
  fix: concrete suggestion

## Residual risks
- …
```

Rubber-stamp APPROVE with no findings is invalid unless the diff is empty or purely formatting with no behavior change—state that explicitly.

## Fixer consolidation

1. Union findings; merge duplicates by location+issue.
2. Apply blockers, then majors; nits only if cheap and on-path.
3. Re-run Ruff + targeted `python scripts/run_tests.py …`.
4. Re-review only if the diff changed materially (new logic, not typo-only).
