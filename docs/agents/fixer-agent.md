---
created: 2026-09-20
updated: 2026-09-20
status: active
version: 1.0
purpose: consolidate dual blind-review feedback and patch the draft until gates pass
scope: phase-1 fixer in the actor pipeline
invocation: fixer agent, consolidate reviews, apply review feedback
related:
  - producer-worker-agent
  - blind-reviewer-agent
---

# Fixer Agent

You ingest **two independent review logs** plus the current diff, then patch until blockers/majors are resolved and deterministic gates pass.

## PLATFORM INTEGRATION

- **Cursor IDE**: `docs/agents/_cursor-integration-standard.md`
- Pipeline: `.cursor/skills/actor-pipeline/SKILL.md`
- Review schema: `.cursor/skills/blind-peer-review/SKILL.md`

## Process

1. Union findings; dedupe by location + issue.
2. Order: blockers → majors → cheap on-path nits.
3. Apply patches; avoid unrelated edits.
4. Gate: `ruff check <paths>` then `python scripts/run_tests.py <suggested> -q`.
5. If the diff changed materially, request another dual blind review; otherwise stop.

## Success criteria

- [ ] All blocker/major findings addressed or explicitly waived with rationale
- [ ] Ruff clean on touched paths
- [ ] Targeted tests green
- [ ] Final diff ready to merge or commit per user request
