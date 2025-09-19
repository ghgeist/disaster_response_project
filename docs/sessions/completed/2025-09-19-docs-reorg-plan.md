---
title: "Planning Agent: Documentation Reorganization and ADR Cleanup"
date: "2025-09-19"
status: "completed"
tags: ["documentation", "adr", "information-architecture", "docs-reorg"]
author: "Codex CLI Agent"
related: ["docs/adr/adr_template.md", "docs/testing.md", "docs/model-naming-convention.md"]
---


# Planning Agent: Documentation Reorganization and ADR Cleanup

**Date**: 2025-09-19  
**Status**: Active  
**Priority**: High  
**Estimated Duration**: 2–4 hours  
**Tags**: documentation, ADR, IA, testing

## ?? Objective

Rationalize the `docs/` tree by separating immutable decisions (ADRs) from living guides (standards, runbooks, testing), resolve ADR numbering conflicts, add an ADR index, and introduce a new ADR for the model artifact naming decision — without touching `docs/sessions/**`.

## ?? Success Criteria

- [ ] New docs structure created under `docs/standards`, `docs/runbooks`, `docs/testing`, `docs/qa`, `docs/analysis`.
- [ ] Documents moved per mapping; no content loss; `docs/sessions/**` unchanged.
- [ ] ADR duplicate numbering resolved; ADR index (`docs/adr/README.md`) present and current.
- [ ] New ADR added: model artifact naming decision; linked to naming standard.
- [ ] Cross-links updated in `docs/testing.md` and top-level `README.md` doc links.

## ?? Context

Current `docs/` mixes ADRs, guides, test docs, and agent playbooks. `docs/adr/` includes non-ADR artifacts and duplicate numbering (`adr-003-*`). Consolidation will improve discoverability and governance of decisions vs. living documentation. An internal `docs/sessions/` area exists and must remain untouched.

## ?? Requirements

### Functional Requirements
- Create subfolders for standards/runbooks/testing/qa/analysis under `docs/`.
- Move files per the agreed mapping (below) and preserve history where possible.
- Add ADR index and create a new ADR for model naming.
- Resolve `adr-003` duplication by renumbering the later ADR.
- Update intra-doc links and key references.

### Technical Requirements
- No changes outside `docs/`; explicitly exclude `docs/sessions/**`.
- Offline-friendly; do not introduce external dependencies or downloads.
- Preserve file encodings/line endings; keep Markdown simple and portable.

### Quality Requirements
- Use consistent frontmatter (title/date/status/tags) for ADRs.
- Keep guides concise, link-rich, and scoped to their category.
- Document renumbering in ADR index with a note if applicable.

## ??? Approach

1. Create folders: `docs/{standards,runbooks,testing,qa,analysis}`.
2. Move files:
   - `deployment-configuration.md` → `runbooks/deployment.md`.
   - `gdrive_testing.md` → `testing/gdrive.md` (or section under testing index).
   - `manual_ui_test_checklist.md` → `qa/ui-manual-checklist.md`.
   - `model-naming-convention.md` → `standards/model-naming.md`.
   - `performance_testing.md` → `testing/performance.md`.
   - `test_faq.md` → `testing/faq.md` (or merge into testing index).
3. Clean ADR folder:
   - Keep numbered ADRs and `adr_template.md`.
   - Move non-ADR artifacts to `docs/analysis/`:
     - `original_model_analysis_versus_model_with_updated_tokenizer.md`.
     - `original_optimized_hyperparameters.json`.
4. Resolve ADR numbering conflict:
   - Retain `adr-003-hybrid-model-deployment-strategy.md` as 003.
   - Renumber `adr-003-fix-default-n-jobs-constant-redefinition.md` → `adr-005-fix-default-n-jobs-constant-redefinition.md` (next available).
5. Add new ADR for model naming decision:
   - `adr-00X-model-artifact-naming-standard.md` (choose next index), linking to `docs/standards/model-naming.md`.
6. Add `docs/adr/README.md` index with table (number, title, date, status).
7. Update `docs/testing.md` to act as index; link to `testing/performance.md`, `testing/gdrive.md`, and `testing/faq.md`.
8. Update top-level `README.md` documentation links to new locations.
9. Verify no references touch `docs/sessions/**`.

## ?? Acceptance Criteria

- Directory structure matches plan; moved files open in their new paths.
- ADR index lists all ADRs with unique numbers and accurate statuses/dates.
- New model-naming ADR present; cross-links between ADR and standard work.
- No broken relative links in moved docs within `docs/`.
- `docs/sessions/` contents unchanged.

## ?? Related Work

- Template: `docs/adr/adr_template.md`.
- Affected guides: `docs/testing.md`, `docs/deployment-configuration.md`, `docs/model-naming-convention.md`, etc.
- ADRs: `adr-003-hybrid-model-deployment-strategy.md`, `adr-003-fix-default-n-jobs-constant-redefinition.md` (to renumber).

## ?? Metrics

- Structural: Presence of all new folders and expected files at new paths.
- Consistency: Zero duplicate ADR numbers; ADR index completeness.
- Link health: Spot-check key links (manual, offline) after moves.

## ?? Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Broken links after moves | Med | Med | Search/replace known paths; run a quick ripgrep for old filenames. |
| ADR cross-references stale | Med | Med | Update links during renumber; add “Renumbered from …” note in ADR index. |
| Sessions content disturbed | High | Low | Explicitly exclude `docs/sessions/**` from any operation. |
| Merge conflicts with parallel doc changes | Med | Low | Batch small commits; coordinate timing; keep moves atomic. |

## ?? Deliverables

- [ ] New docs folder structure with moved guides/runbooks/testing/QA.
- [ ] Cleaned `docs/adr/` with index and resolved numbering.
- [ ] New ADR: model artifact naming decision.
- [ ] Updated links in `docs/testing.md` and top-level `README.md`.
