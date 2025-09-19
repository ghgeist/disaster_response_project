---
title: "Planning Agent: Metrics CSV Resolution"
date: "2025-09-19"
status: "active"
tags: ["plan", "metrics", "fallback", "disasterproject"]
author: "codex-cli"
related: [
  "docs/agents/planning-agent.md",
  "src/disasterproject/utils/config.py",
  "src/disasterproject/utils/metrics_io.py",
  "src/disasterproject/utils/experimental_paths.py",
  "app/services.py"
]
---


# Planning Agent: Metrics CSV Resolution

Date: 2025-09-19  
Status: Active  
Priority: High  
Estimated Duration: 2–4 hours  
Tags: metrics, ETL, app-integrations, robustness

## Objective

Ensure the application reliably loads baseline and optimized median metrics by output class, even when only date-prefixed artifacts exist. Implement a resolver-based fallback so the app uses canonical paths when available and otherwise discovers the latest valid metrics automatically.

## Success Criteria

- [ ] No warning logs "Metrics CSV not found" when dated or experimental metrics exist
- [ ] `load_metric_frames()` returns non-None DataFrames for both baseline and optimized in environments with only dated files
- [ ] UI comparison chart renders precision/recall/F1 for Baseline vs Optimized
- [ ] Unit tests for resolver and fallback behavior pass locally (`pytest -q` or smoke tests)

## Context

The code references canonical metrics files via constants in `src/disasterproject/utils/config.py`:
- `BASE_METRICS_PATH = data/04_fct/fct_median_metrics_by_output_class_base.csv`
- `OPT_METRICS_PATH  = data/04_fct/fct_median_metrics_by_output_class_optimized.csv`

In this workspace, only date-prefixed equivalents exist (e.g., `data/04_fct/2024-04-22_fct_median_metrics_by_output_class_optimized.csv`). As a result, `read_metrics_csv()` logs warnings and returns `None`, and the UI lacks optimized metrics.

Related loaders and consumers:
- Loader: `src/disasterproject/utils/metrics_io.py:read_metrics_csv`
- Consumer: `app/services.py:load_metric_frames` then `ModelHealthMonitor.get_performance_metrics`
- Discovery utility available: `src/disasterproject/utils/experimental_paths.py` (finds latest experimental artifacts)

## Requirements

### Functional Requirements
- Prefer canonical file if present; otherwise automatically select latest suitable dated metrics for base and optimized
- Validate schema minimally (required columns present) before accepting a fallback
- Log the selected path at INFO level when a fallback is used

### Technical Requirements
- Implement a small resolver utility under `src/disasterproject/utils/` using `pathlib`
- Integrate resolver in `app/services.py:load_metric_frames()` without breaking call sites
- Keep `read_metrics_csv` focused on IO (no discovery logic inside it)

### Quality Requirements
- Add targeted unit tests covering canonical, dated fallback, experiment fallback, and no-artifact cases
- Keep changes minimal and aligned with project style (Black/Ruff)

## Approach

1) Add `metrics_resolver.py` in `disasterproject.utils`:
- API: `resolve_metrics_path(kind: Literal['base','optimized']) -> Optional[Path]`
- Order of resolution:
  - Canonical constant path exists → use it
  - Glob `data/04_fct/*_fct_median_metrics_by_output_class_base.csv` or `*_optimized.csv`, pick newest by mtime
  - If none found, consult `ExperimentalPathManager.get_latest_experimental_artifacts()` and accept `metrics_path` if schema matches
- Validate required columns: `precision`, `recall`, `f1-score` (case-insensitive), and `output_class` if present

2) Update `app/services.py:load_metric_frames()`:
- Attempt `read_metrics_csv(BASE_METRICS_PATH)` and `read_metrics_csv(OPT_METRICS_PATH)`
- If `None`, call resolver and re-attempt `read_metrics_csv(resolved_path)`
- Log which path was selected for each kind when using a fallback

3) Add tests `tests/test_metrics_resolver.py`:
- Canonical present → returns canonical
- Only dated present → returns newest dated
- Only experiment artifact present → returns experiment metrics path
- No artifacts → returns `None`

4) Optional maintenance convenience:
- `scripts/promote_metrics.py` copies latest dated CSVs to canonical filenames for explicit deployments
- Document in `scripts/README.md`

## Acceptance Criteria

- Resolver returns expected paths across scenarios; tests pass
- `load_metric_frames()` yields DataFrames in dated-only environments; UI shows baseline vs optimized metrics without errors
- Logs include info about any fallback files used
- No regressions in existing tests (`pytest tests/test_smoke.py -q` passes)

## Related Work

- docs/agents/planning-agent.md: Plan context and decision rationale
- src/disasterproject/utils/experimental_paths.py: Latest experiment artifact discovery
- app/services.py: Metrics load and summarization

## Metrics

- Error rate: count of "Metrics CSV not found" warnings during app start → target 0
- Coverage: unit tests covering resolver paths → target ≥ 4 scenarios
- UI validation: presence of comparison data tuple (precision/recall/F1 arrays of length 2)

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Filename pattern drift | Medium | Medium | Use robust glob patterns and schema validation before accept |
| Selecting stale artifacts | Medium | Low | Choose newest by modification time; allow override via env var later if needed |
| Platform path quirks | Low | Low | Use `pathlib` and avoid string path ops |
| Schema mismatch | Medium | Low | Validate required columns; log warning and skip |

## Deliverables

- [ ] `src/disasterproject/utils/metrics_resolver.py`
- [ ] Updates in `app/services.py` to invoke resolver on fallback
- [ ] `tests/test_metrics_resolver.py`
- [ ] Optional: `scripts/promote_metrics.py` and `scripts/README.md` entry

