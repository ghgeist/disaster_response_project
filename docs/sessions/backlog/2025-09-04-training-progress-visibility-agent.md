---
title: "Model Training Progress Visibility Agent: Add progress indicators to training scripts"
date: "2025-09-04"
status: "Active"
tags: ["documentation", "training", "progress", "ml", "tooling"]
author: "runner"
related: ["docs/agents/active/2025-09-02-update_ml_models.md"]
---


# Model Training Progress Visibility Agent

**Date**: 2025-09-04  
**Status**: Active  
**Priority**: Medium  
**Estimated Duration**: 45–60 minutes  
**Tags**: training, progress, UX, logging

## 🎯 Objective

Add clear, user-visible training progress to the model training scripts so it's easy to see how far along training is and estimate remaining time.

## 📋 Success Criteria

- [ ] `scripts/04_create_production_model.py` prints incremental training progress
- [ ] `scripts/03_create_experimental_model.py` prints incremental training progress
- [ ] Option to enable/disable via CLI flag (e.g., `--verbose-training`)
- [ ] Minimal runtime overhead (<2% slow-down on baseline)
- [ ] Works cross-platform (Windows/macOS/Linux) in standard terminals

## 🔍 Context

Training uses `MultiOutputClassifier(RandomForestClassifier)` over many labels. Default `.fit()` provides no feedback, which makes long runs opaque. Lightweight, clear progress output improves developer experience and observability.

## 📝 Requirements

### Functional Requirements
- Provide user-facing progress during training (either per-label or periodic updates)
- CLI flag `--verbose-training` to toggle progress output
- Optional mode selection `--progress-mode=[logs|bar]` (default: `logs`)

### Technical Requirements  
- Maintain current pipeline (`MultiOutputClassifier` + `RandomForestClassifier`)
- Preserve parallelism settings (`n_jobs` usage)
- Avoid invasive refactors; prefer thin wrappers and callbacks

### Quality Requirements
- Output must be concise and readable; avoid flooding logs
- Disable progress output in non-interactive contexts by default
- No changes to model reproducibility or evaluation logic

## 🛠️ Approach

1. Quick win: enable estimator verbosity when `--verbose-training` is set
   - Set `RandomForestClassifier(verbose=1)` via script flag
2. Polished option: per-label progress bar
   - Manually loop over labels and fit `RandomForestClassifier` per target with `tqdm`
   - Aggregate into a `MultiOutputClassifier`-equivalent structure
3. Logging
   - Use existing `setup_logging()`; gate progress logs on the flag

## 📊 Acceptance Criteria

- With `--verbose-training`, console shows incremental progress (either “building tree X of Y” or a progress bar such as `label 12/36`)
- Training completes successfully with and without the flag
- No errors on Windows terminals; output stays within one or few lines

## 🔗 Related Work

- `src/disaster_classifier/models/pipeline.py`
- `scripts/04_create_production_model.py`
- `scripts/03_create_experimental_model.py`
- `docs/agents/active/2025-09-02-update_ml_models.md`

## 📈 Metrics

- Time to train (baseline vs. verbose mode)
- Number of labels trained vs. total
- Estimated vs. actual total duration (optional)

## 🚨 Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Verbose logs slow training | Low | Low | Default to concise logs; allow opt-in verbosity |
| Terminal width issues on Windows | Low | Medium | Use simple text logs as default; bar optional |
| Future estimator changes | Medium | Medium | Keep logic behind a flag; isolate wrapper |

## 📄 Deliverables

- [ ] Updated scripts with `--verbose-training` and optional `--progress-mode`
- [ ] Minor updates to `README.md` documenting usage
- [ ] Before/after console screenshots (in `docs/images/`)


