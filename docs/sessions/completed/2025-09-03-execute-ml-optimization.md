---
title: "ML Optimization: Negations, RF Downsizing, Thresholding"
date: "2025-09-03"
status: "completed"
session_type: "execute"
tags: ["ml", "nlp", "preprocessing", "modeling", "thresholding"]
author: "runner"
related: ["docs/adr/adr-002-tokenization-trade-offs.md"]
---

# ML Optimization: Negations, RF Downsizing, Thresholding

**Session Type**: EXECUTE  
**Priority**: High  
**Estimated Duration**: 1–2 days  
**Status**: In Progress

## 🎯 Objective

Implement three targeted improvements to the text classification pipeline:
1. Preserve core negations during preprocessing
2. Downsize RandomForest model without materially hurting recall  
3. Apply per-label F2-optimized thresholds to eight high-impact labels

## 📋 Success Criteria

- [ ] Negations preserved: "no", "not", "never", "none", "without", "nor"
- [ ] Acceptance examples work:
  - "We do not need medical help" → related=True, medical_help=False
  - "No water. Please send water." → water=True
- [ ] RF parameters: `n_estimators=100`, `max_depth=25`, `min_samples_leaf=2`, `max_features="sqrt"`
- [ ] Model size ≤ 150–200 MB (down from ~561 MB)
- [ ] Macro recall within ±1 point of baseline
- [ ] Cold load time under a few seconds
- [ ] Thresholding for 8 labels: `medical_help`, `search_and_rescue`, `water`, `food`, `shelter`, `hospitals`, `security`, `weather_related`
- [ ] Zero-recall cases eliminated for the eight labels

## ✅ Progress Update (2025-09-03)

- **Preprocessing**: ✅ Implemented contraction normalization and negation keep-list
- **Modeling**: ✅ Applied downsized RF parameters in pipeline
- **Inference**: ✅ Updated `ModelService.predict` for threshold loading
- **Training**: ✅ Experimental training completed; needs re-run for JSON artifacts

## ▶️ Next Steps

1. **Re-run experimental training** to generate `thresholds.json`, `label_order.json`, `MODEL_INFO.json`
2. **Frozen-eval comparison** with gates: macro recall ↑, weighted F1 drop ≤2 pts, zero-recall eliminated
3. **Promote if gates pass**: copy experimental artifacts to `models/`
4. **App smoke tests**: verify negation cases through Flask UI

## 🧪 Validation Plan

### Unit Checks (15–25 min)
- Tokenizer: negation cases pass
- Thresholds: `thresholds.json` loads; only 8 labels present; missing file falls back to 0.5
- Label order: `label_order.json` matches model's classes

### Retrain + Evaluate (30–40 min)
- Train downsized RF; save to dated artifact
- Run frozen eval; export per-label metrics
- Record: model size, macro recall Δ, weighted F1 Δ

### Compute Thresholds (15–20 min)
- Generate F2-optimal thresholds for 8 labels
- Save `model/thresholds.json` and `model/label_order.json`
- Re-score frozen eval with thresholds

### Gate and Promote (10 min)
- Size ≤ 200 MB
- Macro recall within ±1 pt of baseline
- Weighted F1 drop ≤ 2 pts
- No zero-recall among 8 labels

### App Integration + Smoke Test (20–30 min)
- Point app to local model path
- Verify `ModelService.predict` behavior
- Test critical cases in UI:
  1. "We do **not** need medical help" → medical_help=False
  2. "No water here. Please send water." → water=True
  3. "People trapped on roof. Send **search and rescue**." → search_and_rescue=True
  4. "Storm destroyed houses" → weather_related=True
  5. "All safe, no injuries reported" → medical_help=False
  6. "Hospital is closed" → hospitals=True

## 🚨 Rollback Plan

If any gate fails:
- Keep baseline model
- Switch to TF-IDF + One-vs-Rest LogisticRegression
- Same gates apply: size <50 MB, fast load, recall targets

## 📊 Key Metrics

- Model artifact size (MB)
- Macro recall (frozen eval) vs baseline (Δ points)
- Weighted F1 (frozen eval) vs baseline (Δ points)
- Cold load time (seconds)
- Zero-recall label count among 8 high-impact labels
