---
title: "Threshold Optimization Analysis & Recommendations"
date: "2025-11-06"
status: "active"
tags: ["ml", "thresholds", "optimization", "production"]
author: "ML Engineer Agent"
related: [
  "docs/sessions/completed/2025-11-04-model-performance-improvement-plan.md",
  "docs/sessions/completed/2025-11-06-vocabulary-size-optimization.md"
]
---

# Threshold Optimization Analysis & Recommendations

## Executive Summary

**Finding**: Default threshold of 0.5 is **WAY too high** for critical categories.

**Impact**: 
- With 0.5 defaults: Critical recall = 23.4%
- With optimized thresholds: Critical recall = 65.0% (+178% improvement!)

**Solution**: Implemented smart category-specific defaults and created script to optimize all categories.

---

## Analysis: Is 0.5 Too High?

### Critical Category Optimized Thresholds

From 2025-11-04 optimization session (target recall: 65%):

| Category | Optimized Threshold | vs 0.5 Default | Multiplier |
|----------|---------------------|----------------|------------|
| hospitals | 0.014 (1.4%) | **35x lower** | 0.028x |
| security | 0.020 (2.0%) | **25x lower** | 0.040x |
| search_and_rescue | 0.033 (3.3%) | **15x lower** | 0.066x |
| medical_products | 0.095 (9.5%) | **5x lower** | 0.190x |
| medical_help | 0.124 (12.4%) | **4x lower** | 0.248x |
| shelter | 0.240 (24.0%) | **2x lower** | 0.480x |
| water | 0.362 (36.2%) | **1.4x lower** | 0.724x |
| food | 0.431 (43.1%) | **1.2x lower** | 0.862x |

### Key Insights

1. **Extreme imbalance**: Critical categories need thresholds 2-35x lower than 0.5
2. **Life-safety categories** (hospitals, security) need extremely low thresholds (~1-2%)
3. **Moderate categories** (food, water) can use higher thresholds (~35-43%)
4. **Non-critical categories**: 0.5 may be appropriate, but should be optimized too

### Performance Impact

**Before optimization** (0.5 defaults):
- F1-weighted: 93.7%
- Critical recall: 23.4% ❌ (missing 76.6% of emergencies!)

**After optimization** (category-specific thresholds):
- F1-weighted: 92.6% (-1.1% acceptable trade-off)
- Critical recall: 65.0% ✅ (+178% improvement!)

**Verdict**: 0.5 is **definitely too high** for critical categories. The optimized thresholds dramatically improve emergency detection with minimal F1 cost.

---

## Solutions Implemented

### 1. Smart Category-Specific Defaults ✅

**Updated**: `app/services.py` - `get_thresholds_map()`

Now uses optimized defaults for critical categories:
- Critical categories: Use optimized thresholds from 2025-11-04 session
- Non-critical categories: Use standard 0.5 threshold

**Benefits**:
- Production app works better even without threshold files
- Fallback behavior is now optimized
- No breaking changes (loaded thresholds still take priority)

### 2. All-Category Optimization Script ✅

**Created**: `scripts/optimize_all_thresholds.py`

Extends threshold optimization to ALL categories (not just critical):
- Critical categories: Target recall 65% (configurable)
- Non-critical categories: Target recall 60% (configurable)
- Saves complete threshold map for all 36 categories
- Includes category statistics and performance metrics

**Usage**:
```powershell
python scripts/optimize_all_thresholds.py \
  --model-path experiments/experimental_runs/2025-11-04/lr_baseline_model.pkl \
  --critical-recall 0.65 \
  --non-critical-recall 0.60
```

**Output**: `optimized_all_thresholds.json` with:
- All 36 category thresholds
- Category statistics (recall, precision, F1)
- Performance comparison (baseline vs optimized)

---

## Recommendations

### Immediate Actions

1. **✅ DONE**: Smart defaults implemented in production app
2. **Run optimization**: Execute `optimize_all_thresholds.py` on your production model
3. **Deploy thresholds**: Copy `optimized_all_thresholds.json` to production model directory
4. **Verify**: Check logs to confirm optimized thresholds are loaded

### Future Improvements

1. **Automatic optimization**: Add threshold optimization to model training pipeline
2. **A/B testing**: Test optimized thresholds vs defaults in production
3. **Dynamic thresholds**: Adjust thresholds based on real-world performance
4. **Category-specific targets**: Fine-tune target recall per category based on business needs

### Optimization Strategy

**For Critical Categories**:
- Use target recall: 60-70% (current: 65%)
- Accept lower precision for better recall (life-safety priority)
- Monitor false positive rates

**For Non-Critical Categories**:
- Use target recall: 55-65% (current: 60%)
- Balance precision and recall more evenly
- Can be more conservative

**For Rare Categories**:
- Categories with very few positives may need special handling
- Consider using F2 optimization instead of target recall
- May need to accept higher false positive rates

---

## Technical Details

### Why Are Thresholds So Low?

1. **Class imbalance**: Critical categories have very few positive examples
2. **Model calibration**: LogisticRegression probabilities are well-calibrated but conservative
3. **Recall priority**: For emergencies, missing a case is worse than false alarms
4. **Precision trade-off**: Lower thresholds increase false positives but catch more true emergencies

### Threshold Selection Method

**Current**: `precision_recall_curve` with target recall
- Finds threshold closest to target recall
- Balances precision and recall
- Works well for imbalanced data

**Alternative**: F2 score optimization
- Emphasizes recall (beta=2.0)
- May find different thresholds
- Used in experimental script for comparison

### Validation

All optimized thresholds are validated:
- ✅ F1 maintained ≥ 90%
- ✅ Critical recall improved significantly
- ✅ No data leakage (frozen eval set)
- ✅ Independent validation script confirms results

---

## Example: Threshold Impact

### Before (0.5 default):
```
Message: "Need medical help urgently"
Probability: 0.35 (35%)
Threshold: 0.5
Prediction: NOT medical_help ❌ (missed emergency!)
```

### After (0.124 optimized):
```
Message: "Need medical help urgently"
Probability: 0.35 (35%)
Threshold: 0.124
Prediction: medical_help ✅ (caught emergency!)
```

This is exactly the improvement we want - catching real emergencies that would be missed with 0.5 threshold.

---

## Files Modified

1. **`app/services.py`**:
   - Updated `get_thresholds_map()` with smart defaults
   - Critical categories now use optimized thresholds as fallback

2. **`scripts/optimize_all_thresholds.py`** (NEW):
   - Optimizes thresholds for all 36 categories
   - Configurable target recall per category type
   - Comprehensive statistics and validation

---

## Next Steps

1. **Run optimization**:
   ```powershell
   python scripts/optimize_all_thresholds.py --model-path model/<your_model>.pkl
   ```

2. **Review results**: Check `optimized_all_thresholds.json` for all category thresholds

3. **Deploy**: Copy optimized thresholds to production model directory

4. **Monitor**: Track performance in production to validate improvements

5. **Iterate**: Adjust target recall values if needed based on real-world feedback

---

**Status**: ✅ Smart defaults implemented, optimization script ready  
**Impact**: Critical recall improved from 23% to 65% with minimal F1 cost  
**Recommendation**: Run optimization script and deploy optimized thresholds

