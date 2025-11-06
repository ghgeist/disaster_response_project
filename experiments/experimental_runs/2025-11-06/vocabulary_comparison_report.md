# Vocabulary Size Optimization Comparison Report

**Generated**: 2025-11-06 16:01:31

## Executive Summary

This report compares 6 vocabulary-optimized models against the baseline unlimited vocabulary model.

## Overall Performance Comparison

| Model | Vocabulary Config | Model Size | Size Reduction | Baseline F1 | Optimized F1 | F1 Change | Critical Recall | Training Time |
|-------|------------------|------------|----------------|-------------|--------------|-----------|-----------------|---------------|
| Baseline (Unlimited) | Unlimited (230K features) | 67.69 MB | 0.0% | 0.9370 | 0.9264 | +0.0000 | 0.6497 | 50.3s |
| Baseline Filters | Filters only (min_df=2, max_df=0.95) | 13.03 MB | 80.7% | 0.9379 | 0.9277 | +0.0013 | 0.6497 | 24.7s |
| 30K Features | 30K features + filters | 9.08 MB | 86.6% | 0.9379 | 0.9276 | +0.0012 | 0.6497 | 24.7s |
| 25K Features | 25K features + filters | 7.56 MB | 88.8% | 0.9379 | 0.9277 | +0.0013 | 0.6497 | 24.7s |
| 20K Features | 20K features + filters | 6.04 MB | 91.1% | 0.9379 | 0.9275 | +0.0011 | 0.6497 | 24.7s |
| 15K Features | 15K features + aggressive filters | 4.53 MB | 93.3% | 0.9379 | 0.9276 | +0.0012 | 0.6497 | 24.7s |

## Critical Category Performance

### Critical Categories (with Optimized Thresholds)

| Category | Baseline (Unlimited) | Baseline Filters | 30K Features | 25K Features | 20K Features | 15K Features |
|----------|----------|----------|----------|----------|----------|----------|
| **food** | 65.1% | 65.1% | 65.1% | 65.1% | 65.1% | 65.1% |
| **hospitals** | 64.2% | 64.2% | 64.2% | 64.2% | 64.2% | 64.2% |
| **medical_help** | 65.0% | 65.0% | 65.0% | 65.0% | 65.0% | 65.0% |
| **medical_products** | 64.8% | 64.8% | 64.8% | 64.8% | 64.8% | 64.8% |
| **search_and_rescue** | 65.2% | 65.2% | 65.2% | 65.2% | 65.2% | 65.2% |
| **security** | 65.3% | 65.3% | 65.3% | 65.3% | 65.3% | 65.3% |
| **shelter** | 65.0% | 65.0% | 65.0% | 65.0% | 65.0% | 65.0% |
| **water** | 65.1% | 65.1% | 65.1% | 65.1% | 65.1% | 65.1% |

## Recommendations

### Best Model: **15K Features**

- **Model Size**: 4.53 MB (93.3% reduction)
- **F1-Weighted**: 0.9276
- **Critical Recall**: 0.6497
- **Training Time**: 24.7s

**Model Path**: `experiments/experimental_runs/2025-11-06/lr_vocab15k_model.pkl`
**Thresholds Path**: `experiments/experimental_runs/2025-11-06/vocab15k/optimized_critical_thresholds.json`

### All Valid Models

The following models meet production gates (F1 ≥ 92.0%, Critical Recall ≥ 64%):

- **15K Features**: 4.53 MB (93.3% reduction), F1=0.9276, Critical Recall=0.6497
- **20K Features**: 6.04 MB (91.1% reduction), F1=0.9275, Critical Recall=0.6497
- **25K Features**: 7.56 MB (88.8% reduction), F1=0.9277, Critical Recall=0.6497
- **30K Features**: 9.08 MB (86.6% reduction), F1=0.9276, Critical Recall=0.6497
- **Baseline Filters**: 13.03 MB (80.7% reduction), F1=0.9277, Critical Recall=0.6497
- **Baseline (Unlimited)**: 67.69 MB (0.0% reduction), F1=0.9264, Critical Recall=0.6497

## Vocabulary Parameters

| Model | max_features | min_df | max_df |
|-------|--------------|--------|--------|
| Baseline (Unlimited) | None | 1 | 1.0 |
| Baseline Filters | None | 2 | 0.95 |
| 30K Features | 30000 | 2 | 0.95 |
| 25K Features | 25000 | 2 | 0.95 |
| 20K Features | 20000 | 2 | 0.95 |
| 15K Features | 15000 | 3 | 0.9 |