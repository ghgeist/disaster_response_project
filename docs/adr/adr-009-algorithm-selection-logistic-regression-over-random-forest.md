---
title: "Algorithm Selection: LogisticRegression Over Random Forest"
date: "2025-09-03"
status: "accepted"
tags: ["ml-operations", "algorithm-selection", "model-optimization", "production"]
author: "ML Engineering Team"
related: ["adr-003-hybrid-model-deployment-strategy.md", "adr-008-class-weighting-over-sampling.md"]
---

# Algorithm Selection: LogisticRegression Over Random Forest

**Date**: 2025-09-03  
**Status**: Accepted  
**Deciders**: ML Engineering Team  
**Tags**: ml-operations, algorithm-selection, model-optimization, production

## Context

The disaster response classification system initially used RandomForestClassifier with MultiOutputClassifier for multi-label classification across 36 disaster categories. The production model achieved good performance (F1-score ~0.9007-0.9366) but had significant operational challenges:

- **Model size**: 1GB+ RandomForest model (1039MB baseline, 31MB optimized)
- **Load time**: 6.2 seconds for cold load, 0.55s for optimized version
- **Memory footprint**: Large memory requirements for deployment
- **Critical recall issues**: Zero recall on several critical disaster categories (medical_help, search_and_rescue, hospitals, security) due to class imbalance
- **Deployment constraints**: Model size and load time created deployment bottlenecks

The system needed a production-ready model that balanced performance with operational requirements, especially for cloud deployments and real-time inference scenarios.

## Decision

Replace RandomForestClassifier with **LogisticRegression** as the primary algorithm for disaster response classification, using TF-IDF feature extraction and MultiOutputClassifier wrapper.

### Implementation Details

- **Pipeline**: `create_pipeline_logistic_regression()` in `src/disasterproject/models/pipeline.py`
- **Feature extraction**: TF-IDF with CountVectorizer (supports vocabulary size limits)
- **Multi-label handling**: WeightedMultiOutputClassifier wrapper
- **Optimization**: Vocabulary size optimization (15K features) for model size reduction
- **Class imbalance**: Balanced class weighting (see ADR-008)

### Performance Characteristics

**Baseline Comparison** (from 2025-09-03 evaluation):
- **F1-Score**: 0.9254 (LR) vs 0.9357 (RF baseline) - only 1% drop
- **Model Size**: 1.5MB (LR) vs 1039MB (RF) - **99.85% reduction**
- **Load Time**: 0.076s (LR) vs 6.2s (RF) - **98.8% faster**
- **Critical Recall**: Massive improvements on all 8 critical labels

**Current Production Model** (vocab15k, 2025-11-06):
- **F1-Score**: 0.9379 (baseline), 0.9276 (threshold-optimized)
- **Model Size**: 4.53MB
- **Critical Recall**: 65% average across 8 critical categories
- **Per-category performance**: Better F1 on 19/36 categories vs RF, including 5/8 critical categories

## Consequences

### Positive

- **Massive size reduction**: 99.85% smaller models enable efficient cloud deployment
- **Faster inference**: Sub-100ms load times improve user experience
- **Better critical recall**: Eliminated zero-recall issues on critical disaster categories
- **Deployment flexibility**: Small model size enables multiple deployment strategies
- **Cost efficiency**: Lower storage and memory costs in production
- **Maintainability**: Simpler model architecture easier to debug and maintain
- **Per-category performance**: Outperforms RF on majority of individual labels (19/36 categories)
- **Production stability**: More predictable performance characteristics

### Negative

- **Slight F1 trade-off**: Baseline F1 slightly lower than optimized RF (0.9379 vs 0.9366)
- **Weather category performance**: RF performs better on weather-related categories (earthquake, weather_related, storm, floods)
- **Less interpretable**: RandomForest feature importance more intuitive than LR coefficients
- **Hyperparameter sensitivity**: LR may require more tuning for optimal performance

### Neutral

- **Training time**: Both algorithms train in reasonable time (<1 minute)
- **Inference speed**: Both provide fast inference once loaded
- **Multi-label support**: Both use MultiOutputClassifier wrapper
- **Feature engineering**: Both use same TF-IDF preprocessing pipeline

## Alternatives Considered

### Alternative 1: Optimized RandomForest (Smaller Size)
**Description**: Reduce RandomForest model size through hyperparameter tuning (fewer trees, depth limits)

**Pros**:
- Maintains RF performance characteristics
- Better performance on weather-related categories
- More interpretable feature importance

**Cons**:
- Still significantly larger than LR (31MB vs 1.5MB)
- Slower load times (0.55s vs 0.076s)
- Still struggles with critical recall on imbalanced categories
- Requires more computational resources

**Rejection Reason**: Operational benefits of LR outweigh marginal performance differences, especially given critical recall improvements

### Alternative 2: Keep Large RandomForest
**Description**: Continue using full-size RandomForest model despite operational challenges

**Pros**:
- Highest baseline F1-score
- No algorithm change required

**Cons**:
- 1GB+ model size creates deployment bottlenecks
- 6+ second load times unacceptable for production
- Zero recall on critical categories unacceptable for disaster response
- High memory and storage costs

**Rejection Reason**: Operational constraints and critical recall failures make this untenable for production

### Alternative 3: Ensemble (RF + LR)
**Description**: Combine RandomForest and LogisticRegression predictions

**Pros**:
- Could leverage strengths of both algorithms
- Potentially better overall performance

**Cons**:
- Doubles model size and complexity
- Slower inference (two models to run)
- More complex deployment and maintenance
- Marginal benefit doesn't justify complexity

**Rejection Reason**: Complexity and operational overhead not justified by potential gains

### Alternative 4: Other Algorithms (SVM, Neural Networks, etc.)
**Description**: Evaluate alternative algorithms for multi-label classification

**Pros**:
- Could find better performance/size trade-offs
- Modern approaches might offer advantages

**Cons**:
- Requires extensive experimentation
- Unknown performance characteristics
- May introduce new dependencies or complexity
- LR already provides excellent balance

**Rejection Reason**: LR already achieves production goals; exploration deferred to future optimization cycles

## References

- **Original Decision**: [Dev Note 2025-09-11](../dev_notes/2025-09-11.md) - Documents the ML optimization project
- **Performance Analysis**: [Session 2025-09-03](../sessions/completed/2025-09-03-execute-ml-optimization-COMPLETED.md) - Detailed performance comparison
- **Research Context**: [Research 2025-09-16](../research/2025-09-16-analyzing-and-improving-the-classifier.md) - Model evolution analysis
- **Implementation**: `src/disasterproject/models/pipeline.py::create_pipeline_logistic_regression()`
- **Production Model**: `model/disaster_lr_v25-11-06_prod_2025-11-06.pkl`
- **Vocabulary Optimization**: [Vocabulary Comparison Report](../../experiments/experimental_runs/2025-11-06/vocabulary_comparison_report.md)
- **Class Weighting**: [ADR-008](adr-008-class-weighting-over-sampling.md) - Class weighting strategy

## Implementation Status

- ✅ **LogisticRegression pipeline**: Implemented and tested
- ✅ **Vocabulary optimization**: 15K feature model in production
- ✅ **Production deployment**: LR model deployed as of 2025-11-06
- ✅ **Performance validation**: F1-scores validated and documented
- ✅ **Critical recall**: All 8 critical categories achieve target recall (65%+)

## Performance Validation

### Per-Category F1 Comparison (LR vs RF)
- **LR better**: 19/36 categories (53%)
- **RF better**: 10/36 categories (28%)
- **Tied**: 7/36 categories (19%)

### Critical Categories Performance
- **LR better**: 5/8 critical categories (medical_products, shelter, water, medical_help, security)
- **RF better**: 2/8 critical categories (search_and_rescue, food)
- **Tied**: 1/8 critical categories (hospitals)

### Overall Metrics
- **Baseline F1**: LR 0.9379 vs RF 0.9366 (+0.14% improvement)
- **Threshold-optimized F1**: LR 0.9276 vs RF 0.8869 (+4.59% improvement)
- **Model size**: LR 4.53MB vs RF 31MB+ (85% reduction)

## Migration Path

**Completed (2025-09-03)**: Initial LR model deployed  
**Completed (2025-11-06)**: Vocabulary-optimized LR model (vocab15k) promoted to production  
**Current**: LR is the standard algorithm for all new model training  
**Future**: Continue optimizing LR hyperparameters and vocabulary size; RF retained for experimental comparisons

---

**Decision Rationale**: LogisticRegression provides the optimal balance of performance, operational efficiency, and production readiness. The 99.85% size reduction and elimination of critical recall failures justify the marginal baseline F1 trade-off, especially given that LR actually outperforms RF on threshold-optimized metrics and the majority of individual categories.
