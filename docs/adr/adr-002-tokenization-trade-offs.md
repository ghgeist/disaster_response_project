---
title: "Use Domain-Aware Stopword Filtering for Disaster Classification"
date: "2025-01-14"
status: "accepted"
tags: ["nlp", "preprocessing", "disaster-classification", "portfolio"]
author: "Portfolio Developer"
related: []
---

# Use Domain-Aware Stopword Filtering for Disaster Classification

**Date**: 2025-01-14  
**Status**: Accepted  
**Deciders**: Portfolio Developer  
**Tags**: nlp, preprocessing, disaster-classification, machine-learning

## Context

Our disaster tweet classification model is failing to identify critical personal distress messages like "Help me!" due to aggressive text preprocessing. Root cause analysis revealed that standard stopword removal is destroying essential signal:

- **Input**: "Help me!" 
- **Current preprocessing**: `['Help', 'me']` → `['help']` (removes "me" as stopword)
- **Result**: Model only sees generic "help" and misses personal urgency context

This systematic preprocessing issue affects many disaster messages:
- "Save us!" → `['save']` (loses "us")  
- "We need help!" → `['need', 'help']` (loses "we")
- "I'm trapped!" → `['trapped']` (loses "I'm")

Personal pronouns are critical disaster signals indicating individual/group distress, but our NLTK stopword list removes them as "noise."

## Decision

Implement disaster-aware stopword filtering that preserves domain-critical words while maintaining noise reduction benefits.

**Technical Implementation**:
```python
# Preserve disaster-critical personal pronouns and action words
disaster_critical = {'me', 'us', 'we', 'i', 'my', 'our', 'help', 'please', 'save', 'rescue'}

# Modified filtering logic
tokens = [token for token in tokens 
         if token.lower() not in STOPWORDS_SET or token.lower() in disaster_critical]
```

This approach maintains existing bag-of-words + traditional ML architecture while fixing the core preprocessing issue.

## Consequences

### Positive
- **Fast to ship**: 10-minute code change vs weeks for new architecture
- **Zero cost**: Uses existing infrastructure, no API fees
- **Immediate improvement**: "Help me!" becomes `['help', 'me']` instead of `['help']`
- **Clear portfolio story**: Demonstrates systematic problem diagnosis and domain expertise
- **Maintainable**: Easy to debug and understand preprocessing logic
- **Preserves signal**: Personal pronouns retained for distress detection

### Negative
- **Manual feature engineering**: Requires domain knowledge to identify critical words
- **Limited context understanding**: Still treats words independently (no "Help me!" vs "Help wanted" distinction)
- **Maintenance overhead**: May need updates for new disaster-critical terms
- **Potential false positives**: Could misclassify "Help me with homework"

### Neutral
- **Performance ceiling**: Good improvement but not state-of-the-art accuracy
- **Technical debt**: May need eventual upgrade to transformer-based approach

## Alternatives Considered

### 1. BERT/Transformer Model (Local Implementation)
- **Pros**: Superior context understanding, handles nuanced cases, modern approach
- **Cons**: Weeks of implementation time, requires retraining pipeline, GPU infrastructure
- **Why rejected**: Portfolio timeline constraints, need to ship quickly

### 2. BERT API (Hugging Face, OpenAI)
- **Pros**: Fast implementation, state-of-the-art accuracy, minimal infrastructure
- **Cons**: Ongoing API costs for portfolio project, external dependency
- **Why rejected**: Budget constraints for demonstration project

### 3. Minimal Preprocessing (Remove Stopword Filtering Entirely)
- **Pros**: Preserves all signal, simple implementation
- **Cons**: Increases noise, may hurt performance on non-disaster classification
- **Why rejected**: Doesn't optimize for disaster domain, loses preprocessing benefits

### 4. Custom Hard-coded Word Lists
- **Pros**: Domain-specific, interpretable rules
- **Cons**: Maintenance nightmare, brittle to language evolution, not scalable
- **Why rejected**: Industry research shows these approaches are problematic

## References

- [USC Research Paper: "On Identifying Disaster-Related Tweets: Matching-based or Learning-based?"](https://arxiv.org/pdf/1705.02009) - Shows matching-based approaches often outperform learning-based for disaster classification
- [Modern Disaster Classification Research](https://pmc.ncbi.nlm.nih.gov/articles/PMC10909225/) - Recent work using BERT shows minimal preprocessing trends
- Internal analysis: "Help me!" tokenization debugging session
- Kaggle NLP Disaster Tweets Competition - Standard preprocessing approaches and limitations