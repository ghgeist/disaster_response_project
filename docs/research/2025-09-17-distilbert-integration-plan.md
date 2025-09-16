---
title: "Research Plan: Integrating DistilBERT for Improved Classification"
date: "2025-09-17"
status: "Proposal v3"
tags: ['DistilBERT', 'transformers', 'engineering-plan', 'POC']
author: "Gemini"
related: ['docs/sessions/active/2025-09-16-hyperparameter-tuning-plan.md']
---

# Research Plan: Integrating DistilBERT (v3)

## 🎯 Objective

To execute a rigorous, reproducible proof-of-concept (POC) to determine if a DistilBERT-based model offers a quantifiable performance advantage over the existing RandomForest classifier for multi-label disaster message classification.

## 📋 Engineering Plan

This plan moves beyond a simple academic experiment to a detailed engineering POC with clear, reproducible steps and success criteria.

### Phase 1: Environment, Data & Analysis (2-3 hours)

1.  **Environment Specification**:
    *   **Software**: Standardize versions in `requirements.txt` (e.g., `torch==2.1`, `transformers==4.35`) to ensure reproducibility.
    *   **Hardware Strategy**: The POC will be developed for **CPU first**. If a single training epoch exceeds **60 minutes**, a GPU will be deemed a requirement for efficient iteration. All final inference speed benchmarks **must** be run on a CPU to simulate a standard production environment.

2.  **Data Splitting & Integrity**:
    *   The dataset will be split into three non-overlapping sets: **64% for Training**, **16% for Validation**, and the existing **20% Frozen Test Set**.

3.  **Advanced Data Processing**:
    *   **Cleaning Pipeline**: A specific, ordered cleaning function will be applied to all text before tokenization:
        1.  Remove HTML tags.
        2.  Normalize URLs to a `[URL]` token.
        3.  Normalize user mentions (`@...`) and retweets (`RT`) to `[USER]` and `[RETWEET]` tokens.
        4.  Consolidate excess whitespace.
    *   **Class Weight Calculation**: To combat imbalance, `pos_weight` for the loss function will be calculated for each of the 36 classes using the formula: `num_negative_samples / num_positive_samples`.

### Phase 2: Robust Training & Artifact Generation (4-6 hours)

1.  **Script Interface (`scripts/08_create_distilbert_model.py`)**:
    *   The script will be built with a flexible command-line interface using `argparse` to control:
        *   `--epochs`, `--batch-size`, `--learning-rate`
        *   `--output-dir` to specify the destination for the final model package.

2.  **Robust Training Loop**:
    *   **Weighted Loss**: Employ `BCEWithLogitsLoss` with the pre-calculated `pos_weight` tensor.
    *   **Stability**: Use `torch.nn.utils.clip_grad_norm_` to prevent gradient explosion.
    *   **Early Stopping**: Monitor the validation set's weighted F1-score. Halt training with `patience=2` if the score does not improve for two consecutive epochs.

3.  **Model Package Definition**:
    *   Upon completion, the script will generate a self-contained **"model package"** in the specified `--output-dir`. This directory will contain:
        1.  The core model and tokenizer files (from `save_pretrained`).
        2.  A `metadata.json` file containing the git hash of the training code, final performance scores, and the path to the data used.
        3.  An `optimal_thresholds.json` file mapping each of the 36 labels to its ideal decision threshold.

### Phase 3: Rigorous Evaluation & Success Criteria (2-3 hours)

1.  **Threshold Tuning Algorithm**:
    *   After training, the best model is used to generate predictions on the **validation set**. For each label, we will iterate through 100 threshold values (0.01 to 1.0) to find the value that maximizes its individual F1-score. The results are saved to `optimal_thresholds.json`.

2.  **Success Criteria**:
    *   The POC will be deemed successful **only if both** of the following conditions are met on the **frozen test set**:
        1.  The **overall weighted F1-score** shows a relative improvement of at least 5% over the best RandomForest model.
        2.  The average F1-score across a defined set of **critical labels** (`medical_help`, `search_and_rescue`, `water`, `food`, `shelter`) **does not decrease**.

3.  **Disagreement Analysis**:
    *   Instead of random sampling, a qualitative analysis will focus on the **disagreement set**: cases where the RandomForest and DistilBERT models made different predictions. This provides targeted insight into the new model's strengths and weaknesses.

### Phase 4: Path to Production (Post-POC)

If the POC is successful, a separate implementation phase will be planned, including:

1.  **Code Refactoring**: Abstracting the POC script's logic into reusable components within the `src/disasterproject` package.
2.  **Hyperparameter Search**: Conducting a full search for optimal hyperparameters using a library like Optuna.
3.  **Flask Integration**: Building a clean `BertClassifier` service class to handle all prediction logic, abstracting its complexity from the web application layer.
4.  **Deployment**: Updating the `Dockerfile` and deployment documentation to handle the new dependencies and larger model artifacts.

---

This updated plan provides a much clearer, more rigorous, and engineering-focused path forward. It defines not just *what* to do, but *how* to do it reproducibly and what specific criteria will define success.