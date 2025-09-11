1. Week 1: Raise minority-class recall (defer unless needed)

Add bigrams and cap vocab

Change: ngram_range=(1,2), min_df=3, max_features=40_000.

Acceptance: tail-label macro recall +2–5 pts; model size ≤ 200 MB.

Why: phrases like “search and rescue”, “medical help” become learnable features.

Multilabel-aware split

Change: iterative stratification for frozen eval so every label appears in train. If adding a lib is heavy, implement a simple guarantee: if any label has <5 in train, move a few from eval to train and log it.

Acceptance: report per-label train/test supports; none with 0 in train.

Try a lighter linear baseline that often beats RF on text

Change: TF-IDF + One-vs-Rest LogisticRegression (or LinearSVM with calibrated probs).

Acceptance: macro recall ≥ RF, file size < 50 MB, load < 1 s.

Classifier Chains pilot

Change: train one chain with LogisticRegression to capture label dependencies.

Acceptance: subset accuracy and multi-label coverage improve; no unacceptable precision regression on critical labels.

Targeted oversampling where feasible

Change: one-vs-rest loop with ADASYN/SMOTE only for labels with 20–200 positives; skip ultra-rare labels. Oversample inside each binary pipeline.

Acceptance: tail-label recall increases; training < 2× baseline time.

2. Week 2: Production hardening (defer unless you keep investing)

Package the project to remove path hacks

Change: add pyproject.toml and install disaster_classifier as a package. Remove sys.path edits and __main__.tokenize shims.

Acceptance: clean import graph; joblib.load works without path manipulation.

Health and metrics upgrades

Change: add /metrics JSON with per-label precision, recall, thresholds, model version; add a health subcheck that runs a 1-msg dummy prediction.

Acceptance: dashboards can scrape /health and /metrics; alerts fire on unhealthy.

Non-English fallback

Change: detect non-ASCII or low English token hit-rate; show a UI nudge “Paste in English” and log a hint. If feasible, add a lightweight translation step before vectorization.

Acceptance: Creole or Turkish examples no longer silently return all zeros without a user hint.

Unify experiments tracking

Change: retire “flat” results; standardize on experiments/<slug>/{config,model,results,log}. Add latest.json mapping experiment → paths.

Acceptance: single source of truth for comparisons; no duplicates.

Document decision rules

Change: write an ADR summarizing thresholds, the related consistency rule, and when to choose LR vs RF.

Acceptance: newcomers can reproduce the production model in one command.