---
created: 2026-01-22
updated: 2026-01-22
---
# Storm Signal — System Scope & Demo Contract

> **Related Document:** See `2025_11_storm_signal_execution.md` for execution plan, completion status, and portfolio strategy.

> **Role of this document**: This is a **system specification and map**, not the territory. It defines boundaries, invariants, and interfaces. Many components are intentionally underspecified and will be filled in later.

---

## 1. Purpose

Storm Signal is an **OSINT-oriented signal detection, triage, and dispatch-support system** designed to surface **low-frequency, high-consequence signals** from noisy social data.

The system is optimized for **attention allocation**, not certainty. Its primary job is to ensure that *potentially critical signals are seen*, even at the cost of increased false positives.

This specification captures the **conceptual architecture and demo constraints** of the project, not a production-ready implementation.

---

## 2. Core Product Thesis

**Classification → Context → Action**

* **Classification** identifies candidate signals.
* **Context** makes those signals interpretable and triageable.
* **Action** enables downstream human or agent intervention.

Classification alone is insufficient for operational use. The product value emerges when individual classifications are placed in **situational context** relative to other signals, time, categories, and uncertainty.

The demo intentionally emphasizes **classification + context**, with action affordances kept minimal.

---

## 3. Demo Scope & Reality Boundary

### 3.1 What is Real in the Demo

* **Single-item classification** of an individual tweet/post
* Structured classification output (categories, confidence, thresholds)
* Deterministic, inspectable behavior at the classification boundary

### 3.2 What is Simulated or Stubbed

* High-volume ingestion pipelines
* Real-time streaming from external platforms
* Large-scale historical databases
* Advanced clustering, geospatial inference, and trend detection
* Full dispatch and response workflows

UI dashboards and system metrics may be **simulated using distributions derived from the model training dataset**, in order to present realistic volumes, category mixes, and temporal patterns without implying live operation.

> **Invariant**: Classification is real. Volume, scale, and integration are illustrative.

---

## 4. System Bias & Risk Posture

Storm Signal is intentionally biased toward **high recall**:

* False negatives (missed critical signals) are considered more costly than false positives.
* The system is expected to surface *candidates*, not conclusions.
* Downstream triage (human or agent) is assumed.

This bias is explicit, configurable, and visible to operators.

---

## 5. System Components (Conceptual)

### 5.1 Input Layer (Conceptual)

* Social posts (e.g., tweets)
* Metadata where available (timestamp, author, language hints)

### 5.2 Normalization Layer (Conceptual)

* Language detection
* Preservation of original text
* Optional translation
* Tracking of original vs translated content

### 5.3 Classification Layer (Real in Demo)

* Multi-label categorization against a predefined taxonomy
* Per-category confidence scores
* Threshold-based flagging
* Versioned model boundary

### 5.4 Contextualization Layer (Partially Simulated)

For a given classified item, the system surfaces:

* Similar or related items (heuristic or stubbed)
* Temporal neighbors
* Category prevalence indicators
* Indicators of rarity vs commonality

Context is presented to **support decision-making**, not to assert ground truth.

### 5.5 Action Layer (Thin by Design)

* Reclassification / override
* Annotation
* Assignment or escalation markers

Actions are logged but not fully operationalized in the demo.

---

## 6. UI Philosophy

The UI is designed as **decision scaffolding**, not automation.

Principles:

* Analytics-first, minimal visual styling
* Light theme only
* Emphasis on uncertainty and partial information
* Fast dismissal and review of false positives

The primary UI surfaces are:

1. **Command & Control Overview**

   * System health
   * Volume and category distributions
   * Language and translation mix

2. **Item Detail & Context View**

   * Original and translated text
   * Classification output
   * Contextual neighbors
   * Action affordances

---

## 7. API Boundary (Conceptual)

The classification system is treated as an independent service boundary.

Expected capabilities:

* Programmatic classification of individual items
* Structured outputs suitable for AI agents
* Explicit versioning
* Feedback hooks for future learning loops

The demo does not implement a full external API, but the interface is treated as first-class in design.

---

## 8. Non-Goals (Explicit)

The following are explicitly out of scope for the current project phase:

* Production-scale ingestion and throughput
* End-to-end disaster response workflows
* Guaranteed geolocation accuracy
* Automated decision-making
* UI polish, theming, or dark mode

These exclusions are intentional to preserve narrative clarity and system legibility.

---

## 9. Future Elaboration (Deferred)

Details intentionally deferred:

* Taxonomy evolution strategy
* Model retraining and evaluation pipelines
* Advanced context retrieval methods
* Multi-agent orchestration

These will be specified only once the current system boundaries are stable.

---

## 10. Guiding Principle

Storm Signal exists to **shape attention under uncertainty**.

Any future expansion should be evaluated against the question:

> *Does this help an operator or agent notice the right thing at the right time?*
