---
created: 2026-01-22
updated: 2026-02-03
status: production
---
# Storm Signal — System Scope & Demo Contract

> **Related Documents:** 
> - See `2026_01_29_storm_signal_dashboard_design_spec.md` for detailed UI/UX specifications
> - See `2026_01_29_storm_signal_implementation_plan.md` for implementation phases and tasks

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

### 5.1 Architecture Stack

**Frontend**: React 18+ (Single Page Application)
* **Framework**: React with TypeScript
* **Styling**: Tailwind CSS (light theme)
* **Build Tool**: Vite (integrated with Flask)
* **State Management**: React Context or Zustand
* **Charts**: Recharts for data visualization

**Backend**: Flask (API Mode)
* **API Endpoints**: JSON endpoints for data and classification
* **Static Files**: Serves compiled React assets
* **ML Pipeline**: scikit-learn based classification (RandomForest with MultiOutputClassifier)

### 5.2 Input Layer (Conceptual)

* Social posts (e.g., tweets) from database
* Metadata: timestamp (simulated), genre (source), original text (for translation detection)
* Real messages from `data/02_stg/stg_disaster_response.db`

### 5.3 Normalization Layer (Conceptual)

* Language detection (currently English-focused)
* Preservation of original text
* Translation tracking (detect when `original` column differs from `message`)

### 5.4 Classification Layer (Real in Demo)

* Multi-label categorization against 36 predefined categories
* Per-category confidence scores (probabilities)
* Severity calculation (HIGH/MEDIUM/LOW based on critical categories and confidence)
* Versioned model boundary
* Real-time classification via `/api/classify` endpoint

### 5.5 Contextualization Layer (Partially Simulated)

For a given classified item, the system surfaces:

* **Category Context**: Volume counts for detected categories (real data from database)
* **Temporal Context**: Simulated timestamps spread over last 6 hours
* **Trend Context**: Simulated time-series data for flagged signals
* **Prevalence Indicators**: Real category counts from database

Context is presented to **support decision-making**, not to assert ground truth.

### 5.6 Action Layer (Thin by Design)

* **Dispatch Assistance**: Simulated operational handoff (logs action, shows success state)
* **Mark as Irrelevant**: For empty classification results
* Actions are logged but not fully operationalized in the demo

---

## 6. UI Philosophy & Layout

The UI is designed as **decision scaffolding**, not automation.

Principles:

* Analytics-first, minimal visual styling
* Light theme only (clean, professional aesthetic)
* Emphasis on uncertainty and partial information
* Fast dismissal and review of false positives
* Desktop-only experience (mobile shows banner message)

### 6.1 Layout Structure

**Three-Panel Desktop Layout** (resizable split-panes):

1. **Left Panel (Feed & Filters)** - Default 40% width
   * Live feed of classified messages
   * Collapsible category filters
   * Real messages with simulated timestamps
   * Severity badges, category tags, confidence scores

2. **Center Panel (Metrics & Trends)** - Default 35% width
   * Volume metrics (simulated)
   * Flagged signals trend graph (simulated)
   * Top categories list (real data)
   * "METRICS SIMULATED" disclaimer badge

3. **Right Panel (Classification Interface)** - Default 25% width
   * Message input form
   * Classification results with category context
   * Severity indicator
   * "Dispatch Assistance" action button

**Header**: Hamburger menu, logo, title, user profile controls

---

## 7. API Boundary

The classification system exposes JSON endpoints via Flask:

**Implemented Endpoints**:

* **`GET /api/feed`**: Paginated list of messages with classifications, filters, simulated timestamps
* **`GET /api/metrics`**: Dashboard metrics (volume, flagged percentage, trend data, top categories)
* **`POST /api/classify`**: Real-time classification of individual messages with severity calculation
* **`GET /api/categories`**: Category metadata (names, groups, volume counts)

**Response Format**: Standard JSON with `{ "success": true, "data": ... }` structure

The API treats classification as a first-class service boundary, suitable for both human operators and AI agents.

---

## 8. Non-Goals (Explicit)

The following are explicitly out of scope for the current project phase:

* Production-scale ingestion and throughput
* End-to-end disaster response workflows
* Guaranteed geolocation accuracy
* Automated decision-making
* Mobile/responsive design (desktop-only)
* Dark mode (light theme only)
* Real-time streaming from external platforms

These exclusions are intentional to preserve narrative clarity and system legibility. The demo focuses on **classification + context**, not operational scale.

---

## 9. Data Sources & Simulation Strategy

### 9.1 Real Data (from database)

* **Messages**: Actual `message` text from `stg_disaster_response.db`
* **Categories**: Real binary labels for all 36 categories
* **Genre**: Source mapping (`direct`, `news`, `social`)
* **Original**: Translation detection when `original` differs from `message`
* **Category Counts**: Real volume statistics from database queries

### 9.2 Simulated Data (for demo realism)

* **Timestamps**: Generated over last 6 hours (most recent first)
* **Volume Metrics**: Simulated "Volume Today" counts (can be based on database size)
* **Trend Graphs**: Simulated time-series data for flagged signals
* **Source Platforms**: Map `genre` to platform name `X` for social sources

> **Invariant**: Classification is real. Volume, scale, and temporal context are illustrative.

## 10. Future Elaboration (Deferred)

Details intentionally deferred:

* Taxonomy evolution strategy
* Model retraining and evaluation pipelines
* Advanced context retrieval methods (similarity search, clustering)
* Multi-agent orchestration
* Real-time streaming integration

These will be specified only once the current system boundaries are stable.

---

## 11. Guiding Principle

Storm Signal exists to **shape attention under uncertainty**.

Any future expansion should be evaluated against the question:

> *Does this help an operator or agent notice the right thing at the right time?*

---

## 12. Implementation Status

**Current Phase**: ✅ **PRODUCTION** - Dashboard deployed and live

**Key Deliverables**:
- ✅ System architecture defined
- ✅ Dashboard design specification complete
- ✅ Implementation plan with phased approach
- ✅ React + Vite frontend setup
- ✅ Flask API endpoints (all operational)
- ✅ Three-panel UI implementation
- ✅ Production deployment complete

**Production Routes**:
- `/dashboard` - Storm Signal Dashboard (live feed, metrics, classification)
- `/production-model` - Model Information Dashboard
- `/about` - About page

**API Endpoints** (all live):
- `GET /api/feed` - Paginated message feed with classifications
- `GET /api/metrics` - Dashboard metrics and trends
- `GET /api/categories` - Category metadata and counts
- `POST /api/classify` - Real-time message classification

See `2026_01_29_storm_signal_implementation_plan.md` for detailed implementation history.
