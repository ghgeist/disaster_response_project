---
created: 2026-01-29
updated: 2026-01-29
status: active
---

# Storm Signal Dashboard — Implementation Plan

> **Goal**: Build a modern React-based intelligence dashboard ("Storm Signal") on top of the existing Flask ML backend.

> **Specs**: See `2026_01_29_storm_signal_dashboard_design_spec.md` for UI/UX requirements.

---

## Phase 1: Environment & Architecture Setup

**Objective**: Establish a "Hybrid" Flask + React environment where Flask serves the API and Vite builds the frontend.

### 1.1 Frontend Initialization
- [ ] Initialize `frontend/` directory in root
- [ ] Setup **Vite** + **React** + **TypeScript**
- [ ] Install **Tailwind CSS** & configure `tailwind.config.js` (Light theme palette)
- [ ] Install core dependencies:
  - `lucide-react` (Icons)
  - `recharts` (Charts/Graphs)
  - `framer-motion` (Animations)
  - `axios` (API requests)
  - `react-resizable-panels` (For the 3-panel layout)
  - `clsx` + `tailwind-merge` (Styling utils)

### 1.2 Flask Integration
- [ ] Configure Flask to serve React's `dist/index.html` as the entry point
- [ ] Configure Vite to proxy API requests to Flask (`localhost:5000`) during dev
- [ ] Update `run.py` to handle both API routes (`/api/*`) and frontend static serving

---

## Phase 2: Backend API Development

**Objective**: Expose data and ML capabilities via JSON endpoints.

### 2.1 API Structure
- [ ] Create `app/api/` blueprint
- [ ] Define standard response format: `{ "success": true, "data": ... }`

### 2.2 Endpoints Implementation
- [ ] **`GET /api/categories`**
  - Return all 36 categories + groups (Critical, Infrastructure, etc.)
  - Return "Volume Today" counts (real db query + simulated trend)
- [ ] **`GET /api/feed`**
  - Params: `limit`, `offset`, `filters`
  - Return: List of messages with simulated timestamps, severity, and tags
  - Logic: Fetch real rows, enrich with simulated metadata (severity, time)
- [ ] **`GET /api/metrics`**
  - Return: `{ vol_today, flagged_count, trend_history[] }`
  - Logic: Mix of real counts and simulated time-series data
- [ ] **`POST /api/classify`**
  - Input: `{ text: "..." }`
  - Logic: Run ML pipeline → Calculate Severity → Return Categories + Context (Volume)

---

## Phase 3: Frontend Core Components

**Objective**: Build the visual skeleton and layout.

### 3.1 Layout System
- [ ] **AppShell**: Main container with full height
- [ ] **Header**: Hamburger Menu (Left), Logo, User Profile (Right)
- [ ] **PanelSystem**:
  - Implement `react-resizable-panels`
  - Left (40%), Center (35%), Right (25%)
  - Persist layout state to localStorage

### 3.2 Shared UI Components
- [ ] `Badge` (Severity: High/Med/Low)
- [ ] `Card` (Metrics/Feed Items)
- [ ] `Button` (Primary/Action/Ghost)
- [ ] `DisclaimerPill` (The "Metrics Simulated" status)

---

## Phase 4: Feature Implementation

**Objective**: Connect UI to API and implement business logic.

### 4.1 Left Panel: Feed & Filters
- [ ] **FilterDrawer**: Collapsible section with checkbox groups
- [ ] **FeedList**: Virtualized list (for performance) of message cards
- [ ] **FeedItem**:
  - Message preview (truncated)
  - Severity badge logic
  - Translation indicator
  - Timestamp formatting

### 4.2 Center Panel: Metrics
- [ ] **MetricCards**: "Vol Today" and "Flagged"
- [ ] **TrendChart**: Recharts line graph (Red line for flagged signals)
- [ ] **TopCategories**: Simple list with progress bars/counts

### 4.3 Right Panel: Classification
- [ ] **InputForm**: Textarea + Submit
- [ ] **ResultsView**:
  - Category list with confidence bars
  - "Volume Context" display
  - "Dispatch Assistance" Action Button (w/ loading & success states)
- [ ] **EmptyState**: "Mark as Irrelevant" flow

---

## Phase 5: Polish & Deployment

**Objective**: Ensure professional quality and deployment readiness.

### 5.1 Refinement
- [ ] **Mobile Banner**: "Desktop Only" blocking overlay for small screens
- [ ] **Loading States**: Skeletons for Feed and Metrics
- [ ] **Error Handling**: Toasts for API failures

### 5.2 Deployment Prep
- [ ] Build React app to `dist/`
- [ ] Verify Flask serves `dist/` correctly
- [ ] Update `requirements.txt` (if any new Python deps)
- [ ] Update `README.md` with "How to Run" (npm install + python run.py)

---

## Execution Order

1.  **Scaffold**: Set up React/Vite project structure (Phase 1)
2.  **Backend**: Build the 4 API endpoints (Phase 2)
3.  **Layout**: Build the resizable 3-panel shell (Phase 3)
4.  **Feed**: Implement Feed + Filters (Phase 4.1)
5.  **Classify**: Implement Classification flow (Phase 4.3)
6.  **Metrics**: Implement Charts/Graphs (Phase 4.2)
