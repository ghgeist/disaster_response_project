---
created: 2026-01-29
updated: 2026-01-29
status: superseded
type: analysis
superseded_by: 2026_01_29_storm_signal_dashboard_implementation_plan.md
related: 
  - 2026_01_29_storm_signal_dashboard_design_spec.md
  - 2026_01_29_storm_signal_system_spec.md
---

> **⚠️ This document has been superseded.**  
> **Please see**: `2026_01_29_storm_signal_dashboard_implementation_plan.md` for the current implementation plan.

# Figma Code Analysis & Integration Plan

> **Purpose**: Analyze the Figma-generated React code against the design specification and document integration requirements for the Storm Signal Dashboard.

---

## Executive Summary

The Figma code provides a **complete React SPA structure** with:
- ✅ Three-panel resizable layout (Feed, Metrics, Classification)
- ✅ Mock data structures matching design spec
- ✅ UI components (shadcn/ui style)
- ✅ Tailwind CSS styling
- ⚠️ Mock data that needs real database integration
- ⚠️ No backend API integration yet

**Integration Status**: ~70% complete on frontend, 0% on backend integration.

**Simplified Approach**: Use inline utilities instead of separate files for simple operations (category mapping, timestamps, source mapping). This reduces complexity and keeps code maintainable.

**Desktop Design**: Target 1280px minimum width, 1440px recommended, 1920px maximum. Panels are resizable with sensible min/max constraints.

---

## 1. Component Comparison: Figma Code vs Design Spec

### ✅ Implemented Components

| Component | Figma Status | Design Spec Status | Notes |
|-----------|--------------|-------------------|-------|
| **Header** | ✅ Complete | ✅ Required | Hamburger menu, logo, title, status badges, user profile |
| **Resizable Panels** | ✅ Complete | ✅ Required | Uses `react-resizable-panels` library |
| **Feed Panel** | ✅ Complete | ✅ Required | Filter drawer, feed items, category tags |
| **Metrics Panel** | ✅ Complete | ✅ Required | KPI cards, trend chart, top categories |
| **Classification Panel** | ✅ Complete | ✅ Required | Input form, results display, dispatch button |
| **Mobile Warning** | ✅ Complete | ✅ Required | Dismissible banner for mobile users |
| **Category Groups** | ✅ Complete | ✅ Required | Critical Needs, Infrastructure, Weather, Other |

### ⚠️ Partial Implementation

| Component | Figma Status | Design Spec Status | Gap |
|-----------|--------------|-------------------|-----|
| **Severity Calculation** | ✅ Logic exists | ✅ Required | Uses mock data; needs real probability mapping |
| **Category Mapping** | ⚠️ Display names only | ✅ Required | Needs mapping from internal names (snake_case) to display names |
| **Translation Detection** | ✅ UI support | ✅ Required | UI shows globe icon, but needs `original` column check |
| **Source Mapping** | ⚠️ Hardcoded | ✅ Required | Needs mapping from `genre` column (direct/news/social) |

### ❌ Missing Components

| Component | Design Spec Status | Notes |
|-----------|-------------------|-------|
| **API Integration** | ❌ Not started | No Flask API endpoints connected |
| **Real Data Loading** | ❌ Not started | All data is mocked |
| **Pagination** | ❌ Not implemented | Feed shows all items; needs pagination |
| **Category Context** | ⚠️ Mock volumes | Needs real category volume calculations |
| **Dispatch Action Logging** | ❌ Not implemented | Button exists but no backend endpoint |

---

## 2. Data Structure Mapping

### 2.1 Mock Data Structure (Figma Code)

**File**: `_vendor/figma_make/src/app/data.ts`

```typescript
interface SignalItem {
  id: string;                    // "SIG-1001"
  timestamp: Date;                 // Generated (last 6 hours)
  source: string;                  // "X", "News", "Direct Report"
  content: string;                 // Message text
  originalContent?: string;        // If translated
  language: Language;             // "en", "es", "fr", "ht"
  riskLevel: RiskLevel;            // "HIGH" | "MEDIUM" | "LOW"
  categories: string[];            // Top 3 category names
  classifications: Classification[]; // All detected categories with confidence
  isTranslated: boolean;
}

interface Classification {
  category: string;                // Display name (e.g., "Medical Help")
  confidence: number;               // 0 to 1
}
```

### 2.2 Database Schema (Current System)

**Table**: `stg_disaster_response`

| Column | Type | Description | Maps To |
|--------|------|-------------|---------|
| `id` | INTEGER | Primary key | `SignalItem.id` → `SIG-{id}` |
| `message` | TEXT | Message content | `SignalItem.content` |
| `original` | TEXT | Original text (if translated) | `SignalItem.originalContent` |
| `genre` | TEXT | Source type: "direct", "news", "social" | `SignalItem.source` (needs mapping) |
| `medical_help` | INTEGER | Binary label (0/1) | Category flag |
| `water` | INTEGER | Binary label (0/1) | Category flag |
| ... (36 total categories) | INTEGER | Binary labels | Category flags |

**All 36 Categories** (from `TARGET_COLUMNS`):
```
related, request, offer, aid_related, medical_help, medical_products,
search_and_rescue, security, military, child_alone, water, food, shelter,
clothing, money, missing_people, refugees, death, other_aid,
infrastructure_related, transport, buildings, electricity, tools, hospitals,
shops, aid_centers, other_infrastructure, weather_related, floods, storm,
fire, earthquake, cold, other_weather, direct_report
```

### 2.3 Mapping Requirements

#### Category Name Mapping (Internal → Display)

| Internal Name (DB) | Display Name (UI) | Status |
|-------------------|-------------------|--------|
| `medical_help` | "Medical Help" | ✅ Matches |
| `medical_products` | "Medical Products" | ✅ Matches |
| `search_and_rescue` | "Search & Rescue" | ✅ Matches |
| `water` | "Water" | ✅ Matches |
| `food` | "Food" | ✅ Matches |
| `shelter` | "Shelter" | ✅ Matches |
| `security` | "Security" | ✅ Matches |
| `hospitals` | "Hospitals" | ✅ Matches |
| `infrastructure_related` | "Infrastructure" | ⚠️ Needs mapping |
| `transport` | "Transport" | ✅ Matches |
| `buildings` | "Buildings" | ✅ Matches |
| `electricity` | "Electricity" | ✅ Matches |
| `tools` | "Tools" | ✅ Matches |
| `shops` | "Shops" | ✅ Matches |
| `aid_centers` | "Aid Centers" | ⚠️ Needs mapping |
| `other_infrastructure` | "Other Infrastructure" | ⚠️ Needs mapping |
| `weather_related` | "Weather Related" | ⚠️ Needs mapping |
| `floods` | "Floods" | ✅ Matches |
| `storm` | "Storm" | ✅ Matches |
| `fire` | "Fire" | ✅ Matches |
| `earthquake` | "Earthquake" | ✅ Matches |
| `cold` | "Cold" | ✅ Matches |
| `other_weather` | "Other Weather" | ⚠️ Needs mapping |
| `missing_people` | "Missing People" | ⚠️ Needs mapping |
| `refugees` | "Refugees" | ✅ Matches |
| `death` | "Death" | ✅ Matches |
| `clothing` | "Clothing" | ✅ Matches |
| `money` | "Money" | ✅ Matches |
| `other_aid` | "Other Aid" | ⚠️ Needs mapping |
| `direct_report` | "Direct Report" | ⚠️ Needs mapping |

**Note**: Figma code uses display names directly. Need to create mapping utility.

#### Genre → Source Mapping

| Database Value | Display Name | Notes |
|---------------|--------------|-------|
| `direct` | "Direct Report" | ✅ Direct mapping |
| `news` | "News" | ✅ Direct mapping |
| `social` | "X" | Standardize on X for social sources |

**Design Spec**: Map `genre` to platform names (can randomize for variety).

#### Timestamp Generation

**Current**: Mock data generates timestamps over last 6 hours randomly.

**Required**: 
- Use actual message timestamps if available in DB
- If not available, generate simulated timestamps over last 6 hours
- Sort by most recent first

**Database Check Needed**: Does `stg_disaster_response` table have a timestamp column?

#### Translation Detection

**Logic**: 
```python
is_translated = (
    row['original'] is not None 
    and row['original'] != row['message']
    and row['original'].strip() != ''
)
```

**UI**: Globe icon (🌐) shown next to language code when `isTranslated === true`.

---

## 3. API Requirements Analysis

### 3.1 Required Endpoints (Design Spec)

#### `GET /api/feed`
**Purpose**: Paginated list of messages with classifications

**Query Parameters**:
- `page` (int, default: 1)
- `limit` (int, default: 25)
- `categories[]` (array, optional): Filter by category names

**Response Format**:
```json
{
  "items": [
    {
      "id": "SIG-1001",
      "timestamp": "2026-01-29T15:24:59Z",
      "source": "X",
      "content": "Urgent: Water rising rapidly...",
      "originalContent": null,
      "language": "en",
      "riskLevel": "HIGH",
      "categories": ["Water", "Search & Rescue", "Floods"],
      "classifications": [
        {"category": "Water", "confidence": 0.92},
        {"category": "Search & Rescue", "confidence": 0.88}
      ],
      "isTranslated": false
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 25,
    "total": 150,
    "totalPages": 6
  }
}
```

**Implementation Notes**:
- Load messages from `stg_disaster_response` table
- For each message, run classification to get probabilities (or use cached results)
- Calculate severity using design spec algorithm
- Map internal category names to display names
- Generate simulated timestamps if DB doesn't have them
- Filter by categories if `categories[]` provided

#### `GET /api/metrics`
**Purpose**: Dashboard metrics (simulated + real)

**Response Format**:
```json
{
  "volToday": 14502,
  "flaggedRate": 4.2,
  "topCategories": [
    {"name": "Medical Help", "count": 1247},
    {"name": "Water", "count": 892},
    {"name": "Food", "count": 756}
  ],
  "trendData": [
    {"time": "6h ago", "count": 45},
    {"time": "5h ago", "count": 120},
    {"time": "4h ago", "count": 80},
    {"time": "3h ago", "count": 210},
    {"time": "2h ago", "count": 150},
    {"time": "1h ago", "count": 95},
    {"time": "Now", "count": 60}
  ],
  "simulated": {
    "volToday": true,
    "flaggedRate": true,
    "trendData": true,
    "topCategories": false
  }
}
```

**Implementation Notes**:
- `volToday`: Simulated (can use DB row count as base)
- `flaggedRate`: Simulated percentage
- `trendData`: Simulated time series (random walk or static)
- `topCategories`: **REAL** - Calculate from database category counts

#### `POST /api/classify` (Enhance Existing)
**Purpose**: Classify a message (enhance existing `/classify` endpoint)

**Current**: `/classify` exists but returns HTML template or complex JSON with hierarchy.

**Required**: Simple JSON response matching React component expectations.

**Request**:
```json
{
  "message": "Urgent: Water rising rapidly..."
}
```

**Response**:
```json
{
  "categories": [
    {"name": "Water", "confidence": 0.92, "volume": 892},
    {"name": "Search & Rescue", "confidence": 0.88, "volume": 421}
  ],
  "severity": "HIGH",
  "maxConfidence": 0.92,
  "avgConfidence": 0.90
}
```

**Implementation Notes**:
- Reuse existing `ModelService` for classification
- Calculate severity using design spec algorithm
- Get category volumes from database counts
- Return simplified response (no hierarchy comparison needed for dashboard)

#### `GET /api/categories`
**Purpose**: Category metadata (names, groups, counts)

**Response Format**:
```json
{
  "categories": [
    {"internal": "medical_help", "display": "Medical Help", "count": 1247},
    {"internal": "water", "display": "Water", "count": 892}
  ],
  "groups": {
    "Critical Needs": ["Medical Help", "Medical Products", "Search & Rescue", "Water", "Food", "Shelter", "Security", "Hospitals"],
    "Infrastructure": ["Transport", "Buildings", "Electricity", "Tools", "Shops", "Aid Centers", "Other Infrastructure"],
    "Weather": ["Floods", "Storm", "Fire", "Earthquake", "Cold", "Other Weather"],
    "Other": ["Missing People", "Refugees", "Death", "Clothing", "Money", "Other Aid", "Military", "Child Alone", "Request", "Offer", "Direct Report"]
  }
}
```

**Implementation Notes**:
- Use `DataService` to get category counts from database
- Map internal names to display names
- Return category groups matching design spec

### 3.2 Current Flask Routes

| Route | Method | Current Purpose | Needs Change? |
|-------|--------|----------------|---------------|
| `/` | GET | Home page (template) | ✅ Keep as-is |
| `/go` | GET/POST | Legacy classification | ✅ Keep as-is |
| `/classify` | GET/POST | Classification with hierarchy | ⚠️ Enhance for API mode |
| `/health` | GET | Health check | ✅ Keep as-is |

**New Routes Needed**:
- `GET /api/feed` → **NEW**
- `GET /api/metrics` → **NEW**
- `GET /api/categories` → **NEW**
- `POST /api/classify` → **ENHANCE** (add JSON mode)

---

## 4. Integration Gaps

### 4.1 Frontend Gaps

| Gap | Impact | Priority | Solution |
|-----|--------|----------|----------|
| **No API Integration** | High | P0 | Replace `MOCK_SIGNALS` with `fetch('/api/feed')` |
| **Mock Classification** | High | P0 | Replace mock logic with `POST /api/classify` |
| **Hardcoded Category Names** | Medium | P1 | Use category mapping utility |
| **No Error Handling** | Medium | P1 | Add try/catch and error states |
| **No Loading States** | Low | P2 | Add spinners/skeletons (some exist) |
| **No Pagination UI** | Medium | P1 | Add "Load More" button or infinite scroll |

### 4.2 Backend Gaps

| Gap | Impact | Priority | Solution |
|-----|--------|----------|----------|
| **No API Routes** | High | P0 | Create `app/routes/api.py` blueprint |
| **No Feed Endpoint** | High | P0 | Implement `GET /api/feed` |
| **No Metrics Endpoint** | High | P0 | Implement `GET /api/metrics` |
| **No Categories Endpoint** | Medium | P1 | Implement `GET /api/categories` |
| **Classification Response Format** | High | P0 | Enhance `/classify` to return simplified JSON |
| **No Severity Calculation** | High | P0 | Implement severity algorithm from design spec |
| **No Category Mapping** | Medium | P1 | Create mapping utility (internal → display) |
| **No Timestamp Generation** | Medium | P1 | Add timestamp simulation logic |
| **No Category Volume Calculation** | Medium | P1 | Query DB for category counts |

### 4.3 Data Gaps

| Gap | Impact | Priority | Solution |
|-----|--------|----------|----------|
| **Missing Timestamps in DB** | Medium | P1 | Generate simulated timestamps |
| **No Cached Classifications** | Low | P2 | Consider caching model predictions (future optimization) |
| **Genre Mapping** | Low | P2 | Create mapping utility for source names |

---

## 5. Category Mapping Analysis

### 5.1 Category Groups (Design Spec vs Figma Code)

**Design Spec Groups**:
```python
CRITICAL_NEEDS = [
    'medical_help', 'medical_products', 'search_and_rescue',
    'water', 'food', 'shelter', 'security', 'hospitals'
]

INFRASTRUCTURE = [
    'infrastructure_related', 'transport', 'buildings', 'electricity',
    'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure'
]

WEATHER_EVENTS = [
    'weather_related', 'floods', 'storm', 'fire',
    'earthquake', 'cold', 'other_weather'
]

OTHER = [
    'missing_people', 'refugees', 'death', 'clothing', 'money',
    'other_aid', 'military', 'child_alone', 'request', 'offer',
    'direct_report', 'aid_related', 'related'
]
```

**Figma Code Groups** (from `data.ts`):
```typescript
CATEGORY_GROUPS = {
  "Critical Needs": [
    "Medical Help", "Medical Products", "Search & Rescue", "Water", "Food", 
    "Shelter", "Security", "Hospitals"
  ],
  "Infrastructure": [
    "Transport", "Buildings", "Electricity", "Tools", "Shops", 
    "Aid Centers", "Other Infrastructure"
  ],
  "Weather": [
    "Floods", "Storm", "Fire", "Earthquake", "Cold", "Other Weather"
  ],
  "Other": [
    "Missing People", "Refugees", "Death", "Clothing", "Money", 
    "Other Aid", "Military", "Child Alone", "Request", "Offer", "Direct Report"
  ]
}
```

**Differences**:
- Figma uses display names (already human-readable)
- Design spec uses internal names (snake_case)
- Need bidirectional mapping utility

**Note**: `hospitals` appears in both Critical Needs and Infrastructure groups (design spec). Figma code only has it in Critical Needs.

---

## 6. Severity Calculation Comparison

### Design Spec Algorithm
```python
def calculate_severity(categories: dict, probabilities: dict) -> str:
    CRITICAL_CATEGORIES = {
        'medical_help', 'medical_products', 'search_and_rescue',
        'water', 'food', 'shelter', 'security', 'hospitals'
    }
    
    critical_count = sum(
        1 for cat, label in categories.items()
        if cat in CRITICAL_CATEGORIES and label == 1
    )
    
    max_confidence = max(probabilities.values()) if probabilities else 0.0
    
    if critical_count >= 2 or max_confidence > 0.85:
        return 'HIGH'
    elif critical_count >= 1 or max_confidence > 0.70:
        return 'MEDIUM'
    else:
        return 'LOW'
```

### Figma Code Algorithm
```typescript
const calculateSeverity = (classifications: Classification[]): RiskLevel => {
  const criticalCount = classifications.filter(c => 
    CRITICAL_CATEGORIES.includes(c.category) && c.confidence > 0.5
  ).length;
  
  const maxConf = Math.max(...classifications.map(c => c.confidence));

  if (criticalCount >= 2 || maxConf > 0.85) return "HIGH";
  if (criticalCount >= 1 || maxConf > 0.70) return "MEDIUM";
  return "LOW";
};
```

**Differences**:
- Design spec uses binary labels (`label == 1`)
- Figma code uses confidence threshold (`c.confidence > 0.5`)
- **Decision Needed**: Use binary labels (from DB) or probabilities (from model)?

**Recommendation**: Use probabilities from model predictions (more accurate), but check against threshold to determine if category is "flagged".

---

## 7. Implementation Plan

### Phase 0: Validation & Quick Wins (Priority: P0)

**Purpose**: Validate existing infrastructure and capture low-hanging fruit before building new features.

**Tasks**:
1. ✅ Test existing `/classify` JSON endpoint
   - Verify `Content-Type: application/json` works
   - Test `?format=json` query param
   - Document current response format
2. ✅ Build React app standalone
   - Run `npm install` and `npm run build` in `_vendor/figma_make`
   - Verify it runs with mock data
   - Test resizable panels and UI components
3. ✅ Create simple category display name mapping
   - Create dict for ~10 special cases (inline in `api.py`)
   - Use simple `.replace('_', ' ').title()` for others
   - No separate utility file needed

**Estimated Time**: 30 minutes

**Rationale**: Validates we can reuse existing endpoints and confirms frontend works before integration.

---

### Phase 1: Backend API Foundation (Priority: P0)

**Tasks**:
1. ✅ Create `app/routes/api.py` blueprint
2. ✅ Implement `GET /api/categories` endpoint
   - Load category metadata from `DataService`
   - Use inline category display name mapping (dict + simple function)
   - Calculate category counts: `df[category_cols].sum().to_dict()` (5 lines)
   - Return category groups matching design spec
3. ✅ Implement severity calculation (inline function in `api.py`)
   - Simple function using probabilities from model
   - No separate utility file needed

**Estimated Time**: 1.5-2 hours (reduced from 2-3 hours)

**Simplifications**:
- Category mapping: Simple dict + function inline (not separate utility)
- Category counts: Direct pandas operation (not utility module)
- Severity: Inline function (not separate file)

---

### Phase 2: Feed Endpoint (Priority: P0)

**Tasks**:
1. ✅ Implement `GET /api/feed` endpoint
   - Load messages from database via `DataService`
   - Run classifications using existing `ModelService`
   - Calculate severity inline (reuse function from Phase 1)
   - Generate simulated timestamps inline:
     ```python
     from datetime import datetime, timedelta
     timestamp = datetime.now() - timedelta(hours=(index/total)*6)
     ```
   - Map categories to display names (reuse mapping from Phase 1)
   - Implement pagination (simple offset/limit)
   - Add category filtering (filter by category columns)
2. ✅ Source mapping inline (simple dict lookup):
   ```python
   GENRE_TO_SOURCE = {'direct': 'Direct Report', 'news': 'News', 
                      'social': 'X'}
   ```

**Estimated Time**: 3-4 hours (reduced from 4-5 hours)

**Simplifications**:
- Timestamp generation: Inline datetime math (not utility file)
- Source mapping: Simple dict lookup (not utility file)
- Reuse existing `DataService` and `ModelService`

---

### Phase 3: Metrics Endpoint (Priority: P0)

**Tasks**:
1. ✅ Implement `GET /api/metrics` endpoint
   - Calculate real category counts (reuse logic from Phase 1)
   - Generate simulated volume metrics (simple random/static values)
   - Generate simulated trend data (6-hour time series, can be static)
   - Return combined real + simulated data
   - Mark simulated fields in response

**Estimated Time**: 1.5-2 hours (reduced from 2-3 hours)

**Simplifications**:
- Reuse category count calculation from Phase 1
- Simulated data can be static arrays (no complex generation needed)

---

### Phase 4: Classification Enhancement (Priority: P0)

**Tasks**:
1. ✅ Option A: Enhance existing `/classify` endpoint
   - Add `?simple=true` query param for simplified response
   - Transform existing response to match React component needs
   - Add severity calculation (reuse from Phase 1)
   - Add category volume context (reuse from Phase 3)
   
   **OR**
   
   ✅ Option B: Create `/api/classify` wrapper endpoint
   - Thin wrapper around existing `/classify` logic
   - Returns simplified format matching React expectations
   - Reuses `process_prediction_result` and `ModelService`

**Estimated Time**: 1.5-2 hours (reduced from 2-3 hours)

**Decision Point**: Test Phase 0 first, then decide which approach is cleaner.

---

### Phase 5: Frontend Integration (Priority: P0)

**Tasks**:
1. ✅ Copy React build output to `app/static/dashboard/`
2. ✅ Create Flask route to serve React app (`/dashboard`)
   - Serve `index.html` for all routes (SPA routing)
   - Ensure static assets are accessible
3. ✅ Update React components to fetch from API
   - Replace `MOCK_SIGNALS` with `fetch('/api/feed')`
   - Replace mock classification with `POST /api/classify` (or `/classify?simple=true`)
   - Replace mock metrics with `fetch('/api/metrics')`
   - Update category mapping to use API response
4. ✅ Add error handling and loading states
5. ✅ Test end-to-end flow

**Estimated Time**: 4-5 hours (reduced from 4-6 hours)

**Simplifications**:
- React app already built in Phase 0
- Can reuse existing `/classify` endpoint if it works

---

### Phase 6: Polish & Testing (Priority: P1)

**Tasks**:
1. ✅ Add pagination UI (Load More button)
2. ✅ Add error boundaries in React
3. ✅ Add loading skeletons (some already exist)
4. ✅ Test with real database data
5. ✅ Performance optimization (caching, lazy loading) - only if needed

**Estimated Time**: 2-3 hours (reduced from 3-4 hours)

**Total Estimated Time**: 14-18 hours (reduced from 17-24 hours)

---

## 7.1 Desktop Application Design Dimensions

### Viewport Size Standards

**Design Target**: Desktop-only application (no responsive breakpoints needed)

**Common Desktop Viewport Sizes (2024-2025)**:
- **1920×1080** (Full HD) - 22.63% of users
- **1366×768** (HD) - 17.42% of users  
- **1536×864** - 11.13% of users
- **1280×720** (HD) - 6.05% of users
- **1440×900** - 5.63% of users
- **1600×900** - 3.69% of users

**Minimum Supported Width**: **1280px** (covers 95%+ of desktop users)

**Recommended Design Width**: **1440px** (good balance, works well for 3-panel layout)

**Maximum Design Width**: **1920px** (don't design wider, use max-width constraints)

### Panel Width Guidelines

**Current Design Spec**:
- Left Panel (Feed): Default 40%, min 300px, max 60%
- Center Panel (Metrics): Default 35%, min 300px
- Right Panel (Controls): Default 25%, min 250px, collapsible

**Best Practices for Resizable Panels**:

1. **Minimum Widths**:
   - Ensure panels remain usable at minimum widths
   - Feed panel: 300px allows ~40-50 characters per line (readable)
   - Metrics panel: 300px fits chart + cards comfortably
   - Classification panel: 250px fits form + results

2. **Maximum Widths**:
   - Left panel max 60% prevents excessive width (hard to scan)
   - Center/Right panels don't need max (they'll shrink naturally)

3. **Responsive Behavior** (for window resizing):
   - At 1280px width: Panels should auto-adjust to fit
   - Below 1280px: Show mobile warning banner (already implemented)
   - Panels should maintain aspect ratios when resized

4. **Content Density**:
   - **Feed items**: ~120-140px height per item (comfortable scrolling)
   - **Metrics cards**: ~80-100px height (fits 2 cards vertically)
   - **Charts**: Minimum 150px height for readability
   - **Form inputs**: Standard 40-44px height (touch-friendly even on desktop)

### Typography & Spacing

**Font Sizes** (for desktop):
- **Headers**: 14-16px (small headers), 18-20px (main titles)
- **Body text**: 12-14px (readable at 1280px+)
- **Labels/Meta**: 10-11px (timestamps, badges)
- **Code/Mono**: 10-12px (IDs, percentages)

**Spacing**:
- **Panel padding**: 12-16px (comfortable, not cramped)
- **Item spacing**: 8-12px between feed items
- **Card padding**: 16-20px inside metric cards
- **Form spacing**: 12-16px between form elements

### Color & Contrast

**Desktop-Specific Considerations**:
- Higher contrast ratios (WCAG AAA) for long reading sessions
- Subtle backgrounds (light gray/white) reduce eye strain
- Clear visual hierarchy with borders/shadows (more space = can be subtle)

### Performance Considerations

**Viewport Optimization**:
- **Virtual scrolling**: Consider for feed if >100 items (not needed initially)
- **Lazy loading**: Load images/icons on demand
- **Chart rendering**: Limit data points for smooth rendering (6-hour trend = 6-7 points is fine)

### Testing Checklist

**Viewport Testing**:
- ✅ Test at 1280px width (minimum)
- ✅ Test at 1440px width (recommended)
- ✅ Test at 1920px width (maximum)
- ✅ Test panel resizing at each width
- ✅ Verify content doesn't overflow or become unreadable
- ✅ Verify mobile warning shows below 768px (already implemented)

**Browser Testing**:
- Chrome/Edge (Chromium) - primary target
- Firefox - secondary
- Safari - if Mac users expected

---

## 7.2 Updated File Structure Plan

### New Files to Create (Simplified)

```
app/
  routes/
    api.py                    # NEW: API blueprint with inline utilities
  static/
    dashboard/                # NEW: React build output
      index.html
      assets/
        *.js
        *.css
```

### Files to Modify

```
app/
  routes/
    classification.py         # MODIFY: Add ?simple=true mode (optional)
  app.py                      # MODIFY: Register API blueprint
```

### Files NOT Needed (Simplified Approach)

~~`app/utils/severity.py`~~ → Inline function in `api.py`  
~~`app/utils/category_display.py`~~ → Inline dict + function in `api.py`  
~~`app/utils/timestamp_generator.py`~~ → Inline datetime math  
~~`app/utils/source_mapper.py`~~ → Inline dict lookup

**Rationale**: These are simple operations (5-10 lines each) that don't need separate modules. Keep code close to where it's used for easier maintenance.

---

## 9. Dependencies Check

### Frontend Dependencies (Figma Code)

**Already in package.json**:
- ✅ React 18.3.1
- ✅ Vite 6.3.5
- ✅ Tailwind CSS 4.1.12
- ✅ react-resizable-panels 2.1.7
- ✅ recharts 2.15.2 (for charts)
- ✅ lucide-react (icons)
- ✅ date-fns (date formatting)

**No additional dependencies needed** ✅

### Backend Dependencies (Flask App)

**Already available**:
- ✅ Flask (for API routes)
- ✅ pandas (for data loading)
- ✅ sqlalchemy (for database access)
- ✅ ModelService (for classifications)

**No additional dependencies needed** ✅

---

## 10. Risk Assessment

### Low Risk ✅
- Category mapping (straightforward string mapping)
- Timestamp generation (simple date math)
- Source mapping (dictionary lookup)

### Medium Risk ⚠️
- **Performance**: Loading all messages for feed could be slow
  - **Mitigation**: Implement pagination, limit queries
- **Classification Speed**: Running model on every feed item could be slow
  - **Mitigation**: Consider caching predictions, or pre-compute classifications
- **Category Counts**: Calculating counts for all categories on every metrics request
  - **Mitigation**: Cache category counts, refresh periodically

### High Risk 🔴
- **None identified** - All components are straightforward integrations

---

## 11. Testing Strategy

### Unit Tests
- ✅ Severity calculation algorithm
- ✅ Category name mapping
- ✅ Source mapping
- ✅ Timestamp generation

### Integration Tests
- ✅ API endpoints return correct JSON format
- ✅ Feed pagination works correctly
- ✅ Category filtering works
- ✅ Classification endpoint returns expected format

### End-to-End Tests
- ✅ React app loads and displays data
- ✅ Feed updates when filters change
- ✅ Classification panel works end-to-end
- ✅ Metrics panel displays data

---

## 12. Next Steps

### Immediate Actions (Today)
1. ✅ Review this analysis document
2. ✅ **Phase 0**: Test existing `/classify` JSON endpoint (15 min)
3. ✅ **Phase 0**: Build React app standalone and verify (15 min)
4. ✅ Start Phase 1: Backend API Foundation (simplified approach)

### Short-term (This Week)
1. ✅ Complete Phases 1-4 (Backend API) - **14-18 hours total** (reduced from 17-24)
2. ✅ Complete Phase 5 (Frontend Integration)
3. ✅ Basic testing at key viewport sizes (1280px, 1440px, 1920px)

### Medium-term (Next Week)
1. ✅ Phase 6 (Polish & Testing)
2. ✅ Performance optimization (only if needed)
3. ✅ Documentation updates
4. ✅ Cross-browser testing (Chrome, Firefox, Safari)

---

## 13. Open Questions

1. **Severity Calculation**: Use binary labels from DB or probabilities from model?
   - **✅ Resolved**: Use probabilities from model (more accurate)
   
2. **Timestamp Column**: Does `stg_disaster_response` have a timestamp column?
   - **✅ Resolved**: No timestamp column exists → Generate simulated timestamps
   
3. **Caching Strategy**: Should we cache model predictions for feed items?
   - **✅ Resolved**: Start without caching, add if performance issues arise
   
4. **Category Counts**: Should category counts be real-time or cached?
   - **✅ Resolved**: Cache with 5-minute TTL (metrics don't need real-time)

5. **Classification Endpoint**: Enhance existing `/classify` or create `/api/classify`?
   - **Decision Point**: Test Phase 0 first, then decide based on response format complexity
   - **Recommendation**: Try `?simple=true` mode first (less code), fall back to new endpoint if needed

6. **Desktop Viewport Testing**: What's the minimum supported width?
   - **✅ Resolved**: 1280px minimum (covers 95%+ of desktop users)

---

## Appendix A: Category Name Mapping Reference

See `docs/sessions/active/2026_01_29_storm_signal_dashboard_design_spec.md` Section "Category Mapping" for complete mapping table.

---

## Appendix B: API Response Examples

### Feed Response Example
```json
{
  "items": [
    {
      "id": "SIG-1001",
      "timestamp": "2026-01-29T15:24:59Z",
      "source": "X",
      "content": "Urgent: Water rising rapidly in downtown area...",
      "originalContent": null,
      "language": "en",
      "riskLevel": "HIGH",
      "categories": ["Water", "Search & Rescue", "Floods"],
      "classifications": [
        {"category": "Water", "confidence": 0.92},
        {"category": "Search & Rescue", "confidence": 0.88},
        {"category": "Floods", "confidence": 0.75}
      ],
      "isTranslated": false
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 25,
    "total": 150,
    "totalPages": 6
  }
}
```

### Metrics Response Example
```json
{
  "volToday": 14502,
  "flaggedRate": 4.2,
  "topCategories": [
    {"name": "Medical Help", "count": 1247},
    {"name": "Water", "count": 892},
    {"name": "Food", "count": 756},
    {"name": "Shelter", "count": 634},
    {"name": "Search & Rescue", "count": 421}
  ],
  "trendData": [
    {"time": "6h ago", "count": 45},
    {"time": "5h ago", "count": 120},
    {"time": "4h ago", "count": 80},
    {"time": "3h ago", "count": 210},
    {"time": "2h ago", "count": 150},
    {"time": "1h ago", "count": 95},
    {"time": "Now", "count": 60}
  ]
}
```

### Classification Response Example
```json
{
  "categories": [
    {"name": "Water", "confidence": 0.92, "volume": 892},
    {"name": "Search & Rescue", "confidence": 0.88, "volume": 421}
  ],
  "severity": "HIGH",
  "maxConfidence": 0.92,
  "avgConfidence": 0.90
}
```

---

**Document Status**: ✅ Complete - Ready for implementation planning
