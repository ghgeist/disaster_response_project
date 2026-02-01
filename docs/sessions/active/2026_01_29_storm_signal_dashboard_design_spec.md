---
created: 2026-01-29
updated: 2026-01-29
status: active
---

# Storm Signal Dashboard — Design Specification

> **Purpose**: Transform the basic classification demo into a contextualized decision-support dashboard that showcases both ML classification and operational intelligence.

> **Related**: See `2026_01_29_storm_signal_system_spec.md` for system architecture overview.

---

## Design Goals

1. **Contextualize Classifications**: Show how individual signals fit into broader patterns
2. **Decision Support**: Enable operators to triage and prioritize effectively
3. **Portfolio Quality**: Demonstrate full-stack ML engineering (not just model accuracy)
4. **Honest Demo**: Clear distinction between real classification and simulated operational context

---

## Architecture

**Frontend**: React (Single Page Application)
- **Framework**: React 18+
- **Styling**: Tailwind CSS
- **State Management**: React Context or Zustand
- **Build Tool**: Vite (integrated with Flask)

**Backend**: Flask (API Mode)
- **API Endpoints**: Serve JSON data to React frontend
- **Static Files**: Serve compiled React assets

---

## Layout Structure

### Header
- **Left**: [Hamburger Menu] [Logo] "Storm Signal" [Title: "Intelligence Dashboard"]
- **Right**: [Live Status] [Notifications] [Settings] [User Profile]
- **Hamburger Menu**: Opens drawer with "About", "Documentation", "Project Repo", "Logout"

### Three-Panel Layout (Desktop Only)

**Structure**: Resizable split-pane layout
- **Left Panel (Feed)**: Default 40% width, min 300px, max 60%
- **Center Panel (Metrics)**: Default 35% width, min 300px
- **Right Panel (Controls)**: Default 25% width, min 250px, collapsible
- **Dividers**: Draggable handles between panels to adjust width
- **State**: Persist panel sizes in local storage

**Left Panel**: Live Feed & Filters
- Header with Filter toggle
- Active filter chips
- Scrollable list of classified messages
- Real messages from database with simulated timestamps
- Category tags, severity indicators, confidence scores

**Center Panel**: Metrics & Trends
- Volume metrics (simulated)
- Flagged signals trend graph (simulated)
- Top categories (real data)
- Category grouping visualization

**Right Panel**: Classification Interface
- Message input form
- Classification results
- Category context (volume/trends for detected categories)
- Settings/controls

### Mobile Handling

- **No responsive design** — desktop-only experience
- Show dismissible banner message: "Storm Signal is optimized for desktop viewing. Please access from a larger screen for the full experience."
- Use standard mobile detection (user agent or viewport width < 768px)

---

## Panel 1: Live Feed & Filters (Left)

### Header
- Title: "Live Feed"
- Count badge: "Live Feed [N]"
- **Filter Button**: Icon button (funnel) to toggle filter drawer

### Filters (Collapsible Drawer/Section)
- **Category Filters**:
  - Multi-select checkboxes (square icons)
  - Grouped by category groups (Critical Needs, Infrastructure, Weather, Other)
- **Active Filter Chips**:
  - Show selected filters as removable tags at top of list
  - "Clear All" button

### Feed Items

Each item displays:

1. **Timestamp** (simulated)
   - Format: `HH:MM:SS` (e.g., "11:24:59")
   - Generate timestamps spread over last 6 hours
   - Most recent at top

2. **Severity Badge**
   - Color-coded pill: `HIGH` (red), `MEDIUM` (yellow), `LOW` (gray)
   - Calculation:
     - Count critical categories flagged: `medical_help`, `medical_products`, `search_and_rescue`, `water`, `food`, `shelter`, `security`, `hospitals`
     - Severity = `HIGH` if ≥2 critical categories OR max confidence > 0.85
     - Severity = `MEDIUM` if 1 critical category OR max confidence > 0.70
     - Severity = `LOW` otherwise

3. **Confidence Score + Language**
   - Format: `[XX]% [LANG]`
   - Confidence: Max probability across all categories (rounded to nearest %)
   - Language: Always "EN" (English)
   - Translation indicator: If `original` column exists and differs from `message`, show small icon (🌐) next to language code

4. **Message Preview**
   - Truncate to 120 characters
   - Show ellipsis (...) if truncated
   - Use actual `message` text from database
   - If truncated, make clickable to expand (modal or expand in place)

5. **Category Tags**
   - Show top 3 categories (by confidence) as small pills
   - Format: Category name (human-readable, e.g., "Medical Help" not "medical_help")
   - Color: Match severity (red for HIGH, yellow for MEDIUM, gray for LOW)
   - If more than 3 categories, show "+N more" indicator

6. **Source & Signal ID**
   - Source: Map `genre` column:
     - `direct` → "Direct Report"
     - `news` → "News"
     - `social` → "X"
   - Signal ID: Format `SIG-[ID]` where ID = database `id` column (e.g., "SIG-1001")
   
7. **Translation Indicator** (if applicable)
   - Small icon (🌐) next to language code
   - Tooltip: "Translated from Haitian Creole"
   - Only show if `original` column exists and differs from `message`

### Feed Behavior

- **Load**: Display 25 most recent messages (by simulated timestamp)
- **Scroll**: Infinite scroll or "Load More" button
- **Click**: Expand to show full message + all categories + classification details
- **Filter**: Respect active category filters (hide items that don't match)

---

## Panel 2: Metrics & Trends (Center)

### Header
- **Disclaimer Badge**: Small status pill in top-right corner
- Text: `METRICS SIMULATED`
- Style: Subtle gray/outline
- Tooltip: "Dashboard metrics are simulated for demo realism; classification results are real."

### Key Metrics Cards

**VOL TODAY**
- Large number: `14,502` (simulated, can be static or incrementing)
- Subtitle: "Signals processed"
- Styling: Large, prominent

**FLAGGED**
- Large number: `4.2%` (simulated percentage)
- Subtitle: "Requiring attention"
- Color: Red text
- Calculation: Can be static or calculated from displayed feed items

### Flagged Signals Trend (6H)

**Chart**: Line graph
- X-axis: Time (last 6 hours, hourly intervals)
- Y-axis: Count of flagged signals
- Data: Simulated trend (can be static or random walk)
- Styling: Red line, subtle grid
- Tooltip: Show exact count on hover

### Top Categories

**Title**: "TOP CATEGORIES"

**List**: Show top 5-7 categories by actual count from database
- Format: `Category Name: [count]`
- Examples:
  - Medical Help: 1,247
  - Water: 892
  - Food: 756
  - Shelter: 634
  - Search & Rescue: 421
- Use actual category counts from database
- Human-readable names (see Category Mapping below)

### Category Grouping Visualization (Optional)

**Title**: "CATEGORY BREAKDOWN"

**Visualization**: Stacked bar or pie chart showing:
- Critical Needs (sum of: medical_help, medical_products, search_and_rescue, water, food, shelter, security, hospitals)
- Infrastructure (sum of: infrastructure_related, transport, buildings, electricity, tools, hospitals, shops, aid_centers, other_infrastructure)
- Weather Events (sum of: weather_related, floods, storm, fire, earthquake, cold, other_weather)
- Other Aid (sum of: other_aid, missing_people, refugees, death, clothing, money)

---

## Panel 3: Classification Interface (Right)

### Header
- Title: "Classify Message"
- Subtitle: "Enter a message to classify"

### Message Input

**Text Area**
- Placeholder: "Enter a message about a disaster or emergency situation..."
- Multi-line input
- Character counter (optional)
- Submit button: "Classify"
- **Note**: Removed explicit instruction banner ("Paste a raw message...") in favor of clear placeholder text.

### Classification Results

**Display after classification**:

1. **Detected Categories**
   - List of categories with confidence scores
   - Format: `Category Name: [XX]%`
   - Sort by confidence (highest first)

2. **Category Context (New)**
   - For each detected category, show:
     - **Volume Today**: "892 signals today"
     - **Trend**: "High volume" or "Normal" indicator
   - Connects the individual signal to the broader operational picture

3. **Severity Indicator**
   - Same severity calculation as feed items
   - Display badge: HIGH/MEDIUM/LOW

4. **Confidence Summary**
   - Max confidence score
   - Average confidence across flagged categories

5. **Action Section**
   - **"Dispatch Assistance" Button**: Primary action button
   - State 1: "Dispatch Assistance" (Primary Color)
   - State 2 (Loading): Spinner icon
   - State 3 (Success): "Signal Dispatched" (Green) + Checkmark
   - Interaction: Simulates operational dispatch, logs action

### Empty State
- When no categories detected:
  - Show "No Critical Categories Detected" message
  - "Mark as Irrelevant" button (Secondary action)

### Recall Sensitivity (Removed)

- **Removed**: "Recall Sensitivity" slider
- **Rationale**: Classification happens in this panel, threshold adjustment is a model-level concern, not UI control

---

## Category Mapping

### Human-Readable Names

Map internal category names to display names:

| Internal Name | Display Name |
|--------------|--------------|
| medical_help | Medical Help |
| medical_products | Medical Products |
| search_and_rescue | Search & Rescue |
| water | Water |
| food | Food |
| shelter | Shelter |
| security | Security |
| hospitals | Hospitals |
| infrastructure_related | Infrastructure |
| transport | Transport |
| buildings | Buildings |
| electricity | Electricity |
| tools | Tools |
| shops | Shops |
| aid_centers | Aid Centers |
| other_infrastructure | Other Infrastructure |
| weather_related | Weather Related |
| floods | Floods |
| storm | Storm |
| fire | Fire |
| earthquake | Earthquake |
| cold | Cold |
| other_weather | Other Weather |
| missing_people | Missing People |
| refugees | Refugees |
| death | Death |
| clothing | Clothing |
| money | Money |
| other_aid | Other Aid |
| military | Military |
| child_alone | Child Alone |
| request | Request |
| offer | Offer |
| direct_report | Direct Report |
| aid_related | Aid Related |
| related | Related |

### Category Groups

**Critical Needs** (8 categories):
- medical_help, medical_products, search_and_rescue, water, food, shelter, security, hospitals

**Infrastructure** (9 categories):
- infrastructure_related, transport, buildings, electricity, tools, hospitals, shops, aid_centers, other_infrastructure

**Weather Events** (7 categories):
- weather_related, floods, storm, fire, earthquake, cold, other_weather

**Other** (remaining categories):
- missing_people, refugees, death, clothing, money, other_aid, military, child_alone, request, offer, direct_report, aid_related, related

---

## Severity Calculation

### Algorithm

```python
def calculate_severity(categories: dict, probabilities: dict) -> str:
    """
    Calculate severity based on critical categories and confidence.
    
    Args:
        categories: Dict of category -> binary label (0/1)
        probabilities: Dict of category -> confidence score (0-1)
    
    Returns:
        'HIGH', 'MEDIUM', or 'LOW'
    """
    CRITICAL_CATEGORIES = {
        'medical_help', 'medical_products', 'search_and_rescue',
        'water', 'food', 'shelter', 'security', 'hospitals'
    }
    
    # Count critical categories flagged
    critical_count = sum(
        1 for cat, label in categories.items()
        if cat in CRITICAL_CATEGORIES and label == 1
    )
    
    # Get max confidence
    max_confidence = max(probabilities.values()) if probabilities else 0.0
    
    # Determine severity
    if critical_count >= 2 or max_confidence > 0.85:
        return 'HIGH'
    elif critical_count >= 1 or max_confidence > 0.70:
        return 'MEDIUM'
    else:
        return 'LOW'
```

---

## Data Requirements

### Real Data (from database)

- **Messages**: Use `message` column (truncate to 120 chars for preview)
- **Original**: Use `original` column to detect translations
- **Categories**: Use all 36 category columns (binary labels)
- **Genre**: Use `genre` column for source mapping
- **ID**: Use `id` column for signal IDs

### Simulated Data

- **Timestamps**: Generate over last 6 hours (most recent first)
- **Volume metrics**: Simulated counts (can be based on database size)
- **Trend graph**: Simulated time series data
- **Source platforms**: Map `genre` to platform names (can randomize for variety)

---

## API Endpoints Needed

### Backend Routes

1. **`GET /api/feed`**
   - Returns paginated list of messages with classifications
   - Query params: `page`, `limit`, `categories[]` (filter)
   - Response: JSON array of feed items

2. **`GET /api/metrics`**
   - Returns dashboard metrics
   - Response: `{vol_today, flagged_pct, top_categories: [...], trend_data: [...]}`

3. **`POST /api/classify`** (existing, enhance)
   - Classify a message
   - Response: Include severity calculation

4. **`GET /api/categories`**
   - Returns category metadata (names, groups, counts)
   - Response: `{categories: [...], groups: {...}}`

---

## Figma AI Refinement Prompt

```
I'm building a disaster response intelligence dashboard called "Storm Signal". Refine the design based on these updated specifications:

LAYOUT:
- Desktop layout with resizable split-panes
- **Top Header**: [Hamburger Menu] [Logo] [Title] ... [User Profile]
- Left panel (Feed & Filters): 40% width
- Center panel (Metrics): 35% width
- Right panel (Classify): 25% width
- Show draggable dividers between panels

THEME:
- Light theme only (clean, white/gray background)
- Analytics-first professional aesthetic

LEFT PANEL - FEED & FILTERS:
- Header: "Live Feed [N]" with Filter icon button
- Filter Section (Collapsible):
  * "ACTIVE FILTERS" header
  * Multi-select category filters (SQUARE checkboxes) grouped by category groups
  * "Clear All" button
- Feed List: Real-looking messages, category tags, severity badges

CENTER PANEL - METRICS:
- **Disclaimer**: Remove large banner. Add subtle "METRICS SIMULATED" badge/pill in top-right corner.
- Two metric cards: "VOL TODAY" (large number) and "FLAGGED" (percentage in red)
- Line chart: "FLAGGED SIGNALS (6H)" showing trend over last 6 hours
- "TOP CATEGORIES" list: Top 5-7 categories with counts (use real category names: Medical Help, Water, Food, Shelter, Search & Rescue, etc.)
- Optional: Category grouping visualization (Critical Needs, Infrastructure, Weather Events, Other)

RIGHT PANEL - CLASSIFICATION:
- Header: "Classify Message"
- **Remove Instruction Box**: Delete the yellow "Paste a raw message..." box.
- Text input area with submit button
- Results section showing:
  * Detected categories with confidence scores
  * **Category Context**: "Volume Today" count
  * Severity badge
  * **Action Button**: "Dispatch Assistance" (Primary action)
  * **Empty State**: "Mark as Irrelevant" button if no categories found

Refine the design to add the header navigation, dispatch button, and ensure light theme consistency.
```

---

## Notes

- **Simulated vs Real**: Be explicit about what's simulated (timestamps, volume metrics, trends) vs real (classifications, messages, categories)
- **Performance**: Consider pagination and lazy loading for large message feeds
- **Accessibility**: Ensure color contrast, keyboard navigation, screen reader support
- **Error States**: Design for missing data, API failures, loading states
