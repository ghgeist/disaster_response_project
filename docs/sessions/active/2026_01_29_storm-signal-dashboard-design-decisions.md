# Storm Signal Dashboard - Key Decisions

## Architecture
- **Frontend**: React 18+ (Single Page Application) with Tailwind CSS
- **Backend**: Flask (API Mode) + JSON endpoints
- **Build**: Vite (integrated with Flask)

## Design Direction
- **Theme**: Light theme (analytics-first, clean professional aesthetic)
- **Layout**: Three-panel desktop layout with resizable split-panes
- **Header**: Hamburger menu (left), Logo/Title, User controls (right)

## Key Features

### Live Feed (Left Panel)
- **Filters**: Collapsible filter drawer in the left panel
- **Feed**: Real messages, simulated timestamps, severity badges
- **Context**: Source mapping, translation indicators

### Metrics Dashboard (Center Panel)
- **Visuals**: Trend graph, volume metrics, top categories list
- **Data**: Mix of real (categories) and simulated (volume/trends)
- **Disclaimer**: Subtle "METRICS SIMULATED" badge instead of banner

### Classification Interface (Right Panel)
- **Workflow**: Input -> Classify -> Review -> Action
- **Input**: Clean text area (removed instruction banner)
- **Context**: Show "Volume Today" for detected categories to provide operational context
- **Action**: "Dispatch Assistance" button (simulates operational handoff)
- **Empty State**: "Mark as Irrelevant" workflow for negative results

## Severity Calculation
- **High**: ≥2 critical categories OR confidence > 85%
- **Medium**: 1 critical category OR confidence > 70%
- **Low**: Otherwise

## Category Mapping
- Use all 36 actual categories (human-readable names)
- Grouped into: Critical Needs, Infrastructure, Weather Events, Other

## Next Steps
1. ✅ Design spec document finalized
2. ⏳ Refine Figma design (Action button, Header nav)
3. ⏳ Set up React + Vite + Flask environment
4. ⏳ Build API endpoints (`/api/feed`, `/api/metrics`, `/api/classify`)
5. ⏳ Implement frontend components
