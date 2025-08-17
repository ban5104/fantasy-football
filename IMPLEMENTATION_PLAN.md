# Fantasy Football Draft Interface - Comprehensive Implementation Plan

## Executive Summary
Build a production-ready web application for fantasy football draft management featuring a three-column layout with real-time probability calculations, interactive visualizations, and strategic decision support.

## 1. Architecture Overview

### Tech Stack
```
Frontend:
- React 18 + TypeScript 4.9+
- Vite (build tool)
- TailwindCSS + shadcn/ui components
- D3.js for custom visualizations
- Recharts for standard charts
- Zustand for state management
- React Query for data fetching

Backend:
- FastAPI (Python) for API endpoints
- Pandas for data processing
- WebSocket for real-time updates
- Redis for caching (optional)

Deployment:
- Docker containerization
- Nginx reverse proxy
- Single executable option with Electron
```

### Data Flow Architecture
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   CSV Data  │────▶│  Python API  │────▶│ React App   │
│  ESPN/ADP   │     │   FastAPI    │     │  (Vite)     │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                    ┌──────▼──────┐
                    │  WebSocket   │
                    │ Real-time    │
                    └─────────────┘
```

## 2. Component Hierarchy

```
App
├── Layout
│   ├── LeftColumn (Pick Context)
│   │   ├── CurrentPickCard
│   │   ├── NextPickCard
│   │   ├── PickHistorySparkline
│   │   └── RecommendationCard
│   ├── CenterColumn (Player Workspace)
│   │   ├── FilterBar
│   │   ├── PlayerTable
│   │   │   ├── PlayerRow
│   │   │   │   ├── VBDBar
│   │   │   │   ├── AvailabilityBand
│   │   │   │   ├── DecisionPill
│   │   │   │   └── ExpandedDetails
│   │   │   └── TableHeader
│   │   └── Pagination
│   └── RightColumn (Analytics)
│       ├── ScatterPlot
│       ├── PositionScarcity
│       ├── QuickPicksList
│       └── SimulationWidget
└── Modals
    ├── PlayerDetailModal
    ├── SettingsModal
    └── DraftHistoryModal
```

## 3. Implementation Phases

### Phase 1: MVP Foundation (Week 1)
**Goal**: Basic working interface with core functionality

#### Backend Setup
```python
# api/main.py
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

app = FastAPI()

@app.get("/api/players")
async def get_players(current_pick: int = 1):
    # Load and process data
    df = load_ranking_data()
    enhanced = calculate_probabilities(df, current_pick)
    return enhanced.to_dict('records')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Real-time draft updates
    pass
```

#### Frontend Foundation
```typescript
// src/App.tsx
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { DraftProvider } from './contexts/DraftContext'
import { Layout } from './components/Layout'

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <DraftProvider>
        <Layout />
      </DraftProvider>
    </QueryClientProvider>
  )
}
```

#### Core Components
```typescript
// src/components/PlayerTable.tsx
interface PlayerTableProps {
  players: Player[]
  onPlayerDraft: (player: Player) => void
}

export function PlayerTable({ players, onPlayerDraft }: PlayerTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        {/* Sortable headers */}
        {/* Player rows with VBD bars */}
      </table>
    </div>
  )
}
```

**Deliverables**:
- Working API with player data endpoint
- Basic three-column layout
- Sortable player table
- VBD ranking display
- Mark player as drafted functionality

### Phase 2: Probability System Integration (Week 2)
**Goal**: Integrate the 80% ESPN + 20% ADP probability system

#### Enhanced API Endpoints
```python
@app.get("/api/probabilities")
async def calculate_probabilities(
    current_pick: int,
    my_picks: List[int],
    drafted_players: List[str]
):
    # Implement probability calculations
    return probability_matrix

@app.post("/api/simulate")
async def simulate_picks(
    player_name: str,
    picks_until_next: int
):
    # Run Monte Carlo simulation
    return simulation_results
```

#### Probability Visualizations
```typescript
// src/components/AvailabilityBand.tsx
export function AvailabilityBand({ 
  probabilities,
  myPicks 
}: AvailabilityBandProps) {
  return (
    <div className="h-6 relative">
      {/* Sparkline with probability gradient */}
      <svg viewBox="0 0 100 20">
        {/* Path with color gradient */}
      </svg>
    </div>
  )
}
```

**Deliverables**:
- Real-time probability calculations
- Availability bands (sparklines)
- Decision score calculations
- Hover tooltips with simulations

### Phase 3: Interactive Visualizations (Week 3)
**Goal**: Add scatter plots, scarcity widgets, and advanced interactions

#### Scatter Plot Component
```typescript
// src/components/ScatterPlot.tsx
import * as d3 from 'd3'

export function ScatterPlot({ 
  players,
  xAxis = 'vbd_score',
  yAxis = 'probability'
}: ScatterPlotProps) {
  useEffect(() => {
    // D3 scatter plot implementation
    const svg = d3.select(svgRef.current)
    // Add axes, dots, hover interactions
  }, [players])
  
  return <svg ref={svgRef} />
}
```

#### Position Scarcity
```typescript
// src/components/PositionScarcity.tsx
export function PositionScarcity({ position }: { position: string }) {
  return (
    <div className="flex items-center gap-2">
      <Thermometer value={scarcityScore} />
      <span>{remainingPlayers} left</span>
    </div>
  )
}
```

**Deliverables**:
- Risk vs Reward scatter plot
- Position scarcity thermometers
- Quick picks panel
- Expandable player rows
- Keyboard shortcuts

### Phase 4: Production Features (Week 4)
**Goal**: Polish, performance, and deployment

#### Performance Optimizations
```typescript
// Virtualized table for large datasets
import { VirtualList } from '@tanstack/react-virtual'

// Memoized calculations
const memoizedProbabilities = useMemo(() => 
  calculateProbabilities(players, currentPick),
  [players, currentPick]
)
```

#### Mobile Responsiveness
```css
/* Responsive breakpoints */
@media (max-width: 1024px) {
  /* Stack columns vertically */
  .layout { grid-template-columns: 1fr; }
}

@media (max-width: 640px) {
  /* Simplified mobile view */
  .player-table { /* Horizontal scroll */ }
}
```

#### Deployment Setup
```dockerfile
# Dockerfile
FROM node:18 AS frontend-build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY --from=frontend-build /app/dist ./static
COPY api/ ./api
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0"]
```

**Deliverables**:
- WebSocket real-time updates
- Draft history tracking
- Export functionality
- Mobile responsive design
- Docker deployment
- Performance monitoring

## 4. Data Integration Plan

### Data Sources
```python
# data/loader.py
class DataLoader:
    def __init__(self):
        self.espn_path = "data/espn_projections_20250814.csv"
        self.adp_path = "data/fantasypros_adp_20250815.csv"
        self.vbd_path = "draft_cheat_sheet.csv"
    
    def load_and_merge(self):
        # Load all data sources
        # Merge on player name
        # Calculate composite rankings
        return merged_df
```

### Probability Calculations
```python
# core/probability.py
def compute_pick_probabilities(
    available_df: pd.DataFrame,
    espn_weight: float = 0.8,
    adp_weight: float = 0.2
) -> pd.Series:
    """80% ESPN + 20% ADP weighted system"""
    espn_scores = compute_softmax_scores(available_df['espn_rank'])
    adp_scores = compute_softmax_scores(available_df['adp_rank'])
    combined = espn_weight * espn_scores + adp_weight * adp_scores
    return combined / combined.sum()
```

### Real-time Updates
```python
# core/draft_manager.py
class DraftManager:
    def __init__(self):
        self.current_pick = 1
        self.drafted_players = set()
        self.subscribers = []
    
    async def mark_drafted(self, player_name: str):
        self.drafted_players.add(player_name)
        self.current_pick += 1
        await self.notify_subscribers()
```

## 5. Testing Strategy

### Unit Tests
```typescript
// src/components/__tests__/PlayerTable.test.tsx
describe('PlayerTable', () => {
  it('sorts by VBD rank by default', () => {})
  it('updates probabilities on pick change', () => {})
  it('filters by position correctly', () => {})
})
```

### Integration Tests
```python
# tests/test_api.py
def test_probability_calculation():
    response = client.get("/api/probabilities?current_pick=8")
    assert response.status_code == 200
    assert len(response.json()) > 0
```

### E2E Tests
```typescript
// e2e/draft-flow.spec.ts
test('complete draft workflow', async ({ page }) => {
  await page.goto('/')
  await page.click('[data-testid="player-row-1"]')
  await page.click('[data-testid="draft-button"]')
  // Assert player marked as drafted
})
```

### Performance Testing
- Load 500+ players without lag
- Instant sorting/filtering (< 100ms)
- Smooth animations (60fps)
- Memory usage < 100MB

## 6. Deployment Approach

### Development
```bash
# Start backend
cd api && uvicorn main:app --reload

# Start frontend
cd frontend && npm run dev
```

### Production Options

#### Option 1: Cloud Deployment
```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
  frontend:
    build: ./frontend
    ports:
      - "3000:80"
  nginx:
    image: nginx
    ports:
      - "80:80"
```

#### Option 2: Local Electron App
```javascript
// electron/main.js
const { app, BrowserWindow } = require('electron')

function createWindow() {
  const win = new BrowserWindow({
    width: 1600,
    height: 900,
    webPreferences: {
      nodeIntegration: false
    }
  })
  win.loadURL('http://localhost:3000')
}
```

#### Option 3: Static Export
```bash
# Build static version
npm run build
python -m http.server 8000 --directory dist
```

## 7. File Structure

```
fantasy-football-draft/
├── api/
│   ├── main.py
│   ├── core/
│   │   ├── probability.py
│   │   ├── draft_manager.py
│   │   └── data_loader.py
│   └── routers/
│       ├── players.py
│       └── draft.py
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Layout/
│   │   │   ├── PlayerTable/
│   │   │   ├── Visualizations/
│   │   │   └── common/
│   │   ├── contexts/
│   │   ├── hooks/
│   │   ├── utils/
│   │   └── App.tsx
│   ├── package.json
│   └── vite.config.ts
├── data/
│   └── [existing CSV files]
├── tests/
├── docker-compose.yml
├── Dockerfile
└── README.md
```

## 8. Key Implementation Details

### VBD Bar Component
```typescript
export function VBDBar({ value, max, delta }: VBDBarProps) {
  const percentage = (value / max) * 100
  return (
    <div className="relative h-8 bg-gray-100 rounded">
      <div 
        className="absolute h-full bg-gradient-to-r from-green-500 to-blue-500 rounded"
        style={{ width: `${percentage}%` }}
      />
      {delta && (
        <div className="absolute right-2 top-1 text-xs">
          {delta > 0 ? '+' : ''}{delta}
        </div>
      )}
    </div>
  )
}
```

### Decision Score Calculation
```python
def calculate_decision_score(row):
    """VBD Score × Probability at next pick"""
    vbd_score = row['custom_vbd']
    prob_next = row['prob_at_next_pick']
    return vbd_score * prob_next / 100
```

### Target Mode Implementation
```typescript
const [targetPlayer, setTargetPlayer] = useState<Player | null>(null)

function handleTargetMode(player: Player) {
  setTargetPlayer(player)
  // Highlight cells where player is likely available
  // Show probability path to target pick
}
```

## 9. Timeline & Milestones

| Week | Milestone | Deliverables |
|------|-----------|--------------|
| 1 | MVP Foundation | Basic interface, API, player table |
| 2 | Probability Integration | Live calculations, availability bands |
| 3 | Visualizations | Scatter plots, scarcity widgets |
| 4 | Production Ready | Testing, deployment, documentation |

## 10. Success Metrics

- **Performance**: < 100ms response time for all interactions
- **Accuracy**: Probability calculations match Excel reference
- **Usability**: Complete draft in < 2 hours with interface
- **Reliability**: Zero crashes during 3-hour draft session
- **Mobile**: Fully functional on tablet devices

## Next Steps

1. **Immediate**: Set up project structure and dependencies
2. **Day 1-3**: Implement Phase 1 MVP
3. **Day 4-7**: Add probability system
4. **Week 2**: Complete visualizations
5. **Week 3**: Testing and deployment prep
6. **Week 4**: Production deployment

This plan provides a complete roadmap from current Jupyter notebook prototype to production-ready web application with all requested features from the ChatGPT specification.