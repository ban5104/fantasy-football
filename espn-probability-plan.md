# ESPN Draft Probability Visualization Plan

## Overview
Build a Jupyter notebook that visualizes the probability of each player being available at each draft position, similar to the Excel screenshot provided. Use ESPN rankings as the primary driver with configurable weighting for other factors.

## Data Inputs

### Primary Source: ESPN Projections CSV
- 300+ players in ESPN algorithm order (overall_rank)
- Player name, position, team, ESPN rank
- This becomes the baseline "expected draft order"

### Secondary Inputs
- FantasyPros ADP data (for consensus variance adjustments)

## Core Algorithm

### Base Probability Model
```
For each player at each draft position:
1. Calculate expected pick based on ESPN rank + team positional needs
2. Apply normal distribution around expected pick
3. Return probability player is still available
```

### Weighting System
- **ESPN Rank**: 80% weight (primary driver)
- **ADP Deviation**: 20% weight (accounts for community consensus)

## Visualization Approach

### Matrix Layout (like Excel screenshot)
- **Rows**: Players (sorted by ESPN rank)
- **Columns**: Draft positions (1-14 x number of rounds)
- **Colors**: Green (high availability) → Yellow → Red (low availability)
- **Values**: Percentage probability (0-100%)

### Interactive Features
- Filter by position
- Highlight your draft position
- Visual normal distribution charts for each player
- Mark drafted players as 0% (maintains consistent visualization structure)

## Implementation Steps

### Phase 1: Basic Probability Matrix
1. Load ESPN rankings CSV (300 players)
2. Calculate base probabilities using normal distribution
3. Create colored matrix visualization (like Excel screenshot)
4. Add visual normal distribution charts for learning

### Phase 2: Backend Integration
1. Accept drafted player updates from API calls
2. Mark drafted players as 0% (keeps DataFrame structure consistent)
3. Recalculate all remaining probabilities
4. Refresh matrix visualization automatically (simple value updates)

### Phase 3: Enhanced Visuals (Visual Learning Focus)
1. Individual player probability curves
2. Position-by-position distribution charts
3. Draft position "heat maps"
4. Interactive probability exploration tools

## Technical Stack
- **Pandas**: Data manipulation
- **NumPy/SciPy**: Probability calculations
- **Plotly/Matplotlib**: Matrix visualization
- **ipywidgets**: Interactive controls
- **Jupyter**: Development environment

## Key Files
- `espn_probability_matrix.ipynb` - Main visualization notebook
- `probability_engine.py` - Core calculation logic
- `data/espn_projections_20250814.csv` - ESPN player rankings
- `data/fantasypros_adp_20250815.csv` - ADP consensus data
- Integration with existing `draft_engine.py`

## Success Metrics
- Accurate probability predictions vs actual draft results
- Real-time updates under 1 second
- Clean, readable visualization matching Excel format
- Easy integration with existing draft tools