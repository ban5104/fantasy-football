# Minimal Jupyter Draft Board - Architecture Plan

## What We're Building
A single Jupyter notebook that provides everything needed for draft day: pick tracking, visual draft board, smart recommendations, and real-time analysis - all with minimal complexity.

## Simplest Approach
Leverage the existing `draft_engine.py` intelligence but simplify the interface to pure ipywidgets. Use a 3-panel layout: draft controls (left), draft board visualization (center), and recommendations/analysis (right). Everything updates live with each pick.

## Core Components

### 1. Data Layer (Minimal Setup)
```python
# Single initialization cell
import draft_engine as de
import pandas as pd
import yaml

# Load once, use everywhere
config = yaml.safe_load(open('config/league-config.yaml'))
players_df = pd.read_csv('draft_cheat_sheet.csv')
draft_state = de.DraftState(config, user_team_id=1, draft_position=1)
intelligence = de.DraftIntelligence(config, players_df)
```

### 2. Widget Layout (3-Panel Design)

```
┌─────────────────┬──────────────────────┬───────────────────┐
│  DRAFT CONTROLS │   VISUAL BOARD       │  RECOMMENDATIONS  │
│                 │                      │                   │
│  Team: [▼]      │  R1: ●●●●●●●●●●●●●● │  Top Picks:       │
│  Player: [▼]    │  R2: ●●●●●●●●●●●●●● │  1. CMC (RB) 9.5  │
│  [DRAFT]        │  R3: ●●●○○○○○○○○○○○ │  2. Hill (WR) 8.9 │
│                 │                      │                   │
│  [UNDO LAST]    │  ● QB  ● RB  ● WR   │  Scarcity Alert:  │
│  [AUTO PICK]    │  ● TE  ○ Empty      │  - Last Tier 1 RB │
│                 │                      │                   │
│  Your Team:     │  Next: Team 4        │  Your Next: Pick 28│
│  QB: -          │  Round 3, Pick 3     │  (14 picks away)  │
│  RB: McCaffrey  │                      │                   │
│  WR: Hill       │                      │  Position Need:   │
│  TE: -          │                      │  RB ●● WR ●● TE ● │
└─────────────────┴──────────────────────┴───────────────────┘
```

### 3. Implementation Structure

#### Cell 1: Setup & Data Load
```python
# All imports and data loading
# ~20 lines
```

#### Cell 2: Core Draft Manager
```python
class MinimalDraftManager:
    def __init__(self, draft_state, intelligence):
        self.state = draft_state
        self.intel = intelligence
        self.setup_widgets()
    
    def make_pick(self, player_name):
        # Find player, make pick, refresh
        # ~15 lines
    
    def setup_widgets(self):
        # Create all widgets
        # ~30 lines
        
    def refresh_display(self):
        # Update all 3 panels
        # ~20 lines
```

#### Cell 3: Visual Draft Board
```python
def create_draft_board(draft_state):
    # Simple matplotlib grid showing picks
    # Color-coded by position
    # ~25 lines
```

#### Cell 4: Smart Recommendations Panel
```python
def update_recommendations(draft_state, intelligence):
    # Get top 5 recommendations
    # Show with reasoning
    # Highlight scarcity/tier breaks
    # ~20 lines
```

#### Cell 5: Launch Interface
```python
# Create and display the complete interface
manager = MinimalDraftManager(draft_state, intelligence)
display(manager.layout)
```

## Widget Specifications

### Draft Controls (Left Panel)
- **Team Dropdown**: Current team on clock (auto-updates)
- **Player Search**: Type to filter, shows top 10 matches
- **Draft Button**: Big, green, one-click drafting
- **Undo Button**: Simple undo last pick
- **Auto Pick**: Uses intelligence.get_recommendations()[0]
- **Your Roster**: Live update of your team

### Visual Board (Center Panel)
- **Snake Grid**: 14 columns × 16 rows
- **Color Coding**: Position-based colors
- **Hover Info**: Player name, team, stats
- **Current Pick**: Highlighted/animated
- **Your Picks**: Bordered differently

### Recommendations (Right Panel)
- **Top 5 Picks**: With scores and 1-line reasoning
- **Scarcity Alerts**: "Last Tier 1 RB", "2 elite WRs left"
- **Your Next Pick**: Countdown and round info
- **Position Needs**: Visual indicators (●=need, ○=filled)
- **Tier Breaks**: Warning when approaching tier drops

## Data Flow

```
User Action → Draft State Update → Three Updates:
                                    ├→ Refresh Board
                                    ├→ Update Recommendations  
                                    └→ Update Your Roster

Each update is independent and fast (<100ms)
```

## Key Simplifications

### What We're Doing:
1. **Single notebook file** - No external apps or multiple files
2. **Reuse draft_engine.py** - Don't rewrite intelligence
3. **Simple widgets** - Dropdowns, buttons, output areas
4. **Matplotlib for board** - Simple, works everywhere
5. **Auto-refresh everything** - No manual update buttons

### What We're NOT Doing:
1. **No database** - Everything in memory
2. **No authentication** - Single user
3. **No network features** - Local only
4. **No advanced charts** - Just the essentials
5. **No configuration UI** - Use existing YAML

## Performance Optimizations

1. **Cached Recommendations**: Only recalculate when picks change
2. **Incremental Board Updates**: Only redraw changed cells
3. **Debounced Search**: Wait 300ms after typing stops
4. **Lazy Position Filters**: Calculate on-demand

## Time Estimate: 45 minutes to implement

## Quick Start Code

```python
# Cell 1: Complete Working Draft Board
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from draft_engine import DraftState, DraftIntelligence
import pandas as pd
import yaml

# Load data
config = yaml.safe_load(open('config/league-config.yaml'))
players_df = pd.read_csv('draft_cheat_sheet.csv')

# Initialize
draft = DraftState(config, user_team_id=1, draft_position=1)
intel = DraftIntelligence(config, players_df)

# Create UI
team_dropdown = widgets.Dropdown(description='Team:')
player_dropdown = widgets.Dropdown(description='Player:', options=[])
draft_btn = widgets.Button(description='DRAFT', button_style='success')
output = widgets.Output()

def refresh():
    # Update dropdowns
    team_dropdown.value = draft.get_team_on_clock()
    available = [p for p in intel.players_dict.values() 
                 if p.id not in draft.get_available_player_ids()]
    player_dropdown.options = [(p.name, p.id) for p in available[:20]]
    
    # Update display
    with output:
        clear_output()
        recs = intel.get_recommendations(draft, 5)
        for player, score, reason in recs:
            print(f"{player.name} ({player.position}) - {reason}")

def on_draft(b):
    player = intel.players_dict[player_dropdown.value]
    draft.make_pick(draft.get_team_on_clock(), player)
    refresh()

draft_btn.on_click(on_draft)
display(widgets.VBox([team_dropdown, player_dropdown, draft_btn, output]))
refresh()
```

## Next Steps

1. Test with mock draft (5 min)
2. Add visual board with matplotlib (10 min)  
3. Enhance recommendations display (10 min)
4. Add undo/save functionality (10 min)
5. Polish layout with HBox/VBox (10 min)

## Success Metrics

- **Pick Speed**: < 5 seconds to find and draft a player
- **Information Density**: See picks, recommendations, and roster without scrolling
- **Accuracy**: Never miss tier breaks or position runs
- **Simplicity**: New user can draft immediately without training

---

**Remember: The best draft tool is the one you'll actually use on draft day. Keep it simple, keep it fast, keep it effective.**