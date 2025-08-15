# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fantasy football draft analysis and visualization system that provides intelligent draft assistance through a minimal Jupyter-based interface. The system combines sophisticated VBD analysis with real-time draft tracking and AI-powered recommendations to give users a competitive edge during live drafts.

### **Current Implementation Status** ✅
- **Enhanced Draft Board**: Professional Jupyter interface with rich player tiles, value indicators, and position scarcity heat map
- **Advanced Intelligence Engine**: Multi-factor recommendation system considering positional need, scarcity, tier urgency, and round context
- **Real-time Visualization**: Interactive snake draft board with professional appearance suitable for group screencast
- **Clean Architecture**: Organized directory structure with proper separation of concerns
- **All Critical Bugs Fixed**: Scarcity calculations and player dropdown functionality working correctly

## Architecture & Data Flow

### Data Pipeline
```
FantasyPros → scripts/scrape_projections.py → CSV files → Analysis notebooks → Draft tools
```

1. **Data Collection**: `scripts/scrape_projections.py` fetches current season projections from FantasyPros for all positions
2. **Data Processing**: Jupyter notebooks calculate VBD scores, apply custom league scoring, and generate rankings
3. **Draft Tools**: Interactive systems for live draft assistance and post-draft analysis

### Directory Structure
```
├── config/                    # League configuration files
├── data/                      # Player data and projections
│   ├── projections/          # FantasyPros scraped data
│   └── rankings_YYYYMMDD.csv # Processed rankings
├── docs/                      # Documentation and planning files
├── notebooks/                 # Jupyter analysis and draft tools
├── scripts/                   # Data collection utilities
├── src/                       # Core Python modules
└── tmp/                       # Temporary/experimental files
```

### Key Components

**Configuration System** (`config/league-config.yaml`):
- League settings (14 teams, snake draft, roster requirements)
- Custom scoring system (0.04 pass yards, 0.1 rush/rec yards, 4pt pass TD, 6pt rush/rec TD)
- Positional requirements and maximums
- Replacement level calculations for VBD

**Core Data Files**:
- `data/projections/projections_*_YYYYMMDD.csv` - Position-specific player projections
- `data/rankings_YYYYMMDD.csv` - Processed rankings with fantasy points
- `draft_cheat_sheet.csv` - Final draft rankings with VBD scores
- `CSG Fantasy Football Sheet - 2025 v13.01.csv` - Comprehensive player database with ADP, rankings, tiers

**Analysis Notebooks** (`notebooks/`):
- `analyze_projections_executed.ipynb` - Main projection processing and VBD calculation
- `minimal_draft_board.ipynb` - **Enhanced**: Professional 3-panel draft interface with rich player tiles, value indicators, and AI recommendations
- `interactive_draft_board.ipynb` - Real-time draft tracking with team rosters and pick management  
- `draft_preparation.ipynb` - Pre-draft analysis and strategy planning
- `post_draft_analysis.ipynb` - Draft review and team evaluation

**Core Intelligence System** (`src/`):
- `draft_engine.py` - Advanced draft state management and AI recommendation engine with multi-factor analysis
- `data_processor.py` - Flexible data loading with column mapping for various CSV formats

**Utilities** (`scripts/`):
- `scrape_projections.py` - FantasyPros data collection with respectful rate limiting

## Common Development Tasks

### Data Refresh Workflow
```bash
# 1. Scrape latest projections
uv run python scripts/scrape_projections.py

# 2. Process projections and calculate VBD
uv run jupyter notebook notebooks/analyze_projections_executed.ipynb

# 3. Update draft cheat sheet (manual export from notebook)
```

### Running the Enhanced Draft Board
```bash
# Start Jupyter from project root
uv run jupyter notebook

# Open notebooks/minimal_draft_board.ipynb

# Update your draft position in Cell 1:
USER_TEAM_ID = 7      # Your team number (1-14)
USER_DRAFT_POSITION = 7  # Your draft slot

# Enhanced features:
# - Professional 3-panel interface with rich player tiles
# - Value/reach indicators (green borders = value, red = reach)
# - Position scarcity heat map for at-a-glance availability
# - Smart AI recommendations with detailed reasoning
# - One-click drafting with complete undo functionality
# - Auto-pick option for AI-powered selections
# - Professional appearance suitable for group screencast
```

### Environment Setup
```bash
# With uv (recommended - much faster):
uv sync

# Or traditional method:
pip install -r requirements.txt

# Ensure data directories exist
mkdir -p data/projections
```

**Why uv?** 
- **10-100x faster** dependency resolution and installation
- **Single command** replaces pip + venv workflow
- **Lock file** ensures consistent environments
- **Same interface** as npm/yarn for Python projects

## Key Design Patterns

### VBD Calculation
The system uses Value-Based Drafting where player values are calculated relative to replacement level players at each position. Replacement levels are configurable in `league-config.yaml` under the `replacement_level` section.

### Snake Draft Logic
Draft order is calculated dynamically based on round number (odd rounds go 1→14, even rounds go 14→1). This is implemented in the `DraftTracker` class with the `_generate_snake_order()` method.

### Position Scarcity Analysis
The system tracks remaining players by position and tier to identify scarcity situations. This involves:
- Tier break detection (when elite players at a position are exhausted)
- Supply vs demand calculations (remaining startable players vs teams needing that position)
- Opportunity cost analysis (value difference between drafting now vs waiting)

### Data Processing Pipeline
1. Raw projections are loaded from CSV files with position-specific stat columns
2. Custom league scoring is applied to calculate fantasy points
3. VBD scores are calculated using position-specific replacement levels
4. Players are ranked and tiered for draft recommendations

## Technical Notes

- All notebooks use pandas for data manipulation with numpy for calculations
- Interactive widgets use ipywidgets for draft board controls
- Visualizations use plotly for interactive charts and draft boards
- Draft state can be persisted using pickle for session continuity
- Web scraping uses BeautifulSoup with respectful delays (2 seconds between requests)

## Data Structures

**Player Records**: Include projections (passing/rushing/receiving stats), VBD scores, ADP data, positional rankings, bye weeks, and draft metadata.

**Draft State**: Tracks current pick number, round, team rosters, available players, and pick history for undo functionality.

**Team Analysis**: Calculates positional needs based on roster requirements, current roster construction, and remaining draft capital.

The system is designed to be league-configurable while maintaining sophisticated draft intelligence that considers positional scarcity, value opportunities, and roster construction efficiency.