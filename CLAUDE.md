# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fantasy football draft analysis and visualization system that provides intelligent draft assistance through a minimal Jupyter-based interface. The system combines sophisticated VBD analysis with real-time draft tracking and AI-powered recommendations to give users a competitive edge during live drafts.

### **Current Implementation Status** ✅
- **Minimal Draft Board**: Complete Jupyter notebook interface with 3-panel layout (controls, visual board, AI recommendations)
- **Advanced Intelligence Engine**: Multi-factor recommendation system considering positional need, scarcity, tier urgency, and round context
- **Real-time Visualization**: Interactive snake draft board with position color-coding and live updates
- **Emergency Backup System**: Terminal-based draft tracker for ESPN API failures

## Architecture & Data Flow

### Data Pipeline
```
FantasyPros → scrape_projections.py → CSV files → Analysis notebooks → Draft tools
```

1. **Data Collection**: `scripts/scrape_projections.py` fetches current season projections from FantasyPros for all positions
2. **Data Processing**: Jupyter notebooks calculate VBD scores, apply custom league scoring, and generate rankings
3. **Draft Tools**: Interactive systems for live draft assistance and post-draft analysis

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

**Analysis Notebooks**:
- `notebooks/analyze_projections_executed.ipynb` - Main projection processing and VBD calculation
- `notebooks/minimal_draft_board.ipynb` - **PRIMARY**: Simple 3-panel draft interface with AI recommendations
- `notebooks/interactive_draft_board.ipynb` - Real-time draft tracking with team rosters and pick management  
- `notebooks/draft_preparation.ipynb` - Pre-draft analysis and strategy planning
- `notebooks/post_draft_analysis.ipynb` - Draft review and team evaluation

**Core Intelligence System**:
- `src/draft_engine.py` - Advanced draft state management and AI recommendation engine
- `src/data_processor.py` - Flexible data loading with column mapping for various CSV formats

**Emergency Backup System**:
- `backup_draft.py` - Terminal-based draft tracker for ESPN API failures
- `live_draft_tracker.py` - ESPN API integration for real-time draft monitoring

## Common Development Tasks

### Environment Setup

**IMPORTANT**: This project uses `uv` for dependency management but has specific setup requirements due to the non-standard package structure.

#### Method 1: UV Environment (Recommended)
```bash
# Install dependencies (fixed in pyproject.toml v2)
uv sync

# If sync fails, use pip mode:
uv pip install -r requirements_draft_board.txt
uv pip install pandas numpy pyyaml matplotlib plotly jupyter beautifulsoup4 requests lxml scikit-learn scipy ipywidgets pytest

# Ensure data directories exist
mkdir -p data/projections data/draft
```

#### Method 2: Traditional pip (Fallback)
```bash
# Use requirements file for streamlit app dependencies
pip install -r requirements_draft_board.txt
pip install pandas numpy pyyaml matplotlib plotly jupyter beautifulsoup4 requests lxml scikit-learn scipy ipywidgets pytest

# Ensure data directories exist
mkdir -p data/projections data/draft
```

#### Environment Verification
```bash
# Test that core imports work
export PATH="$(uv python find | head -n1 | xargs dirname):$PATH"
python3 -c "import pandas, numpy, yaml, matplotlib, plotly; print('✓ Core dependencies available')"

# Test project modules
python3 -c "import sys; sys.path.insert(0, '.'); import src.draft_engine, src.data_processor, backup_draft; print('✓ Project modules work')"
```

### Data Refresh Workflow
```bash
# 1. Scrape latest projections
uv run python scripts/scrape_projections.py

# 2. Process projections and calculate VBD
uv run jupyter notebook notebooks/analyze_projections_executed.ipynb

# 3. Update draft cheat sheet (manual export from notebook)
```

### Testing

**IMPORTANT**: Tests require proper Python environment setup due to dependency issues.

#### Method 1: Using UV Environment
```bash
# Run tests with proper UV environment
export PATH="$(uv python find | head -n1 | xargs dirname):$PATH"
PYTHONPATH=. python3 run_tests.py

# Run specific test file
PYTHONPATH=. python3 -m pytest tests/test_backup_draft.py -v

# Test backup draft functionality directly
PYTHONPATH=. python3 -c "
import sys
sys.path.insert(0, '.')
from backup_draft import BackupDraftTracker
tracker = BackupDraftTracker()
print('✓ BackupDraftTracker works')
"
```

#### Method 2: Manual Test Validation
```bash
# If formal tests fail, manually verify core functionality:
export PATH="$(uv python find | head -n1 | xargs dirname):$PATH"

# Test 1: Core imports
python3 -c "
import sys; sys.path.insert(0, '.')
import src.draft_engine, src.data_processor
from backup_draft import BackupDraftTracker
print('✓ All modules import successfully')
"

# Test 2: Basic functionality
python3 -c "
import sys; sys.path.insert(0, '.')
from backup_draft import BackupDraftTracker
tracker = BackupDraftTracker()
print('✓ BackupDraftTracker instantiation works')
"
```

#### Common Test Issues & Solutions
- **ModuleNotFoundError**: Ensure proper environment activation with `export PATH="$(uv python find | head -n1 | xargs dirname):$PATH"`
- **Import errors**: Always use `PYTHONPATH=.` to include current directory  
- **Dependency issues**: Re-run the environment setup commands above

### Running Draft Tools

**Primary Draft Board**:
```bash
# Start Jupyter and open the minimal draft board  
uv run jupyter notebook notebooks/minimal_draft_board.ipynb

# Update your draft position in Cell 1:
USER_TEAM_ID = 7      # Your team number (1-14)
USER_DRAFT_POSITION = 7  # Your draft slot
```

**Emergency Backup System**:
```bash
# When ESPN API fails during live draft
python backup_draft.py

# Reset for new draft (delete resume file)
rm data/draft/draft_picks_latest.csv
```

**ESPN API Monitoring**:
```bash
# Monitor live ESPN draft
python live_draft_tracker.py --monitor

# Check current draft status
python live_draft_tracker.py --status
```

## Key Design Patterns

### VBD Calculation
The system uses Value-Based Drafting where player values are calculated relative to replacement level players at each position. Replacement levels are configurable in `config/league-config.yaml` under the `replacement_level` section.

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

## Troubleshooting

### Environment Setup Issues

**Problem**: `uv sync` fails with "Unable to determine which files to ship inside the wheel"
**Solution**: This was fixed in pyproject.toml v2 by adding proper package configuration. If still failing, use fallback method: `uv pip install -r requirements_draft_board.txt`

**Problem**: `ModuleNotFoundError` when running tests or scripts
**Solution**: 
```bash
# Ensure proper UV environment
export PATH="$(uv python find | head -n1 | xargs dirname):$PATH"
# Always include project root in Python path
PYTHONPATH=. python3 your_script.py
```

**Problem**: Tests run but can't find pandas/numpy
**Solution**: Dependencies are in UV environment, not system Python:
```bash
# Verify UV environment is active
uv pip list | grep pandas
# Use UV's Python explicitly
export PATH="$(uv python find | head -n1 | xargs dirname):$PATH"
```

### Import Issues

**Problem**: `from src.draft_engine import DraftEngine` fails
**Solution**: The modules don't export classes with matching names. Use:
```python
import src.draft_engine as draft_engine
import src.data_processor as data_processor  
```

**Problem**: Jupyter notebooks can't find project modules
**Solution**: Add project root to Python path:
```python
import sys
sys.path.insert(0, '.')
```

### Development Workflow Issues

**Problem**: Tests pass but linting/formatting commands missing
**Solution**: This project doesn't have pre-configured linting. For PR preparation, focus on manual testing and core functionality verification.

**Problem**: Multiple dependency files confusing
**Solution**: Use this hierarchy:
1. `uv sync` (if pyproject.toml works)
2. `uv pip install -r requirements_draft_board.txt` (most reliable)
3. Manual pip install of individual packages (last resort)

## Important File Locations

**Primary Workflow Files**:
- `notebooks/minimal_draft_board.ipynb` - Main draft interface
- `scripts/scrape_projections.py` - Data collection
- `backup_draft.py` - Emergency backup system

**Configuration**:
- `config/league-config.yaml` - All league settings and scoring
- `pyproject.toml` - Dependencies and project metadata

**Testing**:
- `tests/test_backup_draft.py` - Backup system tests
- `run_tests.py` - Test runner

**Data Sources**:
- `CSG Fantasy Football Sheet - 2025 v13.01.csv` - Master player database
- `data/projections/` - Scraped projection data
- `data/draft/` - Draft state and picks

## Development Environment

**Package Management**: Uses `uv` for fast dependency resolution (10-100x faster than pip)
**Python Version**: Requires Python >=3.8
**Key Dependencies**: pandas, numpy, plotly, jupyter, beautifulsoup4, requests
**Architecture**: Jupyter-based analysis with Python backend modules for draft logic