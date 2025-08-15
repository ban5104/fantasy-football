# Fantasy Football Draft Analysis & Visualization System

A sophisticated fantasy football draft assistance system that combines VBD (Value-Based Drafting) analysis with real-time draft tracking and AI-powered recommendations.

## 🏈 What This Does

**Pre-Draft:**
- Scrapes current projections from FantasyPros
- Calculates VBD scores based on your league's custom scoring
- Generates position-based rankings and tier analysis
- Creates draft cheat sheets with intelligent recommendations

**During Draft:**
- Real-time draft board with 3-panel interface
- AI recommendations considering scarcity, tier breaks, and roster needs
- Snake draft visualization with position color-coding
- Emergency backup system when ESPN API fails

**Post-Draft:**
- Team analysis and draft grading
- Roster construction evaluation
- Trade and waiver wire recommendations

## 🚀 Quick Start

### 1. Setup Environment

**⚠️ IMPORTANT**: Environment setup can be tricky due to project structure. Follow these steps carefully:

```bash
# Method 1: UV (Recommended - fixed in v2)
uv sync

# Method 2: If sync fails, use pip mode
uv pip install -r requirements_draft_board.txt
uv pip install pandas numpy pyyaml matplotlib plotly jupyter beautifulsoup4 requests lxml scikit-learn scipy ipywidgets pytest

# Verify setup works
export PATH="$(uv python find | head -n1 | xargs dirname):$PATH"
python3 -c "import pandas, numpy; print('✓ Dependencies ready')"
```

**Common Issues**: See [Troubleshooting](#troubleshooting) section below for environment problems.

### 2. Configure Your League
Edit `config/league-config.yaml` with your league settings:
- Scoring system (passing/rushing/receiving points)
- Roster requirements (14 teams, snake draft, etc.)
- Team names and draft preferences

### 3. Get Latest Data
```bash
# Scrape current season projections
uv run python scripts/scrape_projections.py

# Process and calculate VBD scores
uv run jupyter notebook notebooks/analyze_projections_executed.ipynb
```

### 4. Draft Day
```bash
# Primary: Interactive draft board
uv run jupyter notebook notebooks/minimal_draft_board.ipynb

# Backup: If ESPN API fails
python backup_draft.py
```

## 📊 System Architecture

```
FantasyPros → scrape_projections.py → CSV files → Jupyter Analysis → Draft Tools
```

**Data Flow:**
1. **Collection**: Scrape projections from FantasyPros
2. **Processing**: Apply custom scoring, calculate VBD
3. **Analysis**: Generate rankings, tiers, and recommendations
4. **Draft**: Real-time assistance with intelligent recommendations

## 🔧 Key Components

### Draft Tools
- **`notebooks/minimal_draft_board.ipynb`** - Primary draft interface
- **`backup_draft.py`** - Emergency terminal-based tracker
- **`live_draft_tracker.py`** - ESPN API integration

### Data Processing
- **`scripts/scrape_projections.py`** - Data collection
- **`src/draft_engine.py`** - Draft logic and AI recommendations
- **`src/data_processor.py`** - Flexible CSV handling

### Configuration
- **`config/league-config.yaml`** - League settings and scoring
- **`CSG Fantasy Football Sheet - 2025 v13.01.csv`** - Player database

## 📁 Directory Structure

```
├── config/              # League configuration
├── data/
│   ├── projections/     # Scraped projection data
│   └── draft/          # Draft state and picks
├── notebooks/          # Jupyter analysis interfaces
├── scripts/            # Data collection utilities
├── src/               # Core Python modules
├── tests/             # Test suite
└── docs/              # Additional documentation
```

## 🎯 Draft Day Workflow

### Normal Operation (ESPN API Working)
1. Open `notebooks/minimal_draft_board.ipynb`
2. Set your team ID and draft position
3. Use 3-panel interface for picks, visualization, and AI recommendations
4. System auto-updates with live draft data

### Emergency Backup (ESPN API Failed)
1. Run `python backup_draft.py`
2. Enter player names as picks happen
3. Auto-saves in ESPN-compatible format
4. Resume analysis with backup data

### Reset for New Draft
```bash
# Archive old draft and start fresh
rm data/draft/draft_picks_latest.csv
```

## 🧠 Intelligence Features

**VBD Analysis**: Values players relative to replacement level at each position
**Scarcity Detection**: Identifies when elite players at positions are depleting
**Tier Analysis**: Groups players by value gaps for strategic timing
**Roster Construction**: Optimizes picks based on remaining needs and draft capital

## ⚙️ Development

### Testing

**⚠️ IMPORTANT**: Testing requires proper environment setup.

```bash
# Setup environment first
export PATH="$(uv python find | head -n1 | xargs dirname):$PATH"

# Run all tests
PYTHONPATH=. python3 run_tests.py

# Test backup system
PYTHONPATH=. python3 backup_draft.py

# Manual verification if tests fail
python3 -c "
import sys; sys.path.insert(0, '.')
import src.draft_engine, src.data_processor
from backup_draft import BackupDraftTracker
print('✓ All core modules working')
"
```

### Data Updates
```bash
# Refresh projections (typically weekly during season)
uv run python scripts/scrape_projections.py
```

### Environment
- **Python**: 3.8+
- **Key Dependencies**: pandas, numpy, plotly, jupyter, beautifulsoup4
- **Package Manager**: uv (recommended) or pip

## 📋 League Configuration

The system supports 14-team snake drafts with custom scoring. Key configuration areas:

- **Scoring**: 0.04 pass yards, 0.1 rush/rec yards, 4pt pass TD, 6pt rush/rec TD
- **Roster**: QB, 2RB, 2WR, TE, FLEX, K, DST + 7 bench
- **Replacement Levels**: Customizable for VBD calculations

## 🔄 Workflow Integration

**Pre-Season**: Update projections → Calculate VBD → Generate cheat sheets
**Draft Day**: Live recommendations → Pick tracking → Real-time analysis
**Season**: Waiver analysis → Trade evaluation → Lineup optimization

## 🔧 Troubleshooting

### Environment Setup Issues

**Problem**: `uv sync` fails with "Unable to determine which files to ship"  
**Solution**: Fixed in pyproject.toml v2. If still failing, use: `uv pip install -r requirements_draft_board.txt`

**Problem**: `ModuleNotFoundError` when running scripts  
**Solution**: 
```bash
# Always use UV's Python environment
export PATH="$(uv python find | head -n1 | xargs dirname):$PATH"
# Include project root in Python path
PYTHONPATH=. python3 your_script.py
```

**Problem**: Tests can't find pandas/numpy  
**Solution**: Dependencies are in UV environment, not system Python:
```bash
# Verify UV has packages
uv pip list | grep pandas
# Use UV's Python explicitly
export PATH="$(uv python find | head -n1 | xargs dirname):$PATH"
```

### Import Issues

**Problem**: Can't import from `src/` modules  
**Solution**: 
```python
# In Python scripts or notebooks
import sys
sys.path.insert(0, '.')
import src.draft_engine as draft_engine
```

**Problem**: Jupyter can't find project modules  
**Solution**: Add to first cell:
```python
import sys
sys.path.insert(0, '.')
```

### Development Issues

**Problem**: Which dependency file to use?  
**Solution**: Priority order:
1. `uv sync` (if pyproject.toml works)
2. `uv pip install -r requirements_draft_board.txt` (most reliable)  
3. Manual package installation (last resort)

**Problem**: Tests pass but linting fails  
**Solution**: This project uses manual validation. Focus on core functionality tests.

## 📖 Additional Documentation

- **`CLAUDE.md`** - Technical guidance for Claude Code development (includes detailed troubleshooting)
- **`docs/`** - Detailed implementation guides and enhancement plans
- **`BACKUP_DRAFT_QUICK_GUIDE.md`** - Emergency backup system reference

---

*Built for competitive fantasy football managers who want data-driven draft advantages with reliable backup systems.*