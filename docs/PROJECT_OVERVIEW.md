# Project Overview

## 🎯 System Purpose

Advanced fantasy football draft assistance system designed for competitive 14-team snake drafts. Combines sophisticated VBD analysis with real-time draft tracking and AI-powered recommendations.

## ⚡ Quick Reference

### Essential Commands
```bash
# Setup
uv sync

# Data refresh
uv run python scripts/scrape_projections.py

# Primary draft interface
uv run jupyter notebook notebooks/minimal_draft_board.ipynb

# Emergency backup
python backup_draft.py

# Testing
python run_tests.py
```

### Key Files
- **Primary Interface**: `notebooks/minimal_draft_board.ipynb`
- **Emergency Backup**: `backup_draft.py`
- **Configuration**: `config/league-config.yaml`
- **Data Collection**: `scripts/scrape_projections.py`

## 🏗️ Architecture Summary

**Data Pipeline**: FantasyPros → CSV → Analysis → Draft Tools
**Draft Logic**: VBD calculation + Scarcity analysis + AI recommendations
**Interfaces**: Jupyter notebooks (primary) + Terminal backup (emergency)
**Configuration**: YAML-based league settings with custom scoring

## 🔧 Development Stack

- **Python 3.8+** with pandas, numpy, plotly
- **Jupyter** for interactive analysis
- **uv** for fast package management
- **BeautifulSoup** for web scraping
- **YAML** configuration system

## 📊 Key Features

### Pre-Draft
- FantasyPros projection scraping
- Custom scoring application
- VBD score calculation
- Cheat sheet generation

### Live Draft
- 3-panel draft interface
- Real-time recommendations
- Snake draft visualization
- Emergency backup system

### Intelligence
- Position scarcity detection
- Tier break analysis
- Roster construction optimization
- Value opportunity identification

## 🎮 Usage Patterns

### Normal Draft Day
1. Update projections
2. Configure league settings
3. Open minimal draft board
4. Set team ID and position
5. Follow AI recommendations

### Emergency Situation
1. ESPN API fails
2. Switch to `backup_draft.py`
3. Manual player entry
4. Auto-save in ESPN format
5. Resume analysis seamlessly

## 📁 Project Structure

```
├── README.md              # Main project guide
├── CLAUDE.md              # Technical documentation
├── PROJECT_OVERVIEW.md    # This file
├── config/                # League configuration
├── data/                  # Projections and draft data
├── notebooks/             # Analysis interfaces
├── scripts/               # Utilities
├── src/                   # Core modules
├── tests/                 # Test suite
└── docs/                  # Detailed documentation
```

## 🔄 Typical Workflow

**Weekly**: Update projections → Refresh VBD calculations
**Pre-Draft**: Configure league → Generate cheat sheets → Test systems
**Draft Day**: Launch interface → Follow recommendations → Track picks
**Post-Draft**: Analyze results → Plan waiver moves

## 🛡️ Reliability Features

- **Auto-save**: Prevents data loss during crashes
- **Resume capability**: Continues from interruption points
- **Backup system**: Terminal interface when APIs fail
- **Error handling**: Graceful degradation and recovery
- **Testing suite**: Validates core functionality

## 📈 Success Metrics

- **Sub-15 second** pick selection time
- **Zero crashes** during live draft
- **Professional appearance** for group screencast
- **85%+ accuracy** in AI recommendations
- **Seamless integration** between primary and backup systems