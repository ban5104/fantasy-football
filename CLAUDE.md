# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Core Pipeline
```bash
# Full automated pipeline (scraping → scoring → VBD → rankings)
python scripts/daily_update.py

# Interactive analysis workflow
jupyter notebook notebooks/

# Install dependencies
pip install -r requirements.txt
```

### Data Pipeline Steps (Manual)
```bash
# 1. Data collection
jupyter notebook notebooks/01_data_collection/01_scrape_projections.ipynb

# 2. Basic analysis  
jupyter notebook notebooks/02_analysis/02_basic_rankings.ipynb

# 3. VBD calculations
jupyter notebook notebooks/02_analysis/03_vbd_calculations.ipynb
```

## Architecture Overview

### Data Flow Pipeline
```
FantasyPros → Raw Projections → Fantasy Points → VBD Calculations → Final Rankings
     ↓              ↓               ↓                ↓                    ↓
scraping.py → data/raw/ → scoring.py → vbd.py → data/output/
```

### Core Modules
- **`src/scraping.py`** - Web scraping FantasyPros for 6 positions (QB/RB/WR/TE/K/DST)
- **`src/scoring.py`** - Converts raw stats to fantasy points using league-specific scoring
- **`src/vbd.py`** - Core VBD calculations (VOLS, VORP, BEER, Blended methods)  
- **`src/utils.py`** - Data validation, logging, file I/O utilities

### Configuration System
- **Central config**: `config/league-config.yaml` drives all calculations
- **League settings**: 14 teams, specific roster slots, detailed scoring rules
- **VBD weights**: BEER (50%), VORP (25%), VOLS (25%) for blended rankings
- **Position baselines**: Calculated dynamically from league roster requirements

### VBD Methods Implementation
- **VOLS** (Value Over Like Starters): `baseline = teams × starters`
- **VORP** (Value Over Replacement): `baseline = teams × (starters + 1)`  
- **BEER** (Best Eleven Every Round): `baseline = teams × (starters + 0.5)`
- **Blended**: Weighted combination optimized for balanced draft strategy

### Notebook Organization
- **`01_data_collection/`** - Scraping and data validation
- **`02_analysis/`** - Fantasy scoring and VBD calculations
- **`03_insights/`** - Draft strategy and positional analysis
- **`99_archive/`** - Historical/experimental notebooks

### Data States
- **`data/raw/`** - Timestamped projections by position from FantasyPros
- **`data/output/`** - Final rankings by VBD method (top 300 players)
- **`data/`** - Log files and external reference data

### VBD Output Files (data/output/)
**Primary Custom Rankings:**
- **`vbd_rankings_top300_{YYYYMMDD}.csv`** - Main blended VBD rankings (50% BEER + 25% VORP + 25% VOLS)

**Individual VBD Methods:**
- **`rankings_vbd_beer_top300_{YYYYMMDD}.csv`** - BEER method (aggressive drafting)
- **`rankings_vbd_vorp_top300_{YYYYMMDD}.csv`** - VORP method (balanced approach)  
- **`rankings_vbd_vols_top300_{YYYYMMDD}.csv`** - VOLS method (conservative drafting)

**Supporting Files:**
- **`rankings_{YYYYMMDD}.csv`** - Basic fantasy points rankings (no VBD)
- **`rankings_top300_{YYYYMMDD}.csv`** - Top 300 by fantasy points only
- **`draft_cheat_sheet.csv`** - Draft preparation summary

### Automation Architecture
- **`scripts/daily_update.py`** - Production pipeline with error handling
- **Cron ready**: Designed for `0 6 * * * cd /path && python scripts/daily_update.py`
- **Data validation**: Built-in quality checks before VBD calculations
- **Comprehensive logging**: All operations logged to `fantasy_analysis.log`

## Key Development Patterns

### Configuration Loading
```python
from src.scoring import load_league_config
config = load_league_config()  # Loads config/league-config.yaml
```

### VBD Calculation Flow
```python
# Standard pattern used throughout codebase
df = calculate_fantasy_points_vectorized(df, config)
df_vbd = calculate_all_vbd_methods(df, config) 
top_300 = get_top_players_by_vbd(df_vbd, method='VBD_BLENDED', top_n=300)
```

### Error Handling
- All modules use comprehensive try/catch with logging
- VBD calculations include edge case handling for insufficient players
- Data validation before expensive calculations

## Important Notes

### Testing
- **No formal unit tests** - System relies on runtime validation
- **Data quality checks** in `utils.validate_data_quality()`
- **VBD validation** in `daily_update.py` before file output

### League Customization
- Modify `config/league-config.yaml` for different league settings
- VBD baselines auto-calculate from roster requirements
- Scoring rules handle complex scenarios (e.g., defensive touchdowns)

### Performance
- **Vectorized operations** for fast pandas calculations
- **Position-based processing** allows parallel VBD calculations
- **Top 300 focus** optimizes for draft-relevant players only