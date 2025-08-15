# Fantasy Football Draft Spreadsheet

A comprehensive data science pipeline for fantasy football draft preparation using FantasyPros projections and Value Based Drafting (VBD) analysis.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Scrape latest projections
python scripts/daily_update.py

# Analyze in Jupyter
jupyter notebook notebooks/
```

## 📁 Project Structure

```
fantasy-football-draft-spreadsheet/
├── notebooks/                  # Organized analysis notebooks
│   ├── 01_data_collection/     # Data scraping and validation
│   ├── 02_analysis/           # Core analysis and VBD calculations
│   ├── 03_insights/           # Draft strategy and insights
│   └── 99_archive/            # Old/experimental notebooks
├── src/                       # Reusable Python modules
│   ├── scraping.py           # FantasyPros scraping utilities
│   ├── scoring.py            # Fantasy scoring calculations
│   ├── vbd.py               # Value Based Drafting methods
│   └── utils.py             # General utilities
├── scripts/                  # Automation scripts
│   └── daily_update.py      # Full pipeline automation
├── data/
│   ├── raw/                 # Scraped projection data
│   ├── processed/           # Cleaned and transformed data
│   └── output/              # Final rankings and analysis
├── config/
│   └── league-config.yaml   # League settings and scoring
└── docs/                    # Documentation and research
```

## 🏈 Analysis Workflow

### 1. Data Collection
- **Notebook**: `01_data_collection/01_scrape_projections.ipynb`
- **Purpose**: Scrape current season projections from FantasyPros
- **Output**: Raw projection CSVs in `data/raw/`

### 2. Basic Analysis
- **Notebook**: `02_analysis/02_basic_rankings.ipynb`  
- **Purpose**: Calculate fantasy points using league scoring
- **Output**: Ranked players with fantasy points

### 3. VBD Calculations
- **Notebook**: `02_analysis/03_vbd_calculations.ipynb`
- **Purpose**: Apply multiple VBD methods (VOLS, VORP, BEER, Blended)
- **Output**: Top 300 players ranked by draft value

### 4. Draft Strategy
- **Notebook**: `03_insights/03_draft_strategy.ipynb`
- **Purpose**: Draft board analysis and positional strategy
- **Output**: Draft recommendations and insights

## 🎯 VBD Methods

The project implements four industry-standard VBD calculation methods:

- **VOLS** (Value Over Like Starters): `baseline = teams × starters`
- **VORP** (Value Over Replacement Player): `baseline = teams × (starters + 1)`  
- **BEER** (Best Eleven Every Round): `baseline = teams × (starters + 0.5)`
- **Blended**: Weighted combination (50% BEER + 25% VORP + 25% VOLS)

## ⚙️ Configuration

Edit `config/league-config.yaml` to match your league settings:

```yaml
basic_settings:
  teams: 14
  draft_type: "snake"

roster:
  roster_slots:
    QB: 1
    RB: 2
    WR: 2
    TE: 1
    FLEX: 1
    K: 1
    DST: 1

scoring:
  passing:
    yards: 0.04    # 1 point per 25 yards
    td: 4
    int: -1
  rushing:
    yards: 0.1     # 1 point per 10 yards
    td: 6
  # ... more scoring settings
```

## 🔄 Git Workflow

This project uses feature branches for development:

```bash
# Create feature branch
git checkout -b feature/new-analysis

# Work on focused notebooks
# Commit frequently with clear messages

# Merge when complete
git checkout main
git merge feature/new-analysis
```

## 📊 Automation

### Daily Updates
```bash
# Manual update
python scripts/daily_update.py

# Automated via cron (daily at 6 AM)
0 6 * * * cd /path/to/project && python scripts/daily_update.py
```

### Output Files (data/output/)

**🎯 Primary Custom VBD Rankings:**
- **`vbd_rankings_top300_YYYYMMDD.csv`** - **Main blended rankings** (50% BEER + 25% VORP + 25% VOLS)

**📊 Individual VBD Methods:**
- **`rankings_vbd_beer_top300_YYYYMMDD.csv`** - BEER method (aggressive drafting strategy)
- **`rankings_vbd_vorp_top300_YYYYMMDD.csv`** - VORP method (balanced approach)  
- **`rankings_vbd_vols_top300_YYYYMMDD.csv`** - VOLS method (conservative strategy)

**📈 Reference Rankings:**
- `rankings_YYYYMMDD.csv` - Basic fantasy points rankings (no VBD)
- `rankings_top300_YYYYMMDD.csv` - Top 300 by fantasy points only
- `draft_cheat_sheet.csv` - Draft preparation summary

> **💡 Tip**: Start with `vbd_rankings_top300_YYYYMMDD.csv` for your main draft board. Use individual method files to adjust strategy based on draft position and league tendencies.

## 🛠️ Development

### Adding New Analysis
1. Create focused notebook in appropriate directory
2. Extract reusable code to `src/` modules
3. Add tests and documentation
4. Use feature branches for development

### Notebook Organization
- **One purpose per notebook** - easier debugging and sharing
- **Clear naming convention** - numbered by workflow order
- **Modular code** - extract functions to `src/` modules
- **Archive old notebooks** - keep `99_archive/` clean

## 📈 Key Features

- ✅ **Automated data collection** from FantasyPros
- ✅ **Multiple VBD methods** for comprehensive analysis  
- ✅ **Vectorized calculations** for fast processing
- ✅ **Top 300 player focus** for draft-relevant analysis
- ✅ **Configurable league settings** via YAML
- ✅ **Organized notebook structure** for maintainability
- ✅ **Daily automation** via scripts and cron
- ✅ **Git workflow** optimized for data science

## 🚦 Data Pipeline Status

| Component | Status | Last Updated |
|-----------|--------|--------------|
| FantasyPros Scraping | ✅ Working | 2025-08-14 |
| Fantasy Scoring | ✅ Working | 2025-08-14 |
| VBD Calculations | ✅ Working | 2025-08-14 |
| Top 300 Filtering | ✅ Working | 2025-08-14 |
| Daily Automation | ✅ Working | 2025-08-14 |

---

**Next Steps**: Run `python scripts/daily_update.py` to get started with current projections!