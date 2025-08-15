# Fantasy Football Draft Analysis

This project analyzes fantasy football draft data and projections to help with draft strategy and player evaluation.

## Project Structure

```
├── data/                           # Data files
│   ├── projections/               # Player projections by position
│   ├── espn_projections_20250814.csv  # ESPN Non-PPR Top 300 projections
│   └── rankings_20250814.csv     # Player rankings
├── scripts/                       # Data processing scripts
│   └── extract_espn_projections.py   # Extract data from ESPN PDF
├── config/                        # Configuration files
│   └── league-config.yaml        # League settings
├── *.ipynb                        # Jupyter notebooks for analysis
├── *.csv                          # Various data files
└── requirements.txt               # Python dependencies
```

## Data Sources

### ESPN Projections
- **File**: `data/espn_projections_20250814.csv`
- **Source**: ESPN Fantasy Football Draft Kit (Non-PPR Top 300)
- **Updated**: August 14, 2025
- **Weight**: 80% in probability calculations
- **Columns**:
  - `overall_rank`: Overall draft ranking (1-300)
  - `position`: Player position (QB, RB, WR, TE, K, DST)
  - `position_rank`: Position-specific ranking (e.g., WR1, RB12)
  - `player_name`: Player's full name
  - `team`: NFL team abbreviation
  - `salary_value`: Auction draft salary value ($)
  - `bye_week`: Team's bye week (5-14)

### External ADP Data
- **File**: `data/fantasypros_adp_20250815.csv`
- **Source**: FantasyPros aggregated ADP rankings
- **Updated**: August 15, 2025
- **Weight**: 20% in probability calculations
- **Purpose**: Provides season-long draft position context to balance real-time ESPN rankings

## Scripts

### extract_espn_projections.py
Extracts player data from ESPN Fantasy Football PDF projections and converts to CSV format.

**Usage:**
```bash
python scripts/extract_espn_projections.py [--pdf PDF_PATH] [--output OUTPUT_PATH]
```

**Features:**
- Parses multi-column PDF layout
- Handles all player positions (QB, RB, WR, TE, K, DST)
- Removes duplicates
- Sorts by overall ranking
- Exports to clean CSV format

## Core Probability System

### Dynamic Draft Probability Engine (80% ESPN + 20% ADP)
This project implements a sophisticated probability system for calculating real-time player availability during fantasy drafts:

**Key Features:**
- **Weighted Rankings**: 80% ESPN projections + 20% external ADP data
- **Discrete Survival Calculation**: Step-by-step simulation of picks until your next turn
- **Dynamic Updates**: Recalculates probabilities as players are drafted
- **VBD Integration**: Combines with Value Based Drafting scores for decision guidance

**Core Functions:**
- `compute_pick_probabilities()` - Blends ESPN/ADP rankings using softmax
- `probability_gone_before_next_pick()` - Calculates survival odds for specific players
- `calculate_player_metrics_new_system()` - Full enhanced metrics with new probability system

**Decision Logic:**
- **>80% available**: SAFE - Can wait until next pick
- **30-80% available**: DRAFT NOW - Risky to wait
- **<30% available**: REACH - Must draft now to secure

### Traditional Analysis Capabilities

The structured CSV data also enables standard fantasy football analyses:

1. **Draft Strategy**
   - Positional scarcity analysis
   - Value-based drafting
   - Auction salary optimization

2. **Bye Week Planning**
   - Identify bye week conflicts
   - Balance roster construction

3. **Player Evaluation**
   - Compare players within positions
   - Identify value picks
   - Salary cap analysis

## Requirements

See `requirements.txt` for Python dependencies. Key packages:
- `pdfplumber` - PDF text extraction
- `pandas` - Data manipulation
- `jupyter` - Interactive analysis

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Extract ESPN projections:
   ```bash
   python scripts/extract_espn_projections.py
   ```

3. Open main probability analysis notebook:
   ```bash
   jupyter notebook espn_probability_matrix.ipynb
   ```

## Key Notebooks

- **`espn_probability_matrix.ipynb`** - Main probability system with 80/20 weighted calculations
- **Analysis notebooks** - Various exploratory data analysis notebooks in `/notebooks/`

## League Configuration

Update `config/league-config.yaml` with your league settings:
- Scoring system (PPR/Non-PPR)
- Roster requirements
- Draft format (snake/auction)
- Team count

---

Last updated: August 14, 2025