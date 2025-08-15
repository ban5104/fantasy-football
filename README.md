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
- **Columns**:
  - `overall_rank`: Overall draft ranking (1-300)
  - `position`: Player position (QB, RB, WR, TE, K, DST)
  - `position_rank`: Position-specific ranking (e.g., WR1, RB12)
  - `player_name`: Player's full name
  - `team`: NFL team abbreviation
  - `salary_value`: Auction draft salary value ($)
  - `bye_week`: Team's bye week (5-14)

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

## Analysis Capabilities

The structured CSV data enables various fantasy football analyses:

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

3. Open Jupyter notebooks for analysis:
   ```bash
   jupyter notebook
   ```

## League Configuration

Update `config/league-config.yaml` with your league settings:
- Scoring system (PPR/Non-PPR)
- Roster requirements
- Draft format (snake/auction)
- Team count

---

Last updated: August 14, 2025