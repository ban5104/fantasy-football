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
This project implements a sophisticated probability system for calculating real-time player availability during fantasy drafts using **multi-source exponential decay models** and **discrete survival analysis**.

#### Statistical Methodology

**1. Multi-Source Data Integration**
- **ESPN Rankings** (80% weight): Real-time draft consensus reflecting current player sentiment
- **ADP Data** (20% weight): Season-long historical draft position averages for stability
- **Rationale**: Balances current information with historical context to reduce volatility

**2. Softmax Probability Conversion**
Rankings are converted to pick probabilities using temperature-controlled exponential decay:
```
P(rank_i) = exp(-rank_i / τ) / Σ(exp(-rank_j / τ))
```
- **τ = 5.0**: Temperature parameter controlling probability spread
- **Lower ranks** receive exponentially higher selection probabilities
- **Advantages over normal distribution**: Better models actual draft selection behavior

**3. Discrete Survival Analysis**
For each player, calculates probability of being available through step-by-step simulation:
```
Survival_Probability = Π(1 - P_pick_at_step_j) for j = 1 to picks_until_next_turn
Probability_Gone = 1 - Survival_Probability
```

**Algorithm:**
1. Start with current available player pool
2. For each pick until your next turn:
   - Calculate pick probabilities for all remaining players using weighted softmax
   - Extract target player's selection probability for this step
   - Update survival probability: `survival *= (1 - p_pick_now)`
   - Remove most likely pick from available pool (simulation step)
3. Return final availability probability

#### Key Features
- **Weighted Rankings**: 80% ESPN projections + 20% external ADP data
- **Discrete Survival Calculation**: Step-by-step simulation of picks until your next turn
- **Dynamic Updates**: Recalculates probabilities as players are drafted
- **VBD Integration**: Combines with Value Based Drafting scores for decision guidance
- **Mathematical Robustness**: Probabilities always sum to 1.0, handles edge cases

#### Core Functions
- `compute_softmax_scores(rank_series, tau=5.0)` - Converts rankings to exponential probability scores
- `compute_pick_probabilities(available_df, espn_weight=0.8, adp_weight=0.2)` - Blends ESPN/ADP rankings using weighted softmax
- `probability_gone_before_next_pick(available_df, player_name, picks_until_next_turn)` - Calculates survival odds using discrete simulation
- `calculate_player_metrics_new_system()` - Full enhanced metrics with new probability system

#### Decision Logic
- **>80% available**: SAFE - Can wait until next pick
- **30-80% available**: DRAFT NOW - Risky to wait
- **<30% available**: REACH - Must draft now to secure

#### Statistical Advantages Over Traditional Methods
- **Multi-source robustness** vs single ranking dependency
- **Non-parametric approach** vs fixed normal distribution assumptions
- **Dynamic simulation** vs static probability calculations
- **Realistic selection modeling** vs symmetric probability distributions

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
- **`notebooks/monte_carlo_mvp_simulator.ipynb`** - Monte Carlo draft simulator for real-time decision support
- **Analysis notebooks** - Various exploratory data analysis notebooks in `/notebooks/`

## Monte Carlo Draft Simulator

### Overview
The **Monte Carlo Draft Simulator** (`notebooks/monte_carlo_mvp_simulator.ipynb`) provides real-time draft decision support by simulating thousands of possible draft outcomes to calculate the expected value (EV) of drafting each candidate player.

### Key Features
- **Expected Value Calculation**: Simulates remaining draft to estimate total roster value
- **Availability Prediction**: Calculates probability players will be available at your next pick
- **Mid-Draft Support**: Handles existing roster for in-progress draft decisions
- **Visual Decision Support**: EV bar charts and availability heatmaps
- **Clear Recommendations**: Color-coded guidance (🟢 Wait, 🟡 Consider, 🟠 Draft Now, 🔴 Must Draft)

### How It Works
1. **Simulation Process**:
   - For each candidate player, simulates drafting them
   - Other teams pick probabilistically based on 80% ESPN + 20% ADP rankings
   - You pick greedily (highest projection) in future rounds
   - Calculates final roster value using optimal lineup

2. **Roster Optimization**:
   - Optimizes for standard lineup: QB, 2 RB, 3 WR, TE, FLEX, K, DST
   - Maximizes total projected fantasy points
   - Accounts for positions already filled on your roster

3. **Decision Framework**:
   | Availability | Action | Reasoning |
   |-------------|--------|-----------|
   | >80% | 🟢 WAIT | Very likely available later |
   | 50-80% | 🟡 CONSIDER | Moderate risk |
   | 20-50% | 🟠 DRAFT NOW | High risk of being taken |
   | <20% | 🔴 MUST DRAFT | Won't be available |

### Quick Start Guide

1. **Open the simulator**:
   ```bash
   jupyter notebook notebooks/monte_carlo_mvp_simulator.ipynb
   ```

2. **Configure settings** (Cell 1):
   ```python
   CONFIG = {
       'my_team_idx': 7,        # Your draft position (0-based)
       'current_global_pick': 0, # Current pick number
       'n_sims': 500,           # Number of simulations
       'my_current_roster': []  # Players already drafted
   }
   ```

3. **Run all cells** to get initial recommendations

4. **During your draft**:
   - Update `current_global_pick` after each pick
   - Add drafted players to `my_current_roster`
   - Re-run cells 8-11 for updated recommendations

### Example Usage

**Pre-Draft Setup**:
```python
CONFIG['my_team_idx'] = 7      # Drafting 8th overall
CONFIG['n_teams'] = 12          # 12-team league
CONFIG['rounds'] = 15           # 15 rounds
```

**Mid-Draft Update**:
```python
CONFIG['current_global_pick'] = 31  # Currently pick #32
CONFIG['my_current_roster'] = [
    "Ja'Marr Chase",     # Round 1
    "Saquon Barkley",    # Round 2
    "Lamar Jackson"      # Round 3
]
# Re-run cells 8-11 for new recommendations
```

### Performance Tips
- **Speed**: 500 simulations takes ~30-60 seconds
- **Accuracy**: Increase to 1000+ simulations for important decisions
- **Player Pool**: Top 150 players covers most relevant decisions
- **Candidates**: Evaluates top 10 by projection by default

### Data Requirements
Automatically uses existing project data:
- ESPN rankings from `data/espn_projections_20250814.csv`
- ADP data from `data/fantasypros_adp_20250815.csv`
- Projections from `data/projections/projections_all_positions_20250814.csv`

## League Configuration

Update `config/league-config.yaml` with your league settings:
- Scoring system (PPR/Non-PPR)
- Roster requirements
- Draft format (snake/auction)
- Team count

---

Last updated: December 2024