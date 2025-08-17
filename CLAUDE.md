# Fantasy Football Draft Optimization System - Technical Documentation

## Project Overview
This project implements three complementary systems:
1. **Probability System**: Real-time player availability predictions using weighted rankings
2. **Monte Carlo Draft Optimizer**: Global roster optimization using advanced simulation techniques
3. **Integrated Draft Management**: Real-time sync between manual draft entry and AI recommendations

## System Architecture

### Part 1: Core Probability System (80% ESPN + 20% ADP)

### Statistical Methodology Overview
This system implements a **sophisticated multi-source probability engine** that combines ranking data from multiple sources using exponential decay functions and discrete survival analysis to predict player availability in fantasy football drafts.

### Mathematical Foundation

#### 1. Softmax Probability Conversion
The system converts ordinal rankings into probability distributions using a temperature-controlled softmax function:

```
P(rank_i) = exp(-rank_i / τ) / Σ(exp(-rank_j / τ))
```

Where:
- `rank_i` = ordinal ranking of player i (1, 2, 3, ...)
- `τ` = temperature parameter controlling probability spread (default: 5.0)
- Lower ranks receive exponentially higher probabilities
- Higher τ = more uniform distribution, Lower τ = more concentrated on top ranks

#### 2. Weighted Multi-Source Integration
Rankings from ESPN and ADP sources are combined using weighted softmax scores:

```
Combined_Score_i = (0.8 × ESPN_Score_i) + (0.2 × ADP_Score_i)
Final_Probability_i = Combined_Score_i / Σ(Combined_Score_j)
```

**Rationale for 80/20 weighting:**
- ESPN rankings reflect real-time consensus and current player sentiment
- ADP data provides season-long stability and reduces recency bias
- 80/20 balance prioritizes current information while maintaining historical context

#### 3. Discrete Survival Probability Calculation
For each player, the system calculates survival probability through a step-by-step simulation:

```
Survival_Probability = Π(1 - P_pick_at_step_j) for j = 1 to picks_until_next_turn
Probability_Gone = 1 - Survival_Probability
```

**Algorithm:**
1. Start with current available player pool
2. For each pick until your next turn:
   - Calculate pick probabilities for all remaining players
   - Extract target player's pick probability for this step
   - Update survival probability: `survival *= (1 - p_pick_now)`
   - Remove most likely pick from available pool (simulation step)
3. Return probability player is gone: `1 - survival_probability`

### Implementation Details
The system uses a **weighted ranking approach** with discrete survival probability calculations:

- **80% ESPN rankings** (primary weight) - Real-time draft consensus
- **20% External ADP rankings** (secondary weight) - Season-long average draft position
- **Softmax probability conversion** - Converts ranks to pick probabilities with temperature control
- **Discrete survival calculation** - Step-by-step simulation of picks until your next turn

### Key Functions

#### `compute_softmax_scores(rank_series, tau=5.0)`
Converts rankings to probability scores using exponential decay:
```python
scores = np.exp(-rank_series / tau)
```
- Lower ranks get higher scores
- `tau` parameter controls how steeply probabilities drop (higher tau = more spread out)

#### `compute_pick_probabilities(available_df, espn_weight=0.8, adp_weight=0.2)`
Blends ESPN and ADP rankings into unified pick probabilities:
```python
combined_scores = espn_weight * espn_scores + adp_weight * adp_scores
probs = combined_scores / combined_scores.sum()  # Normalize to sum to 1.0
```

#### `probability_gone_before_next_pick(available_df, player_name, picks_until_next_turn)`
Calculates probability a specific player is drafted before your next pick:
- Simulates each pick step-by-step
- Removes most likely player at each step
- Updates survival probability: `survival_prob *= (1 - p_pick_now)`
- Returns: `1 - survival_prob` (probability player is gone)

### Data Integration
The system integrates with existing draft data:
- **ESPN projections**: `data/espn_projections_20250814.csv`
- **External ADP data**: `data/fantasypros_adp_20250815.csv` 
- **Draft rankings**: `draft_cheat_sheet.csv`

### Usage in Notebook
```python
# Load and merge ranking data
ranking_df = load_and_merge_ranking_data()

# Calculate enhanced metrics with new probability system
enhanced_df = calculate_player_metrics_new_system(
    ranking_df, projections_data, 
    my_picks=[8, 17, 32, 41, 56, 65, 80, 89],
    current_pick=1,
    drafted_players=set()
)
```

## Advantages Over Previous System

### Old System (Normal Distribution)
- Used single ranking source
- Normal distribution approximation
- Static standard deviation (σ=3)
- Less realistic for actual draft behavior

### New System (80/20 Weighted + Discrete Survival)
- Combines multiple ranking sources (ESPN + ADP)
- Discrete step-by-step simulation
- Dynamic probability updates as players are drafted
- More realistic modeling of draft behavior patterns

### Statistical Advantages Over Previous System

#### Previous System (Normal Distribution Approximation)
**Limitations:**
- **Single Source Dependency**: Relied solely on one ranking source
- **Parametric Assumptions**: Used normal distribution with fixed σ=3 parameter
- **Static Modeling**: No dynamic updates as draft progresses
- **Unrealistic Probability Model**: Normal curves don't reflect actual draft selection patterns

**Mathematical Issues:**
```
P(available_at_pick) = 1 - Φ((pick - rank) / σ)  [Normal CDF]
```
- Assumes symmetric probability distribution around player rank
- Fixed standard deviation ignores ranking consensus strength
- No consideration of other drafters' decision-making patterns

#### New System (Multi-Source Exponential Decay + Discrete Survival)
**Advantages:**
- **Multi-Source Robustness**: Combines ESPN real-time + ADP historical data
- **Non-Parametric Approach**: Uses empirical ranking distributions
- **Dynamic Simulation**: Updates probabilities after each simulated pick
- **Realistic Selection Modeling**: Models actual draft selection behavior

**Statistical Improvements:**
1. **Exponential Decay vs Normal**: Better reflects how draft value perception works
2. **Discrete vs Continuous**: Models individual pick decisions rather than continuous probability
3. **Multi-Source Integration**: Reduces single-source bias and ranking volatility
4. **Survival Analysis**: Accounts for sequential elimination of players from available pool

**Mathematical Robustness:**
- Probabilities always sum to 1.0 across available players
- No negative probabilities or mathematical impossibilities
- Handles edge cases (empty pools, already-drafted players)
- Temperature parameter allows calibration to different draft styles

## Decision Logic
The system provides strategic guidance based on availability probabilities:
- **>80% available**: SAFE - Can wait until next pick
- **>70% available at pick after**: WAIT - Target later
- **30-80% available**: DRAFT NOW - Risky to wait  
- **<30% available**: REACH - Must draft now to secure

## Technical Notes
- Temperature parameter `tau=5.0` balances probability spread
- 80/20 weighting can be adjusted via function parameters
- System handles edge cases (empty player pools, players already drafted)
- All probabilities sum to 1.0 across available players
- Compatible with existing ranking systems

---

## Part 2: Monte Carlo Draft Optimizer (MVP Implementation)

### Overview
The Monte Carlo Draft Optimizer (`notebooks/monte_carlo_mvp_simulator.ipynb`) provides real-time draft decision support by simulating thousands of possible draft outcomes to calculate the expected value (EV) of drafting each candidate player.

### Core Concepts

#### Expected Value (EV) Calculation
The optimizer simulates the remainder of the draft many times to estimate:
- **Expected Roster Value**: Total projected fantasy points from optimal starting lineup
- **Marginal Value**: Additional value a player adds to your existing roster
- **Availability Probability**: Chance a player will still be available at your next pick

#### Simulation Methodology
1. **For each candidate player:**
   - Force draft that player to your team
   - Simulate remaining picks using 80/20 ESPN/ADP probabilities
   - Other teams pick probabilistically based on rankings
   - You pick greedily (highest projection available) in future rounds
   - Calculate final roster value using optimal lineup

2. **Across N simulations (default 500):**
   - Average the roster values → Expected Value (EV)
   - Track which players remain available → Availability %
   - Calculate standard deviation → Uncertainty measure

### Roster Optimization Algorithm

#### Starting Lineup Requirements
```python
STARTER_REQUIREMENTS = {
    'QB': 1,   # 1 Quarterback
    'RB': 2,   # 2 Running Backs
    'WR': 3,   # 3 Wide Receivers  
    'TE': 1,   # 1 Tight End
    'FLEX': 1, # 1 Flex (best remaining RB/WR/TE)
    'K': 1,    # 1 Kicker
    'DST': 1   # 1 Defense/Special Teams
}
```

#### Value Calculation
```python
def compute_team_value(chosen_players):
    # Sort each position by projected points
    # Take top N at each position for starters
    # Sum starter projections
    # Add best remaining RB/WR/TE as FLEX
    return total_projected_points
```

### Configuration Parameters

```python
CONFIG = {
    'n_teams': 12,              # League size
    'rounds': 15,               # Number of rounds
    'my_team_idx': 7,           # Your draft position (0-based)
    'current_global_pick': 0,   # Current pick number (0-based)
    'top_k': 150,              # Player pool size to consider
    'candidate_count': 10,      # Top players to evaluate
    'n_sims': 500,             # Simulations per candidate
    'espn_weight': 0.8,        # ESPN ranking weight
    'adp_weight': 0.2,         # ADP ranking weight
    'my_current_roster': []    # Players already drafted
}
```

### Key Features

#### 1. Mid-Draft Support
- Handles existing roster by including already-drafted players
- Calculates marginal value of new players given current team
- Updates availability based on players already taken

#### 2. Decision Framework
The simulator provides clear recommendations based on EV and availability:

| Availability | Decision | Rationale |
|-------------|----------|-----------|
| >80% | 🟢 WAIT | Very likely available at next pick |
| 50-80% | 🟡 CONSIDER | Moderate risk, depends on value |
| 20-50% | 🟠 DRAFT NOW | High risk of being taken |
| <20% | 🔴 MUST DRAFT | Won't be available later |

#### 3. Visualizations

**Expected Value Bar Chart**
- Horizontal bars showing EV for each candidate
- Error bars indicate uncertainty (standard deviation)
- Color-coded by position for quick identification

**Availability Heatmap**
- Shows probability each player survives to your next pick
- Color gradient: Green (safe) → Red (risky)
- Top 20 players by ranking displayed

### Usage Workflow

#### Pre-Draft Setup
```python
# 1. Set your draft position (0-based indexing)
CONFIG['my_team_idx'] = 7  # Drafting 8th

# 2. Adjust league settings if needed
CONFIG['n_teams'] = 12
CONFIG['rounds'] = 15

# 3. Increase simulations for more accuracy
CONFIG['n_sims'] = 1000  # More accurate but slower
```

#### During Draft
```python
# 1. Update current pick (0-based)
CONFIG['current_global_pick'] = 15  # Pick #16

# 2. Add players you've already drafted
CONFIG['my_current_roster'] = ["Ja'Marr Chase", "Josh Allen"]

# 3. Re-run simulation cells (8-11) for updated recommendations
```

#### Interpreting Results
- **Higher EV = Better pick** for maximizing total roster value
- **Lower availability % = Draft now** to avoid missing out
- **Decision combines both** for balanced recommendations

### Technical Implementation Details

#### Pick Probability Model
```python
def build_pick_probs(available_players):
    # Convert rankings to scores (inverse relationship)
    espn_scores = 1.0 / (espn_ranks + 1e-6)
    adp_scores = 1.0 / (adp_ranks + 1e-6)
    
    # Weighted combination
    combined = 0.8 * espn_scores + 0.2 * adp_scores
    
    # Normalize to probabilities
    return combined / combined.sum()
```

#### Simulation Engine
```python
def simulate_with_candidate(candidate_id, n_sims=500):
    for sim in range(n_sims):
        # Start with current roster + candidate
        my_team = current_roster + [candidate_id]
        available = remove_drafted_players()
        
        # Simulate each remaining pick
        for pick in remaining_picks:
            if my_pick:
                # Greedy: take best available by projection
                best = max(available, key=projection)
                my_team.append(best)
            else:
                # Probabilistic: sample based on rankings
                player = sample(available, p=pick_probs)
            available.remove(player)
        
        # Calculate final roster value
        ev = compute_team_value(my_team)
    
    return mean(evs), std(evs)
```

### Performance Considerations

- **Simulation Count**: 500 sims balances speed vs accuracy (30-60 seconds)
- **Player Pool**: Top 150 by ESPN rank captures relevant players
- **Candidate Count**: Evaluating top 10 by projection covers key decisions
- **Common Random Numbers**: Ensures fair comparison between candidates

### Data Requirements

The simulator requires three data files:
1. **ESPN Rankings** (`data/espn_projections_20250814.csv`)
   - Columns: player_name, position, overall_rank
2. **ADP Data** (`data/fantasypros_adp_20250815.csv`)
   - Columns: PLAYER, RANK, POSITION
3. **Projections** (`data/projections/projections_all_positions_20250814.csv`)
   - Columns: PLAYER, MISC_FPTS or FPTS (fantasy points)
   - **Note**: Player names include team abbreviations (e.g., "Ja'Marr Chase CIN") which are automatically stripped during data loading

### Implementation Notes

**File**: `notebooks/monte_carlo_mvp_simulator_fixed.ipynb`

**Data Processing:**
- Handles player name matching between projection data (includes team abbreviations) and ESPN/ADP data (clean names)
- Configured for 14-team league with 2 WR roster spots (QB, 2RB, 2WR, TE, FLEX, K, DST)
- Includes data validation and merge success reporting

**Player Name Normalization:**
```python
# Extract player name without team abbreviation
proj_df['player_name'] = proj_df['player_name'].str.replace(r'\s+[A-Z]{2,3}$', '', regex=True).str.strip()
```

This ensures "Ja'Marr Chase CIN" in projections matches "Ja'Marr Chase" in ESPN/ADP data for proper fantasy point integration.

### Advantages Over Simple Rankings

1. **Considers Your Roster**: Accounts for positions already filled
2. **Models Opponent Behavior**: Simulates realistic draft patterns
3. **Quantifies Risk**: Provides availability probabilities
4. **Optimizes Globally**: Maximizes total roster value, not just next pick
5. **Handles Uncertainty**: Shows confidence intervals via standard deviation

### Limitations & Future Improvements

**Current Limitations:**
- Assumes other teams pick based on consensus rankings
- Doesn't model team-specific needs or strategies
- Uses simple greedy algorithm for your future picks
- No position scarcity adjustments

**Potential Enhancements:**
- Model position runs and scarcity
- Learn team-specific draft tendencies
- Optimize future pick strategy (not just greedy)
- Add trade value considerations
- Include keeper/dynasty league logic

---

## Part 3: Integrated Draft Management System

### Overview
The Integrated Draft Management System combines manual draft entry with real-time AI recommendations by creating a seamless sync between the backup draft script and Monte Carlo MVP simulator.

### Architecture

#### Component Integration
```
Manual Draft Entry → State Synchronization → AI Recommendations
     (backup_draft.py)    (JSON file)         (Monte Carlo)
```

#### Data Flow
1. **Draft Setup**: User selects team position from config-based team list
2. **Manual Entry**: Draft picks entered via terminal interface
3. **Auto-Sync**: State automatically exported to `monte_carlo_state.json`
4. **AI Analysis**: Monte Carlo simulator reloads state for updated recommendations
5. **Decision Support**: Real-time EV calculations and availability predictions

### Implementation Details

#### Team Configuration System
**File**: `config/league-config.yaml`
```yaml
team_names:
  - "Team 1"      # Pick 1, 28, 29, 56...
  - "Team 2"      # Pick 2, 27, 30, 55...
  # ... 14 teams total
```

**Features**:
- Config-driven team names with dynamic fallback
- Snake draft position calculation
- Automatic league size detection

#### Manual Draft Interface
**File**: `backup_draft.py`

**Key Components**:
- `load_team_names_from_config()`: Config loading with fallback to generated names
- `select_draft_position()`: Interactive team selection menu (1-14)
- `export_monte_carlo_state()`: Automatic state sync after each pick
- `show_draft_order()`: Snake draft visualization command

**Team Selection Flow**:
```
🏈 SELECT YOUR DRAFT POSITION
========================================
   1. Team 1
   2. Team 2
   ...
  14. Team 14

Select your position (1-14): 8
✅ You are: Team 8 (Pick #8)
📍 Your picks: #8, #21, #36, #49, ...
```

#### State Synchronization
**File**: `data/draft/monte_carlo_state.json`

**State Structure**:
```json
{
  "my_team_idx": 7,                    // 0-based team index for Monte Carlo
  "current_global_pick": 17,           // 0-based current pick number
  "my_current_roster": ["Player1", "Player2"],  // Your drafted players
  "all_drafted": ["All", "Players"],  // Complete draft state
  "timestamp": "2025-08-17T...",       // Last update time
  "team_name": "Team 8",               // Human-readable team name
  "total_teams": 14                    // League size
}
```

**Index Conversion**:
- Backup draft uses 1-based indexing (Pick #1, Team 1)
- Monte Carlo uses 0-based indexing (team_idx: 0, pick: 0)
- Automatic conversion in `export_monte_carlo_state()`

#### Monte Carlo Integration
**File**: `notebooks/monte_carlo_mvp_simulator_fixed.ipynb`

**Auto-Reload Function**:
```python
def reload_draft_state():
    """Reload CONFIG from backup draft state with validation"""
    # Load and validate JSON state
    # Update Monte Carlo CONFIG
    # Display current roster and pick information
```

**Validation Features**:
- Required keys validation (`my_team_idx`, `current_global_pick`, `my_current_roster`)
- Data type checking (integers, lists, ranges)
- Corrupted JSON handling
- Clear error messages and recovery

### Usage Workflow

#### Pre-Draft Setup
1. **Configure Teams** (one-time):
   ```yaml
   # Edit config/league-config.yaml
   team_names:
     - "John's Team"
     - "Sarah's Squad"
     # ... etc
   ```

2. **Start Draft Session**:
   ```bash
   python backup_draft.py
   ```

#### During Draft
1. **Team Selection**: Choose your position (1-14) from menu
2. **Manual Entry**: Enter picks as usual (`player_name` format)
3. **Auto-Sync**: State exported after each pick automatically
4. **Get Recommendations**:
   - Open `notebooks/monte_carlo_mvp_simulator_fixed.ipynb`
   - Run Cell 0 to reload current state
   - Run Cells 8-11 for updated AI recommendations

#### Commands Available
- **Standard**: Player name entry, team assignments
- **`ORDER`**: Display snake draft order visualization
- **`STATUS`**: Show current draft progress
- **`HELP`**: Display all available commands

### Error Handling & Recovery

#### Graceful Fallbacks
- **Missing Config**: Falls back to `Team 1`, `Team 2`, etc.
- **Corrupted JSON**: Monte Carlo shows clear error, continues working
- **File System Issues**: Creates directories automatically, handles permissions
- **Invalid Input**: Clear validation messages, retry prompts

#### Validation Layers
1. **Input Validation**: Team position range (1-14), numeric input
2. **Config Validation**: Team list length, YAML structure
3. **JSON Validation**: Required keys, data types, value ranges
4. **File System**: Directory creation, write permissions

### Testing & Quality Assurance

#### Automated Testing
- **Core Functionality**: 7 tests in `tests/test_backup_draft.py`
- **Integration**: `test_backup_draft_integration.py`
- **Simple Validation**: `test_simple_integration.py`

#### Manual Testing Checklist
- [ ] Team selection menu displays correctly
- [ ] Draft picks update state file
- [ ] Monte Carlo reloads state successfully
- [ ] Error handling works with invalid input
- [ ] Snake draft order displays correctly

### Performance Considerations

#### File I/O Optimization
- JSON writes only after successful picks
- Directory creation cached
- Minimal file size (essential data only)

#### Memory Management
- State export triggered only after changes
- No persistent connections between systems
- Clean error recovery without memory leaks

### Configuration Customization

#### Team Names
```yaml
# Standard teams
team_names:
  - "Team 1"
  - "Team 2"

# Custom league
team_names:
  - "Dynasty Destroyers"
  - "Fantasy Legends"
```

#### League Settings Integration
```yaml
basic_settings:
  teams: 14              # Detected automatically
  draft_type: "snake"    # Snake draft pattern
```

### Future Enhancement Opportunities

#### Planned Improvements
- **Real-time WebSocket**: Replace file-based sync with live connection
- **Draft Strategy AI**: Pre-draft team strategy recommendations
- **Historical Analysis**: Post-draft performance tracking
- **Mobile Interface**: Web-based draft board for multiple devices

#### Extension Points
- **Custom Scoring**: Integration with league-specific point systems
- **Trade Analysis**: Mid-draft trade value calculations
- **Keeper Leagues**: Historical player retention logic
- **Auction Drafts**: Budget management and bidding strategies

This integration provides a production-ready draft management system combining the reliability of manual entry with the intelligence of AI-powered recommendations.

---

## Development Guide

This file provides guidance to Claude Code (claude.ai/code) when working with this fantasy football draft analysis and assistance system.

### System Overview

This is a production-ready fantasy football draft analysis system combining:
- **Probability-based player availability predictions** using weighted ESPN and ADP rankings
- **Monte Carlo simulation** for optimal draft decision-making
- **Real-time draft tracking** with seamless integration between manual entry and AI recommendations

### Core Components
- **Probability Engine**: 80% ESPN + 20% ADP weighted rankings with softmax conversion
- **Monte Carlo Optimizer**: Simulates thousands of draft outcomes to find optimal picks
- **Draft Tracker**: Terminal-based backup system with Monte Carlo state export
- **Interactive Notebooks**: Jupyter-based draft boards and analysis tools

### Development Commands

#### Environment Setup
```bash
# Install dependencies (UV recommended)
uv sync
# Fallback: uv pip install -r requirements_draft_board.txt

# Verify setup
export PATH="$(uv python find | head -n1 | xargs dirname):$PATH"
python3 -c "import pandas, numpy; print('✓ Dependencies ready')"
```

#### Core Workflows

##### Data Pipeline
```bash
# Scrape latest projections
python scripts/scrape_projections.py

# Process and analyze data
jupyter notebook notebooks/02_analysis/*.ipynb
```

##### Draft Day Operations
```bash
# Primary: Interactive draft board
jupyter notebook notebooks/minimal_draft_board.ipynb

# Backup: Emergency terminal tracker
python backup_draft.py

# Monte Carlo simulator
jupyter notebook notebooks/monte_carlo_mvp_simulator_fixed.ipynb

# Advanced: Real-time ESPN integration
python live_draft_tracker.py
```

##### Testing
```bash
# Run all tests
PYTHONPATH=. python3 -m pytest tests/test_backup_draft.py -v

# Integration testing
python3 test_backup_draft_integration.py
python3 test_simple_integration.py
```

### Architecture Overview

#### Data Flow Pipeline
```
FantasyPros → Raw Projections → Fantasy Points → Rankings → Draft Tools
     ↓              ↓               ↓              ↓              ↓
scraping.py → data/raw/ → scoring.py → analysis → notebooks/backup_draft.py
```

#### Core Modules

##### Data Processing
- **`src/scraping.py`** - Web scraping FantasyPros for all positions (QB/RB/WR/TE/K/DST)
- **`src/scoring.py`** - Converts raw stats to fantasy points using configurable league scoring
- **`src/statistical_analysis.py`** - Advanced statistical modeling and analysis
- **`src/utils.py`** - Data validation, logging, file I/O utilities

##### Draft Tools
- **`backup_draft.py`** - Terminal-based emergency draft tracker with Monte Carlo integration
- **`live_draft_tracker.py`** - ESPN API integration for real-time draft monitoring
- **`draft_board_app.py`** - Streamlit web interface for draft boards
- **`src/draft_engine.py`** - AI recommendation engine
- **`src/data_processor.py`** - Flexible CSV handling for various data sources

##### Interactive Notebooks
- **`notebooks/minimal_draft_board.ipynb`** - Primary 3-panel draft interface
- **`notebooks/monte_carlo_mvp_simulator_fixed.ipynb`** - Monte Carlo draft optimizer
- **`notebooks/draft_preparation.ipynb`** - Pre-draft analysis and preparation
- **`notebooks/interactive_draft_board.ipynb`** - Advanced draft tracking with team rosters

#### Configuration

##### League Settings
**`config/league-config.yaml`** - Master configuration file:
- League settings (14 teams, scoring rules, roster requirements)
- Team names and draft order
- Scoring system (PPR, half-PPR, standard)

```yaml
basic_settings:
  teams: 14
  draft_type: "snake"
  
team_names:
  - "Team 1"
  - "Team 2"
  # ... etc

scoring:
  passing_td: 4
  passing_yards: 0.04
  rushing_td: 6
  # ... etc
```

### Data Architecture

#### Input Data Sources
- **`data/raw/projections_*_YYYYMMDD.csv`** - FantasyPros scraped projections by position
- **`data/CSG Fantasy Football Sheet - 2025 v13.01.csv`** - Master player database with ADP
- **`data/espn_projections_*.csv`** - ESPN player rankings and projections
- **`data/fantasypros_adp_*.csv`** - FantasyPros ADP data
- **External APIs**: ESPN (when available) for live draft data

#### Processing & Output
- **`data/processed/`** - Intermediate calculations and transformations
- **`data/output/`** - Final rankings and analysis outputs
- **`data/draft/`** - Live draft state and pick history
- **`draft_cheat_sheet.csv`** - Formatted draft preparation sheet
- **`draft_picks_latest.csv`** - Current draft state (ESPN-compatible format)
- **`monte_carlo_state.json`** - Monte Carlo simulator state

### Key Development Patterns

#### Configuration Loading
```python
import yaml

# Load league configuration
with open('config/league-config.yaml', 'r') as f:
    config = yaml.safe_load(f)
```

#### Draft State Management
```python
from backup_draft import BackupDraftTracker

# Emergency backup tracking
tracker = BackupDraftTracker()
tracker.run_interactive()
```

#### Monte Carlo Integration
```python
import json

# Load Monte Carlo state
with open('data/draft/monte_carlo_state.json', 'r') as f:
    state = json.load(f)
    
# State contains:
# - my_team_idx: Your team position (0-based)
# - current_global_pick: Current pick number
# - my_current_roster: Your drafted players
# - all_drafted: All drafted players
```

#### Error Handling
- Comprehensive try/catch with structured logging
- Graceful degradation when external APIs fail
- Resume capability for interrupted draft sessions
- Automatic backup saves after each pick

### Development Workflows

#### Adding New Features
1. Prototype in Jupyter notebook
2. Extract reusable functions to `src/` modules
3. Add configuration options to `config/league-config.yaml`
4. Create test cases in `tests/`
5. Update documentation

#### Data Source Integration
1. Add scraping logic to `src/scraping.py`
2. Implement data transformation in `src/data_processor.py`
3. Update configuration schema if needed
4. Add error handling and fallback mechanisms

### Testing

#### Running Tests
```bash
# Draft tracking tests
PYTHONPATH=. python3 -m pytest tests/test_backup_draft.py -v

# Integration tests
python3 test_backup_draft_integration.py
python3 test_simple_integration.py
```

#### Manual Validation
```bash
# Test backup draft tracker
python backup_draft.py

# Test Monte Carlo simulator
jupyter notebook notebooks/monte_carlo_mvp_simulator_fixed.ipynb
```

#### Environment Issues Resolution
- **ModuleNotFoundError**: Ensure `export PATH="$(uv python find | head -n1 | xargs dirname):$PATH"`
- **Import failures**: Use `PYTHONPATH=.` prefix for all Python commands
- **Package conflicts**: Re-run `uv sync` or fallback to `uv pip install -r requirements_draft_board.txt`

### Performance Considerations

#### Optimization Patterns
- **Vectorized pandas operations** for all statistical calculations
- **Efficient player search** using string matching and indexing
- **Caching mechanisms** for frequently accessed data
- **Top 300 focus** to limit memory usage for draft-relevant players

#### Memory Management
- Lazy loading of large datasets
- Incremental processing for live draft updates
- Efficient DataFrame operations

#### Scalability
- Configurable league sizes through YAML configuration
- Modular architecture allowing selective feature enabling
- API rate limiting with respectful delays (2-second intervals)

### Security & Data Handling

#### Data Privacy
- No personal information stored in player databases
- Local file storage only - no external data transmission
- Anonymized draft tracking using team numbers

#### API Usage
- Rate-limited requests to FantasyPros (2-second delays)
- Graceful failure handling when external services are unavailable
- No API keys required for core functionality

### Advanced Features

#### Monte Carlo Draft Simulation
- Simulates thousands of draft outcomes
- Calculates expected value for each player
- Provides availability probabilities
- Optimizes roster construction

#### AI Draft Recommendations
- Multi-factor scoring considering value and need
- Tier break detection for strategic positional runs
- Roster construction optimization
- Pattern analysis for draft flow prediction

### Troubleshooting

#### Environment Setup
- **UV sync failures**: Use `uv pip install -r requirements_draft_board.txt`
- **Python path issues**: Always use `PYTHONPATH=.` prefix
- **Module import errors**: Verify UV environment activation

#### Data Issues
- **Scraping failures**: Check FantasyPros site availability, inspect network logs
- **CSV format errors**: Validate data quality, check for missing columns
- **Draft tracking issues**: Verify CSV format compatibility, check file permissions

#### Performance Problems
- **Slow calculations**: Profile code, consider reducing player scope
- **Memory issues**: Monitor DataFrame sizes, implement lazy loading
- **UI responsiveness**: Optimize real-time update frequency

### League Customization

#### Scoring Configuration
- Modify `config/league-config.yaml` scoring section
- All major scoring systems supported (PPR, Half-PPR, Standard)
- Complex scoring rules (defensive TDs, return yards) handled

#### Roster Configuration
- Flexible position requirements (supports FLEX, SUPERFLEX)
- Configurable bench sizes and roster maximums
- Custom position eligibility rules

### Integration Points

#### External Systems
- **ESPN API**: Live draft monitoring (when available)
- **FantasyPros**: Primary data source for projections
- **CSV Import/Export**: Compatible with popular fantasy platforms

#### Extensibility
- Plugin architecture for new data sources
- Configurable UI components in Jupyter notebooks
- Modular calculation engine for custom analysis

This system provides a production-ready fantasy football draft assistance platform optimized for live draft scenarios with Monte Carlo simulation and probability-based predictions.
