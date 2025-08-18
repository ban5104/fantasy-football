# Fantasy Football Draft Analysis & Monte Carlo Simulation System

A high-performance fantasy football draft system with optimized Monte Carlo simulation engine for strategy evaluation and real-time draft recommendations.

This project provides draft tracking, probability modeling, and AI-powered recommendations through clean, separated modules with **11x faster parallel processing** for live draft decisions.

## 🔬 Core Statistical Framework

**Value-Based Drafting (VBD) Enhanced with Probabilistic Selection Theory**

This system implements multiple VBD methodologies with **probabilistic replacement levels** and **roster-aware utility calculations**:

```
Utility_i = P_i × (VBD_i - R_i^dynamic) × (1 + β × PNI_p)
```

**Key Statistical Components:**
- **Traditional VBD**: VOLS, VORP, BEER methods with configurable baselines
- **Dynamic Replacement Levels**: Real-time calculation based on selection probabilities
- **Positional Need Index (PNI)**: Bayesian roster construction optimization
- **Draft Flow Analysis**: Sigmoid-based position scarcity detection
- **Multi-Factor Utility**: Combines value, availability, and strategic timing

*→ For mathematical details, see [Enhanced Probabilistic VBD Framework](docs/enhanced_probabilistic_vbd_framework.md)*  
*→ For data scientists/engineers, see [Statistical Core Concepts](docs/statistical_core_concepts.md)*

## Project Structure

```
├── backup_draft.py                # Live draft tracker with state export
├── monte_carlo_runner.py          # Clean interface for simulations
├── src/
│   └── monte_carlo/              # Modular simulation system
│       ├── probability.py        # ESPN/ADP probability calculations + envelope sampling
│       ├── opponent.py           # Opponent behavior modeling
│       ├── simulator.py          # Pure simulation engine with probabilistic projections
│       ├── strategies.py         # Strategy definitions
│       ├── replacement.py        # Dynamic replacement level calculations
│       └── __init__.py          # Clean public API
├── data/
│   ├── draft/                   # Live draft state
│   │   └── monte_carlo_state.json
│   ├── espn_projections_*.csv   # ESPN rankings
│   └── fantasypros_adp_*.csv    # ADP data
├── config/
│   └── league-config.yaml       # League settings
└── requirements.txt              # Python dependencies
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

## Quick Start

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Compare strategies before draft:
   ```bash
   # Quick comparison (100 sims, ~30s)
   python monte_carlo_runner.py compare
   
   # Thorough analysis with parallel processing (1000 sims, ~88s)
   python monte_carlo_runner.py compare --n-sims 1000 --parallel
   
   # Adaptive mode with variance reduction (auto-stops at convergence)
   python monte_carlo_runner.py compare_fast
   ```

3. During draft:
   ```bash
   # Terminal 1: Track draft
   python backup_draft.py
   
   # Terminal 2: Get recommendations
   python monte_carlo_runner.py balanced              # Quick (100 sims)
   python monte_carlo_runner.py balanced --parallel   # Fast (4x speedup)
   ```

## Monte Carlo Simulation System

### Architecture

The system uses **modular design** with clean separation of concerns:

- **`probability.py`**: ESPN/ADP probability calculations + Beta-PERT envelope sampling
- **`opponent.py`**: Models how opponents draft (rankings early → needs late)
- **`simulator.py`**: Pure simulation engine with probabilistic projections
- **`strategies.py`**: Strategy definitions (Balanced, Zero-RB, RB-Heavy, Hero-RB, Elite-QB)
- **`replacement.py`**: Dynamic replacement level calculations per simulation
- **`__init__.py`**: Clean public API

### How It Works

1. **Load Data**: ESPN (80%) + ADP (20%) rankings → softmax probabilities + envelope data
2. **Sample Projections**: Beta-PERT distribution from LOW/BASE/HIGH player values
3. **Model Opponents**: Round-based behavior (90% rankings R1 → 20% rankings R7)
4. **Simulate Drafts**: 100+ scenarios per strategy with variable projections
5. **Calculate Value**: Optimal starting lineup from sampled projections
6. **Return Results**: Expected value, variance, common patterns, recommendations

### Available Strategies

- **Balanced**: Equal weight to all positions
- **Zero-RB**: WR/TE heavy, RBs late
- **RB-Heavy**: Load up on RBs early
- **Hero-RB**: One elite RB, then WR/TE
- **Elite-QB**: Prioritize top QB early

### Live Draft Integration

**Automatic State Sync:**
```json
// backup_draft.py exports to monte_carlo_state.json
{
  "my_team_idx": 4,
  "current_global_pick": 17,
  "my_current_roster": ["Player1", "Player2"],
  "all_drafted": [...]
}
```

**Monte Carlo auto-loads and filters:**
```bash
python monte_carlo_runner.py balanced
# → "📡 Loaded draft state: Pick #18, 3 players drafted"
# → "Mode: LIVE DRAFT"
```

### Envelope System
- **Beta-PERT Sampling**: Uses LOW/BASE/HIGH projections for realistic uncertainty
- **Auto-Generation**: Creates ±20% envelopes from base projections if no envelope data
- **Backward Compatible**: Works with existing data without envelope columns
- **Dynamic Replacement**: Calculates replacement levels per simulation using sampled values

### Performance

**Optimized for Live Draft Decisions:**
- **Single-threaded**: 3.0 sims/sec (baseline)
- **4-core parallel**: 11.1 sims/sec (3.8x speedup) ✅
- **6-core parallel**: 13.7 sims/sec (4.6x speedup)

**Practical Timing:**
- 100 simulations: ~9 seconds (parallel) vs ~33 seconds (serial)
- 1000 simulations: ~88 seconds (parallel) vs ~333 seconds (serial)
- CRN adaptive: Auto-stops at convergence with 40-60% variance reduction

**Key Optimizations:**
- Vectorized batch sampling (196x → 1x per simulation)
- Dict caching for O(1) player lookups
- CPU-friendly parallelization (4 workers default, leaves 50% CPU for system)
- Unified methods reduce code complexity by 9%


### Data Requirements
Automatically uses existing project data:
- ESPN rankings from `data/espn_projections_20250814.csv`
- ADP data from `data/fantasypros_adp_20250815.csv`
- Projections from `data/rankings_top300_20250814.csv`
- **Optional**: Envelope data from `data/player_envelopes.csv` (LOW/BASE/HIGH columns)
- **Auto-Generated**: ±20% envelopes created from projections if no envelope file exists

## League Configuration

Update `config/league-config.yaml` with your league settings:
- Scoring system (PPR/Non-PPR)
- Roster requirements
- Draft format (snake/auction)
- Team count

---

## 🏈 What This Does

**Pre-Draft Analysis:**
- Scrapes current projections from FantasyPros for all positions
- Calculates VBD scores using multiple methods (VOLS, VORP, BEER, Dynamic VBD)
- Generates position-based rankings and tier analysis
- Creates customizable draft cheat sheets

**Live Draft Assistance:**
- Interactive 3-panel draft board with AI recommendations
- Real-time Dynamic VBD adjustments based on draft flow
- Enhanced emergency backup system with Dynamic VBD integration
- Snake draft visualization with position color-coding
- Multiple fallback options (full-featured and simplified trackers)

**NEW: Integrated Draft Management System:**
- **Seamless Team Selection**: Config-driven team setup with interactive position selection
- **Real-Time State Sync**: Automatic synchronization between manual draft entry and Monte Carlo AI
- **Dual Interface Support**: Terminal-based draft entry + Jupyter-based AI recommendations
- **Smart Auto-Configuration**: JSON state export with index conversion and validation
- **Enhanced Error Handling**: Graceful fallbacks and comprehensive input validation

**Advanced Features:**
- **Dynamic VBD** with real-time baseline adjustments during live drafts
- **Position scarcity detection** and automatic tier break identification
- Multi-factor AI recommendations considering value, scarcity, and roster needs
- **Monte Carlo Simulation**: 500+ simulations for expected value calculations and availability predictions
- **Integrated Workflow**: Manual draft tracking → AI recommendations → Decision support
- **Simplified implementation** for faster performance and easier maintenance
- Compatible with 8-16 team leagues (fully configurable)

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install dependencies (UV recommended)
uv sync

# Alternative: Direct pip install
uv pip install -r requirements_draft_board.txt

# Verify installation
export PATH="$(uv python find | head -n1 | xargs dirname):$PATH"
python3 -c "import pandas, numpy; print('✓ Ready to go!')"
```

### 2. Configure Your League
Edit `config/league-config.yaml`:
```yaml
basic_settings:
  teams: 14
  draft_type: "snake"

scoring:
  passing: {yards: 0.04, td: 4, int: -1}
  rushing: {yards: 0.1, td: 6}
  receiving: {yards: 0.1, td: 6, reception: 1}  # PPR scoring

roster:
  roster_slots: {QB: 1, RB: 2, WR: 2, TE: 1, FLEX: 1, K: 1, DST: 1}
```

### 3. Get Current Data
```bash
# Automated pipeline: scraping → scoring → VBD → rankings  
python scripts/daily_update.py

# Manual approach using Jupyter notebooks
jupyter notebook notebooks/02_analysis/03_vbd_calculations.ipynb
```

### 4. Draft Day
```bash
# Primary: Interactive draft board (Jupyter)
jupyter notebook notebooks/minimal_draft_board.ipynb

# Backup: Enhanced terminal tracker with Dynamic VBD
python backup_draft.py --dynamic-vbd

# Simplified version for faster startup
python backup_draft_simplified.py

# Reset for new draft
rm data/draft/draft_picks_latest.csv
```

## 📊 Core Features

### VBD Analysis Methods
- **VOLS** (Value Over Like Starters) - `baseline = teams × starters`
- **VORP** (Value Over Replacement Player) - `baseline = teams × (starters + 1)`
- **BEER** (Best Eleven Every Round) - `baseline = teams × (starters + 0.5)`
- **Dynamic VBD** - **Probabilistic replacement levels** adapting to draft flow
- **Enhanced VBD** - **Roster-aware utility calculations** with selection probabilities
- **Blended** - Optimized weighted combination (50% BEER + 25% VORP + 25% VOLS)

### Draft Tools
| Tool | Purpose | When to Use |
|------|---------|-------------|
| `minimal_draft_board.ipynb` | **PRIMARY** - 3-panel draft interface | Normal draft day |
| `backup_draft.py` | Enhanced emergency tracker with Dynamic VBD | ESPN API fails |
| `backup_draft_simplified.py` | Simplified tracker for faster startup | Quick backup needs |
| `interactive_draft_board.ipynb` | Advanced team roster tracking | Detailed analysis |
| `live_draft_tracker.py` | ESPN API integration | Real-time monitoring |

### Output Files
**Main Rankings:** `data/output/vbd_rankings_top300_YYYYMMDD.csv`  
**Individual Methods:** `rankings_vbd_[method]_top300_YYYYMMDD.csv`  
**Draft Sheet:** `draft_cheat_sheet.csv`  
**Live Draft:** `data/draft/draft_picks_latest.csv`

## 🎯 Usage Workflows

### Weekly Data Refresh
```bash
python scripts/daily_update.py  # Full automated pipeline

# Verify Dynamic VBD integration
python3 test_simple_integration.py
```

### Draft Preparation
1. Run data refresh to get current projections
2. Review VBD rankings: `data/output/vbd_rankings_top300_*.csv`
3. Customize league settings if needed
4. Generate draft cheat sheet

### Live Draft
1. **Primary**: Open `notebooks/minimal_draft_board.ipynb`
2. Set your team ID and draft position in Cell 1
3. Use 3-panel interface: controls, draft board, AI recommendations
4. **Backup**: Run `python backup_draft.py --dynamic-vbd` if main system fails
   - **Fast backup**: Use `python backup_draft_simplified.py` for quick startup

### Post-Draft Analysis
- Review team construction in `notebooks/post_draft_analysis.ipynb`
- Analyze draft efficiency and missed opportunities
- Generate trade and waiver wire recommendations

## 🛠️ System Architecture

```
Data Flow: FantasyPros → VBD Analysis → Dynamic Adjustments → Draft Tools

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │ ── │  Static VBD     │ ── │ Dynamic VBD     │ ── │   Draft Tools   │
│                 │    │                 │    │                 │    │                 │
│ • FantasyPros   │    │ • VOLS/VORP     │    │ • Real-time     │    │ • Draft Board   │
│ • Player DB     │    │ • BEER/Blended  │    │ • Scarcity      │    │ • AI Recomm.    │
│ • ESPN API      │    │ • Statistical   │    │ • Flow Analysis │    │ • Backup Track  │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Components
- **Data Pipeline**: `scripts/daily_update.py` → `src/` modules → `data/output/`
- **VBD Engine**: `src/vbd.py` + `src/dynamic_vbd.py` for real-time baseline adjustments
- **Draft Intelligence**: `src/draft_engine.py` with multi-factor recommendations + position scarcity detection
- **User Interfaces**: Jupyter notebooks + enhanced terminal backup with Dynamic VBD
- **Testing Suite**: Comprehensive tests for VBD calculations and draft integration

## 📁 Project Structure

```
├── config/league-config.yaml     # League settings and scoring
├── scripts/daily_update.py       # Automated data pipeline  
├── notebooks/                    # Interactive analysis tools
│   ├── minimal_draft_board.ipynb # PRIMARY draft interface
│   └── 02_analysis/              # VBD calculations
├── src/                          # Core Python modules
│   ├── vbd.py                   # Core VBD calculation engine (VOLS/VORP/BEER)
│   ├── dynamic_vbd.py           # Real-time baseline adjustments
│   ├── draft_engine.py          # AI recommendations with scarcity analysis
│   └── statistical_analysis.py  # Advanced modeling and predictions
├── data/
│   ├── output/                  # VBD rankings and analysis
│   └── draft/                   # Live draft tracking
└── backup_draft.py              # Emergency draft tracker
```

## ⚙️ Configuration

### League Customization
- **Teams**: 8-16 teams supported
- **Scoring**: PPR, Half-PPR, Standard, Custom
- **Roster**: Flexible positions (QB, RB, WR, TE, FLEX, K, DST)
- **VBD**: Adjustable replacement levels and blend weights

### Draft Settings
- **Snake draft**: Automatic order calculation
- **Dynamic VBD**: Real-time baseline adjustments
- **AI recommendations**: Multi-factor scoring system
- **Backup modes**: Multiple fallback options

## 🔧 Troubleshooting

### Environment Issues
```bash
# If UV sync fails
uv pip install -r requirements_draft_board.txt

# If imports fail
export PATH="$(uv python find | head -n1 | xargs dirname):$PATH"
PYTHONPATH=. python3 your_script.py

# Test core functionality
python3 -c "import sys; sys.path.insert(0, '.'); import src.vbd; print('✓ Core modules working')"

# Test Dynamic VBD integration
python3 test_simple_integration.py
PYTHONPATH=. python3 run_tests.py
```

### Common Problems
- **No data**: Run `python scripts/daily_update.py` first
- **Import errors**: Use `PYTHONPATH=.` prefix for Python commands
- **Draft fails**: Use `python backup_draft.py` as emergency backup
- **Old data**: Check timestamps on files in `data/output/`

## 📈 Advanced Features

### Enhanced Probabilistic VBD (Next Generation)
**Coming Implementation**: Upgrade from static to probabilistic replacement levels:

**Current Dynamic VBD:**
- Real-time baseline adjustments based on position scarcity
- Sigmoid-based scaling: `adjustment = scale × tanh(expected_picks / kappa)`
- Draft stage awareness and flow analysis

**Enhanced Framework (Planned):**
- **Probabilistic replacement levels**: `R_dynamic = best_player_with_survival_prob < 0.3`
- **Roster-aware utility**: Combines market scarcity with personal roster needs
- **Selection probability integration**: Uses ESPN algorithm data for availability forecasts
- **Positional Need Index**: Statistical measure of roster construction urgency

**Mathematical Evolution:**
```
# Current: Static replacement
VBD = player_points - replacement_level[position]

# Enhanced: Dynamic utility  
Utility = selection_prob × (VBD - dynamic_replacement) × (1 + roster_need_factor)
```

*See [Implementation Roadmap](docs/enhanced_probabilistic_vbd_framework.md) for technical details*

### AI Recommendations
Multi-factor analysis considering:
- **Value**: VBD score relative to alternatives
- **Need**: Roster construction requirements
- **Scarcity**: Remaining quality at position
- **Timing**: Optimal draft round for position

### Statistical Analysis
- Predictive modeling for player performance
- Monte Carlo simulations for draft outcomes
- Bayesian inference for updated valuations
- Tier analysis for strategic timing

## 📊 Sample Output

**VBD Rankings Preview:**
```
Player              Pos  VBD_BLENDED  VBD_BEER  VBD_VORP  Rank
Christian McCaffrey RB   42.8         45.2      38.1      1
Cooper Kupp        WR   38.4         41.0      34.5      2
Jonathan Taylor    RB   36.7         39.2      32.8      3
```

**AI Recommendation:**
```
🎯 RECOMMENDED: Cooper Kupp (WR)
📊 VBD Score: 38.4 (Rank #2 overall)  
⚡ Scarcity: HIGH - Only 3 elite WRs remain
🏗️ Roster Fit: Addresses WR1 need perfectly
⏰ Timing: Optimal round for WR position
```

---

**Built for competitive fantasy football managers who want data-driven draft advantages with reliable backup systems.**

*For detailed technical documentation, see [CLAUDE.md](CLAUDE.md)*

---

Last updated: December 2024
