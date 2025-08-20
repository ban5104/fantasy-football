# Fantasy Football Draft System - Development Standards

## Quick Start
```bash
# Install & Setup
uv sync

# Draft Day
python backup_draft.py                    # Primary draft tracker with state export
python monte_carlo_runner.py degradation  # Positional degradation analysis (NEW!)
python monte_carlo_runner.py compare      # AI strategy comparison (100 sims, ~30s)
python monte_carlo_runner.py balanced     # Live draft recommendations

# High-Performance Mode (NEW)
python monte_carlo_runner.py balanced --n-sims 1000 --parallel  # 4 cores, ~88s
python monte_carlo_runner.py compare_fast                        # CRN adaptive mode

# Championship DNA Analysis (NEW)
PYTHONPATH=. uv run python notebooks/run_championship_dna.py --pick 5 --sims 1000  # Pattern discovery
PYTHONPATH=. uv run python notebooks/championship_dna_hybrid.py pre-draft --pick 5  # Tier-based recommendations

# Testing
PYTHONPATH=. python3 -m pytest tests/test_backup_draft.py -v
```

## Environment Setup (IMPORTANT)
```bash
# This project uses UV for dependency management
# UV is required - install with: pip install uv

# Setup virtual environment
uv venv
uv sync

# CRITICAL: Always use uv run or PYTHONPATH prefix
PYTHONPATH=. uv run python3 script.py     # Preferred - uses UV environment
PYTHONPATH=. python3 script.py            # Alternative if UV is active

# Common environment issues and fixes:
# 1. "ModuleNotFoundError": Use PYTHONPATH=. prefix
# 2. "No module named 'src'": Run from project root with PYTHONPATH=.
# 3. Package conflicts: Use 'uv sync' to reset environment
# 4. Import errors: Never run scripts directly, always use PYTHONPATH=.
```

## Architecture
- **Probability Engine**: 80% ESPN + 20% ADP weighted rankings with softmax conversion
- **Envelope Sampling**: Beta-PERT distributions from LOW/BASE/HIGH projections for uncertainty modeling
- **Opponent Behavior Model**: Round-based weighting (90% rankings early → 80% needs late)
- **Monte Carlo Engine**: Pure simulation logic with probabilistic projections and strategy evaluation
- **Championship DNA System**: Pattern discovery from top-performing rosters with tier-based guidance
- **Dynamic Replacement**: Per-simulation replacement level calculations using sampled values
- **Live Draft Integration**: backup_draft.py ↔ monte_carlo_state.json ↔ AI recommendations

## Project Structure
```
backup_draft.py         # Primary draft tracker with real-time state export
monte_carlo_runner.py   # Clean interface for simulations
src/
  monte_carlo/          # Modular simulation system
    probability.py      # ESPN/ADP probability calculations + envelope sampling
    opponent.py         # Opponent behavior modeling
    simulator.py        # Pure simulation engine with probabilistic projections + CRN
    crn_manager.py      # Common Random Numbers for variance reduction
    strategies.py       # Strategy definitions (data only)
    replacement.py      # Dynamic replacement level calculations
    __init__.py        # Clean public API
  scraping.py          # FantasyPros data collection
  scoring.py           # Fantasy point calculations
notebooks/              # Championship DNA system (NEW)
  championship_dna_analyzer.py    # Core pattern analysis from top performers
  championship_dna_notebook.ipynb # Interactive analysis notebook
  championship_dna_hybrid.py      # Complete hybrid system implementation
  run_championship_dna.py         # Simple runner for pattern discovery
config/
  league-config.yaml   # League settings & team names
data/
  draft/               # Live draft state & monte_carlo_state.json
  espn_projections_*.csv   # ESPN rankings
  fantasypros_adp_*.csv    # ADP data
```

## Performance Optimizations (NEW)

### Achieved Performance
- **Single-threaded**: 3.0 sims/sec (baseline)
- **4 cores parallel**: 11.1 sims/sec (3.8x speedup) ✅
- **6 cores parallel**: 13.7 sims/sec (4.6x speedup)
- **Target met**: 1000 sims in ~88 seconds with 4 cores

### Key Optimizations
1. **Vectorized Batch Sampling**: Pre-sample all player projections once per simulation (not 196 times)
2. **Dict Caching**: O(1) player lookups instead of O(n) DataFrame operations
3. **CPU-Friendly Parallelization**: ProcessPoolExecutor with 4 workers default
4. **Unified Methods**: Consolidated duplicate code, reduced complexity by 9%

### Parallel Execution
```bash
# CPU-friendly (4 cores, 50% CPU usage)
python monte_carlo_runner.py balanced --n-sims 1000 --parallel

# More cores if needed (75% CPU usage)
python monte_carlo_runner.py balanced --n-sims 1000 --parallel --workers 6

# Adaptive CRN mode (variance reduction + auto-stop)
python monte_carlo_runner.py compare_fast
```

## Core Algorithms

### Probability System (80/20 Weighted)
```python
# Softmax conversion with temperature control
P(rank_i) = exp(-rank_i / τ) / Σ(exp(-rank_j / τ))  # τ=5.0 default

# Multi-source integration
Combined = 0.8 × ESPN_softmax + 0.2 × ADP_softmax

# Discrete survival probability
for each_pick_until_next_turn:
    survival *= (1 - P_player_picked_now)
probability_gone = 1 - survival
```

### Common Random Numbers (CRN) System
```python
# Pre-generate ALL random samples once for variance reduction
class CRNManager:
    player_samples = {}     # {player_id: array(n_sims,)} - Beta-PERT samples
    team_multipliers       # array(n_teams, n_sims) for correlation  
    opponent_seeds         # array(n_sims,) for behavior consistency

# Memory bounds checking (warns if >200MB allocation)
estimated_mb = n_max_sims * 300 * 8 / 1_000_000

# Error handling for missing players
def get_projection(player_id, sim_idx, low, base, high):
    if player_id not in player_samples:
        print(f"Warning: Unknown player {player_id}, using base projection")
        return base

# Adaptive stopping with confidence intervals  
while n_sims < n_max:
    ci_half_width = 1.96 * se
    if ci_half_width < 3.0 OR ci_half_width < 5% of mean:
        CONVERGED = True

# Performance: 40-60% variance reduction vs independent sampling
```

### Envelope Sampling (Beta-PERT)
```python
# Beta-PERT distribution for projection uncertainty
alpha = 1 + concentration × (base - low) / (high - low)
beta = 1 + concentration × (high - base) / (high - low)
sampled_value = low + beta_sample(alpha, beta) × (high - low)

# Auto-generation if no envelope data
low = projection × 0.8
base = projection  
high = projection × 1.2
```

### Monte Carlo Simulation Engine
**What It Does**: Evaluates draft strategies through simulation with realistic opponent behavior

**Core Algorithm:**
```python
# For each strategy (Balanced, Zero-RB, RB-Heavy, Hero-RB, Elite-QB):
1. Load ESPN/ADP data → Calculate 80/20 weighted probabilities + envelope data
2. Sample player projections using Beta-PERT distributions (LOW/BASE/HIGH)
3. Model opponent behavior (rankings-based early → needs-based late)
4. Simulate 100+ complete drafts with variable projections per simulation
5. Calculate roster value from optimal starting lineup using sampled values
6. Return expected value, variance, and common draft patterns
```

**Key Features:**
- **Probabilistic Projections**: Beta-PERT sampling from envelope data for realistic uncertainty
- **Modular Design**: Separate probability, opponent, simulation, and replacement modules
- **Opponent Modeling**: Round-based weighting (90/10 early → 20/80 late)
- **Strategy Testing**: 5 distinct strategies with position multipliers
- **Dynamic Replacement**: Per-simulation replacement levels using sampled values
- **Live Integration**: Auto-loads draft state from backup_draft.py
- **Clean API**: Simple runner script with strategy comparison

## Championship DNA System: A Philosophical Evolution

### The Core Philosophy Shift

**OLD THINKING**: "Follow Zero-RB strategy" → Rigid rules that break under draft chaos  
**NEW THINKING**: "Winners draft 4 RBs (≥2 Tier-2+), 5 WRs (≥3 Tier-2+)" → Flexible blueprint based on actual winning patterns

This isn't just a feature update—it's a fundamental rethinking of how to approach fantasy drafts. We've moved from **prescribing strategies** to **discovering winning patterns**.

### Three Core Philosophies

#### 1. From Rigid Strategies to "North Star" Blueprint
- **What Changed**: No more blind adherence to "Zero-RB" or "Hero-RB" labels
- **New Approach**: Data-driven optimal roster composition from top 10% of simulations
- **Why It Works**: Based on proven "Championship DNA" from thousands of winning teams in YOUR league settings
- **The Insight**: Provides flexible end-goal, not rigid rules—you adapt to draft flow while knowing your destination

#### 2. From Rankings to Actionable Tier-Based Guidance  
- **What Changed**: Stop obsessing over pick #47 vs #52
- **New Approach**: Dynamic tier windows with probabilities ("62% chance for Tier-2 RB by Round 3")
- **Why It Works**: Tiers are resilient to small ranking changes and provide soft, probabilistic guidance
- **The Insight**: "Best player available" changes based on scarcity and your roster—the system knows this

#### 3. From Brittle Plans to Dynamic Pivot Rules
- **What Changed**: No more panic when your target gets sniped
- **New Approach**: Explicit pivot rules that trigger on draft deviations
- **Why It Works**: Prepares you for imperfection with clear, logical alternatives
- **The Insight**: System doesn't say "you're off-track"—it says "here's your best recovery path"

### The Hybrid System in Practice

```bash
# Discover YOUR league's winning patterns (not generic advice)
PYTHONPATH=. uv run python notebooks/run_championship_dna.py --pick 5 --sims 1000

# Get tier-based guidance with pivot rules
PYTHONPATH=. uv run python notebooks/championship_dna_hybrid.py pre-draft --pick 5

# Live adaptation during draft chaos
PYTHONPATH=. uv run python notebooks/championship_dna_hybrid.py live --pick 5
```

### What You Get: Three Actionable Cards

1. **🎯 North Star Blueprint**
   ```
   Your Championship Target:
   RB: 4 players (≥2 Tier-2+)
   WR: 5 players (≥3 Tier-2+)  
   TE: 2 players (≥1 Tier-3+)
   QB: 2 players
   Success Rate: 43% of champions look like this
   ```

2. **📊 Tier Windows (Soft Targets)**
   ```
   Round 3 Probabilities:
   RB Tier-2: 62% available → GOOD WINDOW
   WR Tier-1: 8% available → CLOSING FAST
   Recommendation: Prioritize WR if Tier-1 available
   ```

3. **⚠️ Pivot Alerts (Adaptive Rules)**
   ```
   POSITION RUN DETECTED: 5 RBs in last 6 picks
   → Pivot to WR/TE for next 2 rounds
   → Return to RB in Round 5 (82% Tier-3 availability)
   ```

### The Philosophy Summary

**This system uses your simulation engine not to prescribe a single strategy, but to discover a universe of winning outcomes.** It translates these into:
- An ideal "North Star" roster (where you're heading)
- Early-round "soft targets" (probabilistic guidance, not rigid rules)  
- Clear "pivot alerts" (keeping you on track when chaos hits)

**Bottom Line**: This is about using data to inform an intelligent, adaptive process—not following blind recommendations. You maintain human judgment while being armed with powerful pattern recognition from thousands of simulated championships.

## Commands

### Draft Operations
```bash
# Pre-draft strategy comparison
python monte_carlo_runner.py compare                              # Traditional (100 sims, ~30s)
python monte_carlo_runner.py compare --n-sims 1000 --parallel     # Fast parallel (1000 sims, ~88s)
python monte_carlo_runner.py compare_fast                         # CRN adaptive (auto-stop at convergence)

# Championship DNA analysis (NEW)
PYTHONPATH=. uv run python notebooks/run_championship_dna.py --pick 5 --sims 1000  # Pattern discovery
PYTHONPATH=. uv run python notebooks/championship_dna_hybrid.py pre-draft --pick 5  # Hybrid recommendations

# Live draft workflow
python backup_draft.py                    # Start draft tracker
  → ORDER    # Show snake draft order
  → STATUS   # Current draft state
  → HELP     # All commands

python monte_carlo_runner.py degradation                 # BEST: Positional degradation analysis
python monte_carlo_runner.py balanced                    # Quick recommendations (100 sims)
python monte_carlo_runner.py balanced --n-sims 1000 --parallel  # Thorough analysis (1000 sims)
python monte_carlo_runner.py zero_rb --parallel          # Test specific strategy with speed

# Live Championship DNA recommendations
PYTHONPATH=. uv run python notebooks/championship_dna_hybrid.py live --pick 5      # Live tier-based guidance
```

### Live Draft Integration
```python
# backup_draft.py automatically exports state:
{
  "my_team_idx": 4,              # 0-based team index
  "current_global_pick": 17,     # Current pick number  
  "my_current_roster": [...],    # Your drafted players
  "all_drafted": [...],          # All drafted players
  "team_name": "Your Team"       # Team name from config
}

# Monte Carlo auto-loads this state and filters already-drafted players
python monte_carlo_runner.py balanced
# → "📡 Loaded draft state: Pick #18, 3 players drafted"
# → "Mode: LIVE DRAFT"
```

## Configuration

### League Settings (`config/league-config.yaml`)
```yaml
basic_settings:
  teams: 14
  draft_type: "snake"

team_names:
  - "Team 1"
  - "Team 2"
  # ... 14 teams total

scoring:
  passing_td: 4
  rushing_td: 6
  # PPR/Half-PPR/Standard configurable

roster:
  QB: 1, RB: 2, WR: 2, TE: 1, FLEX: 1, K: 1, DST: 1
```

## Starter Optimizer System (North Star Aligned)

### Architecture Refactoring: 37% Complexity Reduction
The massive 1041-line `starter_optimizer.py` has been **modularized into 4 focused modules** while preserving all functionality and maintaining <2s performance:

```python
# Before: Single 1041-line file with mixed concerns
starter_optimizer.py  # Everything in one massive file

# After: Clean 4-module architecture
starter_core.py        # Core North Star logic (MSG, OC, Starter Sum) 
probability_models.py  # ESPN probability models and caching
optimizer_utils.py     # Data loading and utilities
starter_optimizer.py   # Clean main interface (simplified)
```

### North Star Principle Alignment
**Core Question**: "Does this increase expected starter lineup points under uncertainty, relative to waiting?"

**MSG-OC Decision Framework**:
```python
# Marginal Starter Gain: Improvement from adding candidate
MSG = StarterSum(with_candidate) - StarterSum(without_candidate)

# Opportunity Cost: Cost of picking now vs waiting
OC = StarterSum(pick_now) - StarterSum(pick_at_next_turn)  

# North Star Decision Score
Score = E[MSG] - E[OC]
```

**Key Benefits**:
- **Starter Sum Focus**: Optimizes actual starting lineup points, not VOR heuristics
- **Probabilistic Reasoning**: Handles uncertainty through envelope sampling
- **Modular Design**: Clean separation enables testing and maintenance
- **Performance Maintained**: <2s execution time preserved through smart caching

### Module Responsibilities

#### `starter_core.py` - Pure North Star Logic
- `compute_starter_sum()`: Calculate total points from optimal starting lineup
- `marginal_starter_gain()`: Expected improvement when adding a candidate
- `compute_opportunity_cost_ss()`: Cost of picking now vs waiting
- **Philosophy**: No hardcoded heuristics, pure mathematical optimization

#### `probability_models.py` - Sophisticated Probability Modeling
- `ExpectedBestSimulator`: Cached probability calculations with ESPN data
- `sophisticated_pick_probability()`: Actual ESPN rankings for pick probabilities  
- `softmax_weights()`: Safe probability conversion with overflow protection
- **Philosophy**: Replace assumptions with real draft data

#### `optimizer_utils.py` - Data Infrastructure
- `load_players_from_csv()`: Beta-PERT sampling for projection uncertainty
- `beta_pert_samples()`: Realistic variability modeling around projections
- `compute_dynamic_replacement_levels()`: Context-aware replacement calculations
- **Philosophy**: Data-driven uncertainty modeling, not static assumptions

#### `starter_optimizer.py` - Clean Interface
- `pick_best_now()`: Main decision function with performance monitoring
- Smart candidate filtering and adaptive scenario counts
- Cache management and performance warnings
- **Philosophy**: Simple public API hiding complex optimization logic

## Conventions
- **Data Sources**: ESPN (80% weight) + ADP (20% weight) for robustness
- **Envelope Data**: Beta-PERT sampling from LOW/BASE/HIGH or auto-generated ±20%
- **Opponent Modeling**: Round-based behavior (rankings early → needs late)
- **Live Integration**: Automatic state sync between backup_draft.py and Monte Carlo
- **Modular Design**: 
  - Monte Carlo: Separate probability, opponent, simulation, replacement, and strategy modules
  - Starter Optimizer: 4-module North Star aligned architecture (37% complexity reduction)
- **Performance**: 
  - Single-threaded: 3.0 sims/sec (~333s for 1000 sims)
  - 4-core parallel: 11.1 sims/sec (~88s for 1000 sims) ✅
  - CRN adaptive: Auto-stops at convergence with 40-60% variance reduction
- **Testing**: Always run with `PYTHONPATH=.` prefix

## Do's
- Run strategy comparison BEFORE draft (`PYTHONPATH=. uv run python monte_carlo_runner.py compare_fast`)
- Use Championship DNA for pattern discovery (`PYTHONPATH=. uv run python notebooks/run_championship_dna.py --pick 5 --sims 1000`)
- Use `--parallel` flag for 1000+ simulations (4x speedup)
- Set draft position in backup_draft.py team selection menu
- Use live integration during draft (backup_draft.py → Monte Carlo)
- Try hybrid tier-based recommendations (`PYTHONPATH=. uv run python notebooks/championship_dna_hybrid.py pre-draft --pick 5`)
- Use UV environment (`uv sync`)
- Test system before draft day
- Always prefix commands with `PYTHONPATH=.`
- Run from project root directory

## Don'ts
- Modify monte_carlo_state.json manually (corrupts integration)
- Run without PYTHONPATH=. prefix (will cause import errors)
- Trust single ranking source alone
- Use pip install directly (use `uv add` instead)
- Run scripts from subdirectories
- Use virtual environments other than UV's

## Data Flow
```
1. Draft Tracking (backup_draft.py)
   ↓
2. Auto-export to monte_carlo_state.json (real-time)
   ↓
3. Monte Carlo auto-loads state & filters drafted players
   ↓
4. AI strategy recommendations (context-aware)
```

## Testing Guide

### Running Tests
```bash
# ALWAYS use these exact commands:
PYTHONPATH=. uv run python -m pytest tests/test_backup_draft.py -v   # Full test suite
PYTHONPATH=. uv run python -m pytest tests/ -v                       # All tests
PYTHONPATH=. uv run python monte_carlo_runner.py balanced --n-sims 5 # Quick integration test

# Never use:
pytest tests/                           # Wrong - missing PYTHONPATH
python -m pytest tests/                 # Wrong - not using UV environment
./tests/test_backup_draft.py           # Wrong - direct execution
```

### Test Coverage
- **Unit Tests**: `tests/test_backup_draft.py` (7 tests)
- **Monte Carlo Integration**: Run `monte_carlo_runner.py` with small n-sims
- **Starter Optimizer**: Test modular system with `python src/monte_carlo/starter_optimizer.py`
- **Envelope System**: Automatically tested when running any simulation
- **Live Draft**: Test with `backup_draft.py` → enter picks → check state file

## Debugging

### Common Problems
- **Module not found**: Use `PYTHONPATH=.` prefix
- **"No module named 'src'"**: You're not in project root or missing PYTHONPATH
- **State file corrupted**: Monte Carlo shows error, continues with defaults
- **Player not found**: Check name spelling in backup_draft.py
- **Import errors**: Run `uv sync`
- **Test failures**: Ensure you're using `uv run` not bare `python`

### Validation Checklist
- [ ] Strategy comparison runs (`python monte_carlo_runner.py compare`)
- [ ] Draft tracker saves state (`backup_draft.py` → `monte_carlo_state.json`)
- [ ] Monte Carlo loads live state (shows "📡 Loaded draft state")
- [ ] All tests pass (`pytest tests/test_backup_draft.py -v`)

## Technical Reference

### Statistical Methodology
**Softmax Temperature (τ=5.0)**: Controls probability concentration. Lower τ = sharper peaks on top players.

**80/20 Weighting**: ESPN reflects current consensus, ADP provides stability.

**Opponent Modeling**: Round-based transition from rankings (90%) to needs (80%) by round 7.

### North Star Methodology (Starter Optimizer)
**Decision Framework**: MSG (Marginal Starter Gain) - OC (Opportunity Cost) = Decision Score

**Starter Sum Optimization**: Focus on actual starting lineup points rather than VOR heuristics.

**Modular Architecture**: 37% complexity reduction through focused separation of concerns.

### Envelope System (Beta-PERT Sampling)
**Purpose**: Add realistic variability to player projections instead of static values.

**How It Works**:
- Loads LOW/BASE/HIGH projections from `data/player_envelopes.csv` if available
- Auto-creates ±20% envelopes from base projections if not available
- Uses Beta-PERT distribution (concentration=4.0) for realistic sampling
- Just-in-time sampling to optimize memory usage

**Files Involved**:
- Monte Carlo system: `probability.py`, `replacement.py`, `simulator.py`
- Starter Optimizer: `optimizer_utils.py` (Beta-PERT sampling), `starter_core.py` (Starter Sum)

### Module Architecture
```python
# Monte Carlo simulation modules (clean separation of concerns)
src/monte_carlo/
├── probability.py         # ESPN/ADP probabilities only
├── opponent.py            # Behavior modeling only  
├── simulator.py           # Pure simulation logic
├── strategies.py          # Strategy data (no logic)
├── crn_manager.py         # Common Random Numbers
├── replacement.py         # Dynamic replacement calculations
└── __init__.py           # Public API

# Starter optimizer modules (North Star aligned, modular)
├── starter_core.py        # Core MSG-OC logic & Starter Sum calculations
├── probability_models.py  # ESPN probability models & ExpectedBestSimulator
├── optimizer_utils.py     # Data loading, Beta-PERT sampling, utilities
└── starter_optimizer.py   # Clean main interface (simplified from 1041 lines)
```

**System Usage**:
- **Monte Carlo**: Full draft strategy evaluation and comparison (`monte_carlo_runner.py`)
- **Starter Optimizer**: Individual pick optimization during live drafts (North Star aligned)
- **Integration**: Monte Carlo provides strategy guidance, Starter Optimizer provides pick-by-pick decisions

### File Formats
```
# ESPN format
player_name,position,overall_rank

# ADP format  
PLAYER,RANK,POSITION

# Envelope format (optional)
player_name,low,base,high

# State export (from backup_draft.py)
monte_carlo_state.json
```

---

This system provides modular Monte Carlo simulation for optimal fantasy football draft strategy evaluation and Championship DNA pattern discovery for identifying winning roster construction patterns.
- test_starter_simple.py to be used for development simple probability model that is deterministic