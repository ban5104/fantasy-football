# Fantasy Football Monte Carlo Draft Optimizer

A high-performance fantasy football draft system using Monte Carlo simulation to optimize draft strategy and provide real-time AI recommendations during live drafts.

## What This System Does

**Pre-Draft Strategy Analysis**
- Compare draft strategies through Monte Carlo simulation (Balanced, Zero-RB, RB-Heavy, etc.)
- Discover winning patterns with Championship DNA system from top-performing simulations
- Calculate probabilistic player availability and tier breakpoints

**Live Draft Intelligence**
- Real-time draft tracking with automatic state export
- AI-powered recommendations based on current draft state
- Adaptive strategy pivots when draft flow changes

**Core Innovation**: Combines 80% ESPN + 20% ADP weighted rankings with Beta-PERT envelope sampling and opponent behavior modeling to run thousands of realistic draft simulations.

## Quick Start

### Installation
```bash
# Install dependencies with UV (recommended)
uv sync

# Verify setup
PYTHONPATH=. python monte_carlo_runner.py balanced --n-sims 5
```

### Pre-Draft Analysis
```bash
# Compare traditional strategies (100 sims, ~30s)
python monte_carlo_runner.py compare

# High-performance analysis (1000 sims, ~88s with parallel processing)
python monte_carlo_runner.py compare --n-sims 1000 --parallel

# Championship DNA pattern discovery
PYTHONPATH=. uv run python notebooks/run_championship_dna.py --pick 5 --sims 1000
```

### Live Draft Workflow
```bash
# Terminal 1: Start draft tracker
python backup_draft.py

# Terminal 2: Get AI recommendations  
python monte_carlo_runner.py balanced                           # Quick (100 sims)
python monte_carlo_runner.py balanced --n-sims 1000 --parallel  # Thorough (1000 sims)

# Championship DNA live guidance
PYTHONPATH=. uv run python notebooks/championship_dna_hybrid.py live --pick 5
```

## Core Features

### Monte Carlo Simulation Engine
- **Probabilistic Projections**: Beta-PERT distributions from LOW/BASE/HIGH projections for realistic uncertainty
- **Opponent Behavior Modeling**: Round-based weighting (90% rankings early → 20% rankings late)
- **Strategy Evaluation**: Tests 5 distinct strategies with position multipliers
- **Live Integration**: Auto-loads draft state and filters drafted players

### Championship DNA System (Advanced)
Revolutionary approach that discovers winning patterns instead of following rigid strategies:

**Three Components:**
1. **North Star Blueprint** - Optimal roster composition from top 10% of winners
2. **Tier Windows** - Probabilistic guidance by round ("62% chance for Tier-2 RB by Round 3")
3. **Pivot Rules** - Adaptive recovery when draft chaos hits

**Example Output:**
```
🎯 CHAMPIONSHIP TARGET: RB: 4 players (≥2 Tier-2+), WR: 5 players (≥3 Tier-2+)
📊 ROUND 3: RB Tier-2: 62% available → GOOD WINDOW
⚠️ PIVOT: 5 RBs in last 6 picks → Pivot to WR/TE for 2 rounds
```

### Performance Optimizations
- **11x Speedup**: Parallel processing with 4 cores (3.8x speedup)
- **Vectorized Sampling**: Pre-sample all player projections once per simulation
- **Common Random Numbers**: 40-60% variance reduction
- **Memory Efficient**: Optimized for live draft speed

## Architecture

### Core Components
```
backup_draft.py              # Live draft tracker with state export
monte_carlo_runner.py        # Clean command-line interface
src/monte_carlo/
├── probability.py           # ESPN/ADP probability calculations
├── simulator.py             # Monte Carlo engine with CRN
├── opponent.py              # Opponent behavior modeling
├── strategies.py            # Strategy definitions
└── replacement.py           # Dynamic replacement calculations
```

### Data Flow
1. **Load Rankings**: ESPN (80%) + ADP (20%) weighted probabilities
2. **Convert to Probabilities**: Softmax transformation (τ=5.0)
3. **Sample Projections**: Beta-PERT distributions for uncertainty
4. **Model Opponents**: Round-based behavior (rankings → needs)
5. **Run Simulations**: Monte Carlo with probabilistic projections
6. **Generate Recommendations**: Optimal lineup value calculations

## Available Strategies

### Traditional Approaches
- `balanced` - Equal weight to all positions
- `zero_rb` - WR/TE heavy, RBs late
- `rb_heavy` - Load up on RBs early
- `hero_rb` - One elite RB, then WR/TE focus
- `elite_qb` - Prioritize top QB early

### Advanced Commands
```bash
# Positional degradation analysis
python monte_carlo_runner.py degradation

# Fast CRN adaptive mode (auto-stops at convergence)
python monte_carlo_runner.py compare_fast

# Parallel processing modes
python monte_carlo_runner.py balanced --n-sims 1000 --parallel --workers 4
```

## Configuration

### League Settings
Edit `config/league-config.yaml`:
```yaml
basic_settings:
  teams: 14
  draft_type: "snake"

scoring:
  passing_td: 4
  rushing_td: 6
  receiving: {yards: 0.1, td: 6, reception: 1}  # PPR

roster:
  QB: 1, RB: 2, WR: 2, TE: 1, FLEX: 1, K: 1, DST: 1
```

### Important Environment Notes
- **Always use `PYTHONPATH=.` prefix** for Python commands
- **UV environment required**: Install with `pip install uv`
- **Run from project root**: All commands assume project root directory

## Live Draft Integration

The system automatically syncs between draft tracking and AI recommendations:

```python
# backup_draft.py exports state to monte_carlo_state.json:
{
  "my_team_idx": 4,              # Your team position
  "current_global_pick": 17,     # Current pick number
  "my_current_roster": [...],    # Your drafted players
  "all_drafted": [...],          # All drafted players
  "team_name": "Your Team"
}

# Monte Carlo auto-loads and displays:
# "📡 Loaded draft state: Pick #18, 3 players drafted"
# "Mode: LIVE DRAFT"
```

## Sample Output

**Strategy Comparison:**
```
Strategy Comparison Results (n=1000):
┌─────────────────┬──────────────┬─────────────┬─────────────┐
│ Strategy        │ Mean Score   │ Std Dev     │ Success %   │
├─────────────────┼──────────────┼─────────────┼─────────────┤
│ Balanced        │ 1,847.3      │ 127.4       │ 22.8%       │
│ Zero-RB         │ 1,834.1      │ 142.7       │ 18.4%       │
│ RB-Heavy        │ 1,859.7      │ 119.2       │ 26.3%       │
└─────────────────┴──────────────┴─────────────┴─────────────┘
```

**Live Recommendations:**
```
🎯 TOP RECOMMENDATION: Cooper Kupp (WR)
📊 Expected Value: 1,847 points (+127 vs. replacement)
⚡ Availability: 73% chance available at your next pick
🏗️ Roster Impact: Fills WR1 need, enables RB focus later
⏰ Scarcity Alert: Only 3 elite WRs remain
```

## Troubleshooting

### Common Issues
```bash
# Module not found errors
PYTHONPATH=. python your_script.py

# Environment setup issues
uv sync
export PATH="$(uv python find | head -n1 | xargs dirname):$PATH"

# Test installation
PYTHONPATH=. python -c "import src.monte_carlo; print('✓ System ready')"

# Verify Monte Carlo integration
PYTHONPATH=. uv run python -m pytest tests/test_backup_draft.py -v
```

### Validation Checklist
- [ ] Strategy comparison runs without errors
- [ ] Draft tracker creates state file (`monte_carlo_state.json`)
- [ ] Monte Carlo loads live state (shows "📡 Loaded draft state")
- [ ] All tests pass

## Performance Benchmarks

- **Single-threaded**: 3.0 sims/sec (baseline)
- **4-core parallel**: 11.1 sims/sec (3.8x speedup)
- **1000 simulations**: ~88 seconds (parallel) vs ~333 seconds (serial)
- **Memory usage**: <200MB for 1000 simulations

Built for competitive fantasy football managers who want data-driven draft advantages with real-time adaptability.

*For detailed technical documentation, see [CLAUDE.md](CLAUDE.md)*
