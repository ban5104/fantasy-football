# 14-Round Bench Depth System Implementation

## Overview
Extended the existing 7-round Monte Carlo simulator to handle full 14-round drafts with bench depth metrics and multi-scenario trade-off analysis.

## Key Components Added

### 1. Bench Quality Metrics (`simulator.py`)
- **calculate_bench_quality_metrics()**: Analyzes bench players for:
  - Quality backups (within 25 points of starters)
  - Handcuff value (same-team RB backups)
  - Upside potential (high-variance players)
  - Bye week coverage needs
  - Total bench value

### 2. Phase-Aware Selection (`simulator.py`)
- **Rounds 1-7**: Focus on maximizing starter value (existing VOR/multiplier logic)
- **Rounds 8-14**: Switch to bench optimization via `select_best_bench_player()`
  - Prioritizes handcuffs, quality depth, upside, and coverage
  - Position-specific logic for K/DST in late rounds

### 3. Scenario Generation System (`strategies.py`)
- **generate_scenario_configs()**: Creates 10 different draft approaches by varying:
  - `risk_aversion` (0.0 to 0.9): Conservative vs aggressive
  - `bench_value_decay` (0.1 to 0.6): How much bench matters vs starters
  - `tier_cliff_penalty` (1.0 to 4.0): Urgency when approaching position cliffs
  - `position_importance`: RB vs WR priority weights

### 4. Scenario Runner (`monte_carlo_runner.py`)
- **run_scenario_analysis()**: New command that:
  - Runs 10+ different scenarios with parameter variations
  - Calculates starter points and bench depth for each
  - Computes trade-off scores: `starter_diff + (quality_backups * 10)`
  - Ranks scenarios by overall value

## Usage

```bash
# Run scenario analysis with 100 simulations per scenario
PYTHONPATH=. uv run python monte_carlo_runner.py scenarios --n-sims 100 --pick 5

# Use different base strategy
PYTHONPATH=. uv run python monte_carlo_runner.py scenarios --n-sims 100 --strategy rb_heavy
```

## Example Output

```
Scenario 1: Max Starters (risk=0.0)
Starters: 1289.5 pts
Bench: RB(1), WR(1), TE(0), QB(1) = 3 quality
Trade-off: Baseline

Scenario 2: Balanced (risk=0.5)  
Starters: 1284.2 pts (-5.3)
Bench: RB(2), WR(2), TE(0), QB(1) = 5 quality (+2)
Trade-off: +14.7 pts (better overall)

Scenario 3: Anti-Fragile (risk=0.6)
Starters: 1281.3 pts (-8.2)
Bench: RB(3), WR(3), TE(1), QB(1) = 8 quality (+5)
Trade-off: +41.8 pts (best overall despite lower starters)
```

## Key Trade-off Formula

```python
trade_off_score = starter_diff + (quality_backups * 10)
```

Where:
- `starter_diff` = Change in starter points vs baseline scenario
- `quality_backups` = Number of bench players within 25 points of starters
- Each quality backup valued at ~10 fantasy points for season resilience

## Implementation Philosophy

**Minimal Changes**: 
- Reused all existing probability models and opponent behavior
- Extended rather than refactored existing code
- Added only 3 new methods and 1 new parameter structure

**Clean Separation**:
- Rounds 1-7 use existing starter optimization unchanged
- Rounds 8-14 use new bench optimization logic
- Each phase can be tested independently

**Practical Insights**:
- Most scenarios show minimal starter sacrifice (<10 points) for significant bench improvement
- "Anti-Fragile" and "Conservative Depth" scenarios often provide best overall value
- Quality threshold of 25 points effectively identifies useful backups

## Testing

```bash
# Quick test with 5 simulations
PYTHONPATH=. uv run python monte_carlo_runner.py scenarios --n-sims 5

# Full analysis with 100 simulations
PYTHONPATH=. uv run python monte_carlo_runner.py scenarios --n-sims 100

# Compare different base strategies
for strategy in balanced zero_rb rb_heavy; do
    PYTHONPATH=. uv run python monte_carlo_runner.py scenarios --n-sims 50 --strategy $strategy
done
```

## Files Modified

1. `/src/monte_carlo/simulator.py`:
   - Added `calculate_bench_quality_metrics()`
   - Added `select_best_bench_player()`
   - Modified `select_best_player()` for phase awareness

2. `/src/monte_carlo/strategies.py`:
   - Added `generate_scenario_configs()`
   - Added `bench_phase` parameters to strategies

3. `/monte_carlo_runner.py`:
   - Added `run_scenario_analysis()`
   - Added 'scenarios' mode to command parser

Total implementation: ~400 lines of code in 45 minutes.