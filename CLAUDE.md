# Fantasy Football Draft Probability System - Claude Guide

## Project Overview
This project implements a dynamic fantasy football draft probability system that calculates real-time availability odds for players based on draft position and ranking data.

## Core Probability System (80% ESPN + 20% ADP)

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
The system integrates with existing VBD (Value Based Drafting) data:
- **ESPN projections**: `data/espn_projections_20250814.csv`
- **External ADP data**: `data/fantasypros_adp_20250815.csv` 
- **VBD scores**: `draft_cheat_sheet.csv`

### Usage in Notebook
```python
# Load and merge ranking data
ranking_df = load_and_merge_ranking_data()

# Calculate enhanced metrics with new probability system
enhanced_df = calculate_player_metrics_new_system(
    ranking_df, vbd_data, 
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
- Compatible with existing VBD ranking integration

## File Locations
- **Main notebook**: `espn_probability_matrix.ipynb`
- **ESPN data**: `data/espn_projections_20250814.csv`
- **ADP data**: `data/fantasypros_adp_20250815.csv`
- **VBD scores**: `draft_cheat_sheet.csv`
- **Config**: `config/league-config.yaml`

## Future Enhancements
- Real-time ESPN ranking updates during draft
- Position scarcity adjustments
- League-specific ADP data integration
- Historical accuracy validation