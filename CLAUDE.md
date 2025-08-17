# Fantasy Football Draft Optimization System - Technical Documentation

## Project Overview
This project implements two complementary systems:
1. **Probability System**: Real-time player availability predictions using weighted rankings
2. **Monte Carlo Draft Optimizer**: Global roster optimization using advanced simulation techniques

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
- Compatible with existing VBD ranking integration

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