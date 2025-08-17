# VBD (Value Based Drafting) Research & Implementation Plan

## Executive Summary

Value Based Drafting (VBD) is a fantasy football draft strategy that assigns relative value to players by comparing their projected fantasy points against a "baseline" player at the same position. The key insight is that a player's absolute point total is less important than how much better they are than readily available alternatives.

## VBD Baseline Methods

### 1. VOLS (Value Over Last Starter)
**Formula:** `baseline_rank = teams × starters_at_position`

**Logic:** The baseline is the last starting player at each position across all teams.

**Example (14-team league):**
- QB (1 starter): baseline = 14th QB
- RB (2 starters): baseline = 28th RB  
- WR (2 starters): baseline = 28th WR
- TE (1 starter): baseline = 14th TE
- K (1 starter): baseline = 14th K
- DST (1 starter): baseline = 14th DST

**When to use:** Best for identifying premium starters and understanding positional scarcity at the starter level.

### 2. VORP (Value Over Replacement Player)
**Formula:** `baseline_rank = teams × (starters_at_position + bench_factor)`

Where `bench_factor` typically equals 1 (first bench player).

**Example (14-team league):**
- QB: baseline = 14 × (1 + 1) = 28th QB
- RB: baseline = 14 × (2 + 1) = 42nd RB
- WR: baseline = 14 × (2 + 1) = 42nd WR  
- TE: baseline = 14 × (1 + 1) = 28th TE

**When to use:** Better for evaluating depth and bench value, considers waiver wire availability.

### 3. BEER (Best Eleven Every Round)
**Formula:** `baseline_rank ≈ teams × (starters + 0.5)`

**Logic:** Balances between VOLS and VORP, accounting for the fact that not every team drafts backup at every position every round.

**Example (14-team league):**
- QB: baseline = 14 × 1.5 = 21st QB
- RB: baseline = 14 × 2.5 = 35th RB
- WR: baseline = 14 × 2.5 = 35th WR
- TE: baseline = 14 × 1.5 = 21st TE

**When to use:** General purpose baseline that balances starter and bench considerations.

### 4. Blended VBD
**Formula:** Weighted combination of multiple methods

**Example:** `VBD_Blended = 0.5 × VBD_BEER + 0.25 × VBD_VORP + 0.25 × VBD_VOLS`

**When to use:** Most comprehensive approach, reduces impact of any single method's weaknesses.

## Implementation Architecture

### Core Components

#### 1. Baseline Calculator
```python
class BaselineCalculator:
    def __init__(self, config):
        self.teams = config['basic_settings']['teams']
        self.roster_slots = config['roster']['roster_slots']
        
    def calculate_vols_baseline(self):
        """Teams × starters at position"""
        
    def calculate_vorp_baseline(self, bench_factor=1):
        """Teams × (starters + bench_factor)"""
        
    def calculate_beer_baseline(self):
        """Teams × (starters + 0.5)"""
        
    def calculate_custom_baseline(self, multiplier):
        """Teams × (starters × multiplier)"""
```

#### 2. VBD Engine
```python
class VBDEngine:
    def __init__(self, player_data, baseline_calculator):
        self.data = player_data
        self.baseline_calc = baseline_calculator
        
    def calculate_vbd(self, method='beer'):
        """Calculate VBD for all players using specified method"""
        
    def calculate_all_vbd_methods(self):
        """Calculate VBD using all methods"""
        
    def calculate_blended_vbd(self, weights):
        """Calculate weighted average of multiple VBD methods"""
```

#### 3. Position Handler
```python
class PositionHandler:
    # Handle special cases
    FLEX_POSITIONS = ['RB', 'WR', 'TE']
    
    def adjust_for_flex(self, baselines):
        """Adjust baselines considering FLEX spots"""
        
    def handle_positional_scarcity(self, position_data):
        """Apply scarcity adjustments"""
```

## Detailed Calculations

### Step 1: Determine Baseline Ranks
For each position and method:

```python
def get_baseline_rank(position, method, config):
    teams = config['basic_settings']['teams']  # 14
    starters = config['roster']['roster_slots'][position]
    
    if method == 'vols':
        return teams * starters
    elif method == 'vorp':
        return teams * (starters + 1)
    elif method == 'beer':
        return int(teams * (starters + 0.5))
    elif method == 'custom':
        # Use config replacement_level
        return config['replacement_level'][position]
```

### Step 2: Get Baseline Points
```python
def get_baseline_points(df, position, baseline_rank):
    pos_players = df[df['Position'] == position].copy()
    pos_players = pos_players.sort_values('Fantasy_Points', ascending=False)
    
    if len(pos_players) >= baseline_rank:
        # Use the player at baseline_rank
        baseline_points = pos_players.iloc[baseline_rank - 1]['Fantasy_Points']
    else:
        # Not enough players, use last available
        baseline_points = pos_players.iloc[-1]['Fantasy_Points']
    
    return baseline_points
```

### Step 3: Calculate VBD
```python
def calculate_vbd_for_position(df, position, baseline_points):
    pos_mask = df['Position'] == position
    df.loc[pos_mask, 'VBD'] = df.loc[pos_mask, 'Fantasy_Points'] - baseline_points
    return df
```

### Step 4: Handle FLEX Position
```python
def adjust_for_flex(df, config):
    flex_slots = config['roster']['roster_slots'].get('FLEX', 0)
    if flex_slots > 0:
        # FLEX can be RB, WR, or TE
        # Adjust baselines considering additional flex spots
        flex_positions = ['RB', 'WR', 'TE']
        # Complex calculation considering cross-position value
```

## Edge Cases & Considerations

### 1. Insufficient Players
- If fewer players than baseline rank exist, use the last available player
- Log warning for positions with insufficient depth

### 2. FLEX Position Handling
- FLEX spots affect RB, WR, and TE baselines
- Consider using composite baseline for FLEX-eligible positions

### 3. Superflex/2QB Leagues
- QB baseline dramatically changes
- May need separate calculation method

### 4. Dynasty vs Redraft
- Dynasty leagues need age-adjusted VBD
- Consider multi-year projections

## Output Format

### DataFrame Structure
```
Player | Position | Team | Fantasy_Points | VBD_VOLS | VBD_VORP | VBD_BEER | VBD_Blended | VBD_Rank
```

### Analysis Views
1. **Overall Rankings**: All positions sorted by VBD_Blended
2. **Positional Rankings**: Within-position VBD comparisons  
3. **Tier Analysis**: Natural VBD breakpoints
4. **Value Gaps**: Identify positional cliffs

## Implementation Timeline

### Phase 1: Core VBD Calculations (Immediate)
- Implement VOLS, VORP, BEER methods
- Basic VBD calculation for all players
- Simple output format

### Phase 2: Advanced Features (Next)
- Blended VBD with configurable weights
- FLEX position handling
- Position scarcity adjustments

### Phase 3: Analysis & Visualization (Future)
- VBD tier identification
- Draft strategy recommendations
- Interactive VBD explorer

## Code Integration Points

### Current Codebase
- `analyze_projections.ipynb`: Main analysis notebook
- `config/league-config.yaml`: League settings including replacement levels
- Existing `calculate_fantasy_points_vectorized()` function provides point totals

### New Components to Add
1. `vbd_calculator.py`: Core VBD calculation module
2. Update notebook with VBD calculation cells
3. Add VBD columns to output DataFrames
4. Create VBD comparison visualizations

## Recommended Defaults

For a 14-team league with standard roster (1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX):

### Baseline Ranks by Method
| Position | VOLS | VORP | BEER | Custom (Current) |
|----------|------|------|------|------------------|
| QB       | 14   | 28   | 21   | 14               |
| RB       | 28   | 42   | 35   | 28               |
| WR       | 28   | 42   | 35   | 28               |
| TE       | 14   | 28   | 21   | 14               |
| K        | 14   | 28   | 21   | 14               |
| DST      | 14   | 28   | 21   | 14               |

### Recommended Blended Weights
- BEER: 50% (balanced approach)
- VORP: 25% (bench/depth consideration)
- VOLS: 25% (starter value emphasis)

## References & Sources

1. **Subvertadown**: Comprehensive VBD baseline comparison, BEER methodology
2. **FantasyPros**: VBD implementation, VORP/VOLS/VONA definitions
3. **Fantasy Football Analytics**: R-based VOR calculations, statistical approach
4. **Industry Standard**: Most major platforms use variations of these methods

## Next Steps

1. Review this research document
2. Decide on which methods to implement (recommend all four)
3. Create modular implementation in existing notebook
4. Test with current player projections
5. Generate comparative VBD rankings
6. Analyze differences between methods
7. Create draft strategy recommendations based on VBD tiers