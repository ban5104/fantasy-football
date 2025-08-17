# Enhanced Probabilistic VBD Framework

## Executive Summary

Upgrade Dynamic VBD from static replacement levels to **real-time probabilistic calculations** that account for draft flow and roster construction needs. Instead of assuming QB #14 is always replacement level, calculate who's realistically available based on selection probabilities and your specific roster gaps.

## Core Formula

```
Utility_i = P_i × (VBD_i - R_i^dynamic) × (1 + β × PNI_p)
```

Where:
- **P_i**: Probability player is available at your next pick
- **VBD_i**: Player's value-based drafting score  
- **R_i^dynamic**: Dynamic replacement level (best player likely to survive next 10-20 picks)
- **PNI_p**: Positional Need Index for your roster
- **β**: Weighting factor for roster needs (suggested 0.3)

## Component Definitions

### 1. Player VBD (VBD_i)
Your existing custom value score from multiple VBD methods (BEER, VORP, VOLS blend).

### 2. Availability Probability (P_i)  
Chance the player survives until your pick, derived from:
- ESPN algorithm predictions
- Historical draft patterns
- Current draft flow analysis

### 3. Dynamic Replacement (R_i^dynamic)
Instead of fixed QB #14/RB #28, calculate realistic replacement level:

```python
def calculate_dynamic_replacement(position, selection_probs, horizon=20):
    """Find realistic replacement based on survival probability"""
    available_players = get_available_at_position(position)
    likely_survivors = [p for p in available_players 
                       if selection_probs[p.name] < 0.3]  # <30% to be taken
    return likely_survivors[0].fantasy_points if likely_survivors else fallback_baseline
```

**Examples:**
- Early draft: Top QBs have 90% selection probability → replacement drops to QB #18
- Mid draft: Remaining QBs have 40% probability → replacement stays QB #12  
- Late draft: Only scrubs left at <10% → replacement rises to QB #25

### 4. Positional Need Index (PNI_p)

Statistical measure of roster urgency, not gut feel:

```python
def calculate_positional_need_index(position, my_roster, selection_probs):
    """Calculate expected shortfall cost for position"""
    # How many slots still needed (starters + reasonable bench)
    slots_needed = get_remaining_slots(position, my_roster)
    
    # Expected supply: sum of (1 - selection_probability) for startable players
    expected_supply = sum([1 - prob for prob in selection_probs 
                          if prob < 0.5])  # Players likely available
    
    # Shortfall: positive means you'll likely fall short
    shortfall = max(0, slots_needed - expected_supply)
    
    # Cost: VBD gap between startable and replacement level
    shortfall_cost = calculate_vbd_gap_at_position(position)
    
    return shortfall * shortfall_cost
```

**Components:**
- **Remaining slots (S_p)**: Starters + FLEX eligibility you still need
- **Expected supply (μ_p)**: Startable players likely to be available later
- **Shortfall (S_p - μ_p)**: Expected deficit if you don't act
- **Shortfall cost (L_p)**: VBD penalty for missing out on startable players

## Implementation Strategy

### Phase 1: Dynamic Replacement Integration
Enhance existing `DynamicVBDTransformer` to use probabilistic replacement levels instead of static baselines.

### Phase 2: Roster Need Module
Add `PositionalNeedCalculator` class to compute PNI based on current roster state and selection probabilities.

### Phase 3: Unified Utility Function
Combine all factors into single utility score for draft recommendations.

## Real-World Examples

### Scenario 1: QB Run Starting
```
Josh Allen: VBD=15.2, P=0.95, R_dynamic=12.8, PNI=0.8 (have backup QB)
Utility = 0.95 × (15.2 - 12.8) × (1 + 0.3 × 0.8) = 2.28 × 1.24 = 2.83
```

### Scenario 2: Need RBs, Market Drying Up  
```
Saquon Barkley: VBD=18.5, P=0.7, R_dynamic=14.1, PNI=2.4 (need 2 RBs, few left)
Utility = 0.7 × (18.5 - 14.1) × (1 + 0.3 × 2.4) = 3.08 × 1.72 = 5.30
```

### Scenario 3: Deep Position, Already Stocked
```
Another WR: VBD=12.3, P=0.4, R_dynamic=11.8, PNI=0.2 (have 3 WRs, many available)
Utility = 0.4 × (12.3 - 11.8) × (1 + 0.3 × 0.2) = 0.2 × 1.06 = 0.21
```

## Benefits

### ✅ **Market-Aware**
Replacement levels adapt to actual draft flow, not theoretical math.

### ✅ **Roster-Intelligent**  
Accounts for your specific needs and team construction.

### ✅ **Probabilistically Sound**
Incorporates uncertainty and availability risk into valuations.

### ✅ **Unified Framework**
Single utility score combines value, scarcity, availability, and need.

### ✅ **Backward Compatible**
Can fallback to existing VBD if probability data unavailable.

## Technical Notes

- **β parameter**: Suggested starting value 0.3, tune based on draft results
- **Horizon**: 10-20 picks for replacement level calculation  
- **Probability threshold**: 0.3 (30%) for "likely to survive" classification
- **FLEX handling**: Distribute need proportionally across eligible positions
- **Cache optimization**: Pre-calculate PNI values to avoid real-time computation overhead

This framework transforms VBD from a static ranking system into a dynamic, context-aware draft assistant that adapts to both market conditions and your team's evolving needs.