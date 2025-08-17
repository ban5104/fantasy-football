# Enhanced Dynamic VBD Design: Market Scarcity + Personal Need

## Problem Statement
Current Dynamic VBD only considers market scarcity (probability forecasts) but ignores personal roster construction needs. A player might be scarce in the market but not valuable to MY team if I'm already stocked at that position.

## Solution: Dual-Factor Scarcity Model

### 1. Market Scarcity (Current)
- Position probability forecasts
- Expected picks by position over horizon
- Reflects overall draft flow

### 2. Personal Need Factor (New)
```python
def calculate_personal_need_multiplier(my_roster: Dict[str, int], position: str, config: dict) -> float:
    """
    Calculate how much I need this position based on my current roster
    
    Returns:
    - 2.0+ = High need (missing starters)
    - 1.0 = Normal need (meeting minimum requirements) 
    - 0.3-0.8 = Low need (have surplus at position)
    """
    roster_slots = config['roster']['roster_slots']
    required = roster_slots.get(position, 1)
    current = my_roster.get(position, 0)
    
    if current == 0:
        return 2.5  # Desperate need - no players at position
    elif current < required:
        return 1.5 + (required - current) * 0.5  # Still need starters
    elif current == required:
        return 1.0  # Met minimum, normal priority
    else:
        surplus = current - required
        return max(0.3, 1.0 - (surplus * 0.2))  # Diminishing returns
```

### 3. Combined Urgency Calculation
```python
def calculate_combined_urgency(market_scarcity: float, personal_need: float) -> float:
    """
    Combine market and personal factors
    
    Examples:
    - High market scarcity + High personal need = Maximum urgency
    - High market scarcity + Low personal need = Moderate urgency  
    - Low market scarcity + High personal need = Moderate urgency
    - Low market scarcity + Low personal need = Minimum urgency
    """
    # Geometric mean provides balanced weighting
    return np.sqrt(market_scarcity * personal_need)
```

## Implementation Plan

### Phase 1: Data Structure Enhancement
```python
@dataclass 
class DraftState:
    """Enhanced draft state with personal roster tracking"""
    current_pick: int
    drafted_players: Set[str]
    my_roster: Dict[str, List[str]]  # NEW: {position: [player_names]}
    my_team_id: int                  # NEW: Which team am I?
```

### Phase 2: Dynamic VBD Enhancement
```python
class DynamicVBDTransformer:
    def _compute_adjustments(self, df: pd.DataFrame, 
                           probabilities: ProbabilityForecast,
                           draft_state: DraftState) -> Dict[str, Dict[str, float]]:
        """Enhanced with personal need integration"""
        
        for position in df['POSITION'].unique():
            # Current market scarcity calculation
            market_scarcity = position_prob * probabilities.horizon_picks
            
            # NEW: Personal need calculation  
            personal_need = self._calculate_personal_need(
                draft_state.my_roster, position
            )
            
            # NEW: Combined urgency
            combined_urgency = np.sqrt(market_scarcity * personal_need)
            
            # Apply to existing sigmoid
            adjustment = self.scale * np.tanh(combined_urgency / self.kappa)
```

### Phase 3: Configuration Integration
```yaml
# config/league-config.yaml
dynamic_vbd:
  enabled: true
  params:
    scale: 3.0
    kappa: 5.0
    personal_need:
      enabled: true
      desperate_multiplier: 2.5    # No players at position
      shortage_base: 1.5           # Missing starters
      surplus_decay: 0.2           # Diminishing returns per extra player
      minimum_need: 0.3            # Floor for overstocked positions
```

## Benefits of Integration

### ✅ **Unified Model**
- Single coherent VBD adjustment
- No need to mentally combine separate metrics
- Maintains existing Dynamic VBD infrastructure

### ✅ **Contextual Intelligence** 
- RB scarce in market but I have 3 RBs → Lower adjustment
- WR abundant in market but I have 0 WRs → Higher adjustment
- Accounts for FLEX considerations and roster construction

### ✅ **Real-time Adaptation**
- Updates automatically as I draft players
- Responds to other teams' picks affecting my needs
- Integrates seamlessly with existing caching

## Example Scenarios

### Scenario 1: RB Run Starting
```
Market: RB probability = 0.8 (high scarcity)
My Roster: 3 RBs already drafted
Personal Need: 0.4 (low - overstocked)
Combined: sqrt(0.8 * 0.4) = 0.57 (moderate adjustment)
Result: Still valuable but not urgent for me
```

### Scenario 2: WR Drought, I Need WRs
```
Market: WR probability = 0.3 (low scarcity)  
My Roster: 1 WR, need 3 starters
Personal Need: 2.0 (high - missing starters)
Combined: sqrt(0.3 * 2.0) = 0.77 (higher adjustment)
Result: Market thinks WRs available, but I need them NOW
```

### Scenario 3: Perfect Storm
```
Market: TE probability = 0.9 (very high scarcity)
My Roster: 0 TEs 
Personal Need: 2.5 (desperate)
Combined: sqrt(0.9 * 2.5) = 1.5 (maximum urgency)
Result: Must draft TE immediately
```

## Technical Implementation

### New Methods to Add:
1. `_calculate_personal_need()` - Core need calculation
2. `_get_my_roster_count()` - Extract position counts from draft state
3. `_combine_urgency_factors()` - Market + personal integration
4. Enhanced `DraftState` with roster tracking

### Backward Compatibility:
- If `my_roster` not provided, defaults to `personal_need = 1.0` (current behavior)
- Existing probability forecasts continue to work
- Configuration remains optional

This approach gives you sophisticated roster-aware scarcity while maintaining the elegance of your existing Dynamic VBD system.