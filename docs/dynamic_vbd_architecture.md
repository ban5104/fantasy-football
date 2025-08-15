# Dynamic VBD Architecture Plan

**Status**: Architecture Complete - Ready for Implementation  
**Estimated Implementation Time**: 45 minutes  
**Last Updated**: 2025-08-15

## Project Overview

This document outlines the architecture for implementing **Dynamic Value Based Drafting (VBD)** in our fantasy football analysis system. Dynamic VBD adjusts player rankings in real-time based on draft probability forecasts, providing a competitive advantage over static ranking systems.

### What is Dynamic VBD?

Traditional VBD uses fixed "replacement level" baselines (e.g., "14th QB is replacement level"). Dynamic VBD adjusts these baselines based on actual draft behavior:

- **Static VBD**: "Josh Allen has +88 VBD over 14th QB"
- **Dynamic VBD**: "QBs being ignored → 18th QB is now replacement → Josh Allen has +105 VBD"

## Problem Statement

### Current Issues with Static VBD
1. **High correlation between VBD methods** (r = 0.94-0.98) - should be more differentiated
2. **Doesn't account for actual draft dynamics** - real drafts don't follow theoretical patterns
3. **FLEX position ignored** in baseline calculations
4. **No adaptation to market behavior** - can't exploit draft runs or position scarcity

### Our Solution
Create a **probability-consuming transformation layer** that:
- Takes position draft probabilities from existing sophisticated models
- Transforms them into mathematically sound baseline adjustments
- Maintains all existing VBD calculation logic unchanged

## System Architecture

### High-Level Data Flow
```
1. Existing Probability System (espn_probability_matrix.ipynb)
   ↓ 
   Produces: {'RB': 0.60, 'WR': 0.25, 'QB': 0.15, 'TE': 0.05}
   
2. Dynamic VBD Transformer (NEW)
   ↓
   Applies: Sigmoid curve mapping + confidence weighting
   
3. Baseline Adjustment Calculator (NEW)
   ↓
   Computes: Continuous point interpolation (not rank shifts)
   
4. Enhanced VBD Calculator (MODIFIED)
   ↓
   Uses: Adjusted baselines → new VBD scores
   
5. Output: Real-time adjusted player rankings
```

### File Structure
```
src/
├── dynamic_vbd.py          # NEW - Main dynamic VBD engine
├── vbd.py                  # MODIFIED - Add baseline override helper (~10 lines)
├── statistical_analysis.py # EXISTING - Advanced VBD analysis
├── scoring.py              # EXISTING - Fantasy point calculations
└── scraping.py             # EXISTING - Data collection

config/
└── league-config.yaml      # MODIFIED - Add dynamic VBD parameters

tests/
└── test_dynamic_vbd.py     # NEW - Comprehensive test suite

docs/
└── dynamic_vbd_architecture.md  # THIS FILE
```

## Implementation Details

### Component 1: Dynamic VBD Engine (`src/dynamic_vbd.py`)

**Core Classes:**
```python
@dataclass
class DraftState:
    current_pick: int
    drafted_players: Set[str]
    remaining_picks: List[int]

@dataclass  
class ProbabilityForecast:
    horizon_picks: int                    # How many picks ahead
    position_probs: Dict[str, float]      # {'RB': 0.60, 'WR': 0.25, ...}
    position_confidence: Dict[str, float] # Uncertainty weighting (optional)

class DynamicVBDTransformer:
    def transform(self, df, probabilities, draft_state) -> pd.DataFrame
    def _compute_adjustments(self) -> Dict[str, float]
    def _apply_interpolation(self) -> pd.DataFrame
```

**Mathematical Core:**
```python
def compute_adjustment(expected_picks, adp_std, scale=3.0, kappa=5.0):
    """Transform expected picks into baseline point adjustments"""
    # Nonlinear mapping with diminishing returns
    raw_adjustment = scale * np.tanh(expected_picks / kappa)
    
    # Confidence weighting (high uncertainty = smaller adjustments)
    confidence = 1.0 / (1.0 + adp_std)
    
    # Final adjustment in "player equivalents"
    return raw_adjustment * confidence

def interpolate_baseline(pos_df, baseline_idx, adjustment):
    """Continuous point interpolation (not discrete rank shifts)"""
    if baseline_idx >= len(pos_df) - 1:
        return pos_df.iloc[-1]['FANTASY_PTS']
    
    # Get points at baseline rank and next rank
    pts_at_baseline = pos_df.iloc[baseline_idx]['FANTASY_PTS']
    pts_at_next = pos_df.iloc[baseline_idx + 1]['FANTASY_PTS']
    
    # Linear interpolation between ranks
    point_diff = pts_at_baseline - pts_at_next
    adjusted_points = pts_at_baseline - (adjustment * point_diff)
    
    return adjusted_points
```

### Component 2: VBD Integration Helper (`src/vbd.py`)

**Minimal Addition (~10 lines):**
```python
def calculate_all_vbd_methods(df, config, baseline_overrides=None):
    """Enhanced VBD calculation with optional baseline overrides
    
    Args:
        df: Player projections DataFrame
        config: League configuration
        baseline_overrides: Dict[position][method] → adjusted baseline points
    """
    # Get baseline calculations
    baselines = calculate_position_baselines(df, config)
    
    # Apply overrides if provided
    if baseline_overrides:
        baselines = _apply_baseline_overrides(baselines, baseline_overrides, df)
    
    # Continue with existing VBD calculation logic...
    return df_with_vbd
```

### Component 3: Configuration Schema

**Add to `config/league-config.yaml`:**
```yaml
dynamic_vbd:
  enabled: true
  params:
    scale: 3.0              # Max baseline adjustment in "player equivalents"
    kappa: 5.0              # Sigmoid steepness (lower = more aggressive)
    confidence_factor: 0.5  # Weight for uncertainty reduction
    cache_ttl: 300          # Cache lifetime in seconds
    max_adjustment: 2.0     # Cap adjustments at ±2 standard deviations
  methods:
    - BEER                  # Which VBD methods to adjust
    - VORP
    - VOLS
  monitoring:
    log_adjustments: true   # Log adjustment magnitudes
    log_cache_hits: true    # Monitor cache performance
```

### Component 4: Caching Strategy

**Deterministic Cache Keys:**
```python
class CacheKey:
    @staticmethod
    def generate(draft_state, probabilities, config_version):
        """Generate deterministic cache key for identical inputs"""
        state_str = json.dumps({
            'picks': sorted(draft_state.get('picked', [])),
            'prob_checksum': hashlib.md5(
                probabilities.to_json().encode()
            ).hexdigest(),
            'version': config_version
        }, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()
```

## Integration with Existing System

### Current Probability System
The existing `espn_probability_matrix.ipynb` notebook already provides:
- Position draft probabilities over time horizons
- Player availability percentages
- ESPN + ADP weighted models

### Integration Pattern
```python
# In your existing probability notebook:
from src.dynamic_vbd import DynamicVBDTransformer, DraftState, ProbabilityForecast

# After calculating position probabilities...
position_probs = compute_position_probabilities(available_df, picks_to_next)

# Create forecast object
forecast = ProbabilityForecast(
    horizon_picks=picks_to_next,
    position_probs=position_probs,
    position_confidence=compute_confidence_metrics()  # Optional
)

# Update VBD scores dynamically
transformer = DynamicVBDTransformer(config, base_projections)
dynamic_rankings = transformer.transform(
    available_df, 
    forecast, 
    current_draft_state
)

# Use dynamic_rankings instead of static VBD
```

## Testing Strategy

### Required Test Coverage

1. **Mathematical Accuracy**
   ```python
   def test_nonlinear_mapping():
       """Verify sigmoid behavior and diminishing returns"""
       # Test that adjustment grows sublinearly with expected_picks
       
   def test_confidence_weighting():
       """Verify uncertainty dampening"""
       # High ADP std should reduce adjustment magnitude
   ```

2. **Robustness**
   ```python
   def test_interpolation_continuity():
       """Ensure smooth baseline transitions"""
       # No discontinuities at integer boundaries
       
   def test_edge_cases():
       """Handle zero players, missing data, extreme values"""
   ```

3. **Performance**
   ```python
   def test_cache_determinism():
       """Verify cache key stability"""
       # Same inputs → same cache key
       
   def test_cache_performance():
       """Verify cache hit rates in realistic scenarios"""
   ```

4. **Historical Validation**
   ```python
   def test_historical_backtest():
       """Validate on 2024 draft data"""
       # Compare predicted vs actual draft outcomes
   ```

### Backtest Methodology
1. Load historical draft data (2023-2024 seasons)
2. Run dynamic VBD at each pick using historical probability forecasts
3. Measure:
   - How often pick recommendations changed vs static VBD
   - Whether changes improved roster outcomes (Monte Carlo simulation)
   - Correlation between adjustment magnitude and actual position runs

## Production Deployment

### Rollout Strategy
1. **Phase 1**: Deploy with `dynamic_vbd.enabled: false` (testing only)
2. **Phase 2**: A/B test different scale/kappa parameters
3. **Phase 3**: Enable for live drafts with monitoring
4. **Phase 4**: Tune parameters based on performance data

### Monitoring & Alerting
- **Adjustment magnitude distribution** - detect when baselines shift dramatically
- **Cache hit rates** - ensure performance doesn't degrade
- **Calculation time** - monitor real-time responsiveness
- **Error rates** - track fallback to static VBD

### Error Handling
- **Zero available players**: Return original VBD unchanged
- **Missing projections**: Use position average fallback
- **Invalid probabilities**: Log warning, use static VBD
- **Extreme adjustments**: Cap at ±2 standard deviations
- **Cache failures**: Recompute with performance logging

## Mathematical Justification

### Why Continuous Interpolation?
- **Smooth transitions**: No artificial discontinuities at rank boundaries
- **Respects point differentials**: Accounts for actual value gaps between players
- **Mathematically sound**: Linear interpolation preserves monotonicity

### Why Sigmoid Curve Mapping?
- **Diminishing returns**: High probabilities don't create unrealistic adjustments
- **Bounded output**: Natural ceiling prevents extreme baseline shifts
- **Tunable steepness**: Kappa parameter controls sensitivity

### Why Confidence Weighting?
- **Uncertainty principle**: Low consensus should reduce adjustment magnitude
- **Risk management**: Avoids overreacting to unreliable forecasts
- **Adaptive behavior**: More aggressive when consensus is high

## Success Metrics

### Technical Metrics
- ✅ **Implementation time**: Under 60 minutes
- ✅ **Test coverage**: >90% line coverage
- ✅ **Performance**: <100ms calculation time
- ✅ **Cache hit rate**: >80% in typical draft scenarios

### Business Metrics
- 📈 **Draft accuracy**: Improved prediction of when players get drafted
- 📈 **Roster quality**: Better final roster composition vs static VBD
- 📈 **Market inefficiency detection**: Identify and exploit position runs
- 📈 **User satisfaction**: More responsive and accurate draft recommendations

## What We're NOT Doing

To maintain minimal scope and focused implementation:

- ❌ **Not creating multiple configuration files**
- ❌ **Not building a complex plugin system**
- ❌ **Not implementing machine learning models**
- ❌ **Not creating new data pipelines**
- ❌ **Not modifying existing VBD calculation logic**
- ❌ **Not changing the probability prediction system**

## Next Steps for Implementation

1. **Create `src/dynamic_vbd.py`** with core mathematical functions
2. **Add baseline override helper** to `src/vbd.py`
3. **Update configuration schema** in `league-config.yaml`
4. **Write comprehensive tests** covering all mathematical edge cases
5. **Create integration notebook cell** for existing probability system
6. **Run backtest validation** on historical draft data
7. **Deploy with monitoring** and tune parameters

## Questions for Implementation

Before starting implementation, clarify:

1. **Column naming**: Does `base_projections` use `Player`, `POSITION`, `FANTASY_PTS`?
2. **Existing VBD interface**: Can we add optional parameter to `calculate_all_vbd_methods`?
3. **Configuration location**: Add to existing `league-config.yaml` or separate file?
4. **Logging framework**: What logging system is currently used?
5. **Test framework**: Is pytest the preferred testing framework?

---

**This document provides a complete architectural blueprint for implementing Dynamic VBD. The design prioritizes mathematical rigor, production robustness, and clean integration while maintaining minimal scope and implementation complexity.**