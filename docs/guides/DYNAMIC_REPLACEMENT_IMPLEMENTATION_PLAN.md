# Dynamic Replacement Levels Implementation Plan

## Problem Statement
The current system uses **static replacement levels** (e.g., 28th best RB) for VOR calculations but **dynamic opportunity costs** (what survives to next round). This inconsistency causes incorrect player valuations, particularly overvaluing QBs.

## Core Issue
- **Value calculation**: Uses static replacement (28th best RB available now)
- **Cost calculation**: Uses dynamic replacement (best RB surviving to next round)
- **Result**: Inconsistent VOR baselines leading to ~80% QB selection in Round 3

## Solution Overview
Implement dynamic replacement levels that:
1. Simulate what players survive to the next pick
2. Include FLEX position considerations
3. Cache results for performance
4. Use consistent methodology between value and cost

## Implementation Details

### 1. Add Dynamic Replacement Function (New)
**Location**: `src/monte_carlo/starter_optimizer.py` (after line 140)

```python
# Cache for dynamic replacement levels
_dynamic_replacement_cache = {}

def _pool_signature(pool_players, top_n=40):
    """Create signature from top N players for cache key."""
    top_ids = [p['id'] for p in sorted(pool_players, 
              key=lambda x: np.mean(x['samples']), reverse=True)[:top_n]]
    return tuple(top_ids)

def compute_dynamic_replacement_levels(pool_players, current_pick, replacement_pick, 
                                      sim: ExpectedBestSimulator, positions=None, 
                                      n_sims_inner=200, top_n_sig=40):
    """
    Compute replacement levels dynamically based on expected best player 
    at replacement_pick (typically next_my_pick).
    
    Args:
        pool_players: Available players
        current_pick: Current global pick number
        replacement_pick: Pick to evaluate replacement at (e.g., next_my_pick)
        sim: ExpectedBestSimulator instance
        positions: Positions to calculate (defaults to all starter positions)
        n_sims_inner: Simulations for expectation (200 for speed/accuracy balance)
        top_n_sig: Top N players for pool signature
    
    Returns:
        Dict of {position: replacement_value} including FLEX
    """
    if positions is None:
        positions = list(sim.league["starters_by_pos"].keys())
    
    # Create cache signature
    sig = (_pool_signature(pool_players, top_n=top_n_sig), 
           current_pick, replacement_pick, n_sims_inner)
    if sig in _dynamic_replacement_cache:
        return _dynamic_replacement_cache[sig]
    
    replacement = {}
    
    # Calculate replacement for each position
    for pos in positions:
        stats = sim.expected_best_for_position_at_pick(
            pos, current_pick - 1, replacement_pick, pool_players, n_sims=n_sims_inner)
        # Use p50 (median) for robustness against outliers
        replacement[pos] = stats.get("p50", stats.get("mean", 0.0))
    
    # FLEX = max(RB, WR, TE) at replacement pick
    flex_vals = []
    for p in ("RB", "WR", "TE"):
        if p in positions:
            stats = sim.expected_best_for_position_at_pick(
                p, current_pick - 1, replacement_pick, pool_players, n_sims=n_sims_inner)
            flex_vals.append(stats.get("p50", stats.get("mean", 0.0)))
    replacement["FLEX"] = max(flex_vals) if flex_vals else 0.0
    
    _dynamic_replacement_cache[sig] = replacement
    return replacement
```

### 2. Update marginal_starter_value Function
**Location**: `src/monte_carlo/starter_optimizer.py` (line 303)

**Change from static to dynamic replacement:**

```python
def marginal_starter_value(player, roster_state, starter_slots, all_players, 
                          current_pick, next_my_pick, sim, league=DEFAULT_LEAGUE):
    """
    Calculate marginal value using dynamic replacement levels.
    """
    pos = player["pos"]
    filled = roster_state.get(pos, [])
    need = starter_slots.get(pos, 0)
    
    # Use dynamic replacement levels
    replacement_levels = compute_dynamic_replacement_levels(
        all_players, current_pick, next_my_pick, sim, n_sims_inner=150)
    replacement_val = replacement_levels.get(pos, 0.0)
    
    # Calculate player's VOR
    player_value = float(np.mean(player["samples"]))
    player_vor = player_value - replacement_val
    
    if len(filled) < need:
        return player_vor
    
    # Compute worst starter by VOR
    current_starters = sorted(filled, key=lambda x: float(np.mean(x["samples"])), reverse=True)[:need]
    if not current_starters:
        return player_vor
    
    worst_starter_value = float(np.mean(current_starters[-1]["samples"]))
    worst_starter_vor = worst_starter_value - replacement_val
    
    return max(0.0, player_vor - worst_starter_vor)
```

### 3. Update pick_best_now with Two-Stage Evaluation
**Location**: `src/monte_carlo/starter_optimizer.py` (line 369)

**Add two-stage evaluation for performance:**

```python
def pick_best_now(pool_players, roster_state, current_pick, next_my_pick, league=DEFAULT_LEAGUE,
                  top_k_candidates=20, scenarios=500, pick_prob_fn=None, 
                  probability_model=None, rng=None):
    """
    Two-stage evaluation:
    1. Fast ranking pass (150-200 sims) for all candidates
    2. Detailed evaluation (500-600 sims) for top 3
    """
    rng = np.random.default_rng() if rng is None else rng
    sim = ExpectedBestSimulator(pool_players, league=league, pick_prob_fn=pick_prob_fn, 
                               probability_model=probability_model, rng=rng)
    
    # Stage 1: Fast ranking with 150 sims
    fast_replacement = compute_dynamic_replacement_levels(
        pool_players, current_pick, next_my_pick, sim, n_sims_inner=150)
    
    # Calculate VOR for sorting
    def calculate_vor(player):
        player_value = float(np.mean(player["samples"]))
        replacement_val = fast_replacement.get(player["pos"], 0.0)
        return player_value - replacement_val
    
    pool_sorted = sorted(pool_players, key=calculate_vor, reverse=True)
    candidates = pool_sorted[:max(top_k_candidates, 60)]
    
    # Fast evaluation of all candidates
    scores = []
    for cand in candidates[:top_k_candidates]:
        imm_val = marginal_starter_value(cand, roster_state, league["starters_by_pos"], 
                                        pool_players, current_pick, next_my_pick, sim, league)
        opp_cost = compute_opportunity_cost(cand, roster_state, current_pick, next_my_pick, 
                                           pool_players, sim, fast_replacement)
        score = imm_val - opp_cost
        scores.append((cand, score, imm_val, opp_cost))
    
    # Stage 2: Detailed evaluation of top 3
    scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = scores[:3]
    
    if top_candidates:
        # Clear cache for detailed pass
        _dynamic_replacement_cache.clear()
        
        # Detailed replacement levels with 500 sims
        detailed_replacement = compute_dynamic_replacement_levels(
            pool_players, current_pick, next_my_pick, sim, n_sims_inner=500)
        
        best_choice = None
        best_score = -1e9
        debug = []
        
        for cand, _, _, _ in top_candidates:
            imm_val = marginal_starter_value(cand, roster_state, league["starters_by_pos"], 
                                            pool_players, current_pick, next_my_pick, sim, league)
            opp_cost = compute_opportunity_cost(cand, roster_state, current_pick, next_my_pick,
                                               pool_players, sim, detailed_replacement)
            score = imm_val - opp_cost
            debug.append((cand["name"], cand["pos"], imm_val, opp_cost, score))
            
            if score > best_score:
                best_score = score
                best_choice = cand
    else:
        # Fallback if no candidates
        best_choice = candidates[0] if candidates else None
        best_score = 0.0
        debug = []
    
    return {
        "pick": best_choice,
        "score": best_score,
        "debug": sorted(debug, key=lambda x: x[4], reverse=True) if debug else []
    }
```

### 4. Update compute_opportunity_cost
**Location**: `src/monte_carlo/starter_optimizer.py` (line 335)

**Already accepts replacement_levels parameter, just ensure it uses dynamic:**

```python
def compute_opportunity_cost(player, roster_state, current_pick, next_my_pick, 
                            pool_players, sim: ExpectedBestSimulator, 
                            replacement_levels=None, positions_to_consider=None):
    """
    Opportunity cost using consistent dynamic replacement levels.
    Note: replacement_levels should be the same dynamic levels used for value calculation.
    """
    # Existing implementation already handles passed replacement_levels
    # Just ensure we pass dynamic levels from pick_best_now
```

## Key Implementation Notes

1. **Don't hardcode replacement_pick**: Use actual `next_my_pick` or calculate from draft position
2. **Use p50 over mean**: More robust to outlier simulations
3. **Cache with pool signature**: Avoid recalculating for same player pool
4. **Two-stage evaluation**: Balance speed and accuracy
5. **FLEX consideration**: Max of (RB, WR, TE) at replacement pick

## Testing Strategy

1. **Unit test dynamic replacement**:
   - Verify FLEX = max(RB, WR, TE)
   - Check cache effectiveness
   - Validate p50 usage

2. **Integration test**:
   - Run full simulation with dynamic replacement
   - Verify QB selection drops from 80% to reasonable levels
   - Check consistency between value and cost calculations

3. **Performance test**:
   - Measure impact of two-stage evaluation
   - Verify cache hit rates
   - Ensure <2 second decision time

## Expected Outcomes

- QB selection in Round 3 should drop from ~80% to ~20-30%
- RB/WR values should increase due to proper FLEX consideration
- Consistent VOR baseline between value and opportunity cost
- Decision time remains under 2 seconds with caching

## Migration Path

1. Add `compute_dynamic_replacement_levels` function
2. Update function signatures to pass required parameters
3. Implement two-stage evaluation in `pick_best_now`
4. Test with small simulations first
5. Validate against expected draft patterns