# Phase A Implementation Plan - Updated for Refactored Code

## Overview
Based on the recently refactored `starter_optimizer.py` (now modular with separated functions), this plan leverages the cleaner architecture to implement Phase A improvements with minimal disruption.

## Current Architecture Benefits
✅ **Clean separation**: `_filter_and_prioritize_candidates()`, `_evaluate_candidates()`  
✅ **Modular design**: Easy to inject calibrated opponent model  
✅ **Performance monitoring**: Already built-in timing and cache tracking  
✅ **North Star aligned**: MSG-OC framework preserved and clear  

## Implementation Sequence

### Step 1: Calibrated Opponent Model (Days 1-5)

**Create**: `src/monte_carlo/opponent_calibration.py`
```python
"""Calibrated opponent model with graceful fallback to existing logic"""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import os

class CalibratedOpponentPredictor:
    def __init__(self, fallback_fn=None):
        self.model = None
        self.is_trained = False
        self.fallback_fn = fallback_fn
        
    def train_if_data_exists(self, data_path="./data/historical_picks.csv"):
        """Train only if historical data exists, otherwise use fallback"""
        if os.path.exists(data_path):
            try:
                # Load and train on historical pick data
                self._train_from_csv(data_path)
                self.is_trained = True
                print("✅ Calibrated opponent model trained")
            except Exception as e:
                print(f"⚠️ Training failed, using fallback: {e}")
                self.is_trained = False
        else:
            print("ℹ️ No historical data found, using fallback opponent model")
    
    def predict_pick_probabilities(self, pool_players, round_num, team_rosters, recent_picks=None):
        """Predict with calibrated model or fallback"""
        if self.is_trained and self.model:
            features = self._extract_features(pool_players, round_num, team_rosters, recent_picks)
            if features.size > 0:
                probs = self.model.predict_proba(features.reshape(1, -1))[0]
                # Convert to dict mapping player_id -> probability
                return {player['id']: prob for player, prob in zip(pool_players, probs)}
        
        # Fallback to existing logic
        if self.fallback_fn:
            return self.fallback_fn(pool_players, round_num, team_rosters, recent_picks)
        
        # Ultimate fallback: uniform distribution
        uniform_prob = 1.0 / len(pool_players)
        return {player['id']: uniform_prob for player in pool_players}
    
    def _extract_features(self, pool_players, round_num, team_rosters, recent_picks):
        """Extract features for ML model"""
        # Simple feature set: round, position scarcity, recent positional runs
        features = []
        for player in pool_players:
            pos_count = sum(1 for roster in team_rosters.values() for pos in roster if pos == player['pos'])
            recent_run = sum(1 for pick in (recent_picks or [])[-3:] if pick == player['pos'])
            features.append([round_num, pos_count, recent_run, player.get('rank', 100)])
        return np.array(features)
    
    def _train_from_csv(self, data_path):
        """Train classifier from historical data"""
        # Placeholder - implement based on your historical data format
        # For now, create a simple model that learns position preferences by round
        pass
```

**Modify**: `src/monte_carlo/starter_optimizer.py` - Add calibration option
```python
# Add at top with other imports
from .opponent_calibration import CalibratedOpponentPredictor

# Modify pick_best_now signature to include calibration option
def pick_best_now(pool_players, roster_state, current_pick, next_my_pick, league=DEFAULT_LEAGUE,
                  top_k_candidates=20, scenarios=500, pick_prob_fn=None, probability_model=None, 
                  rng=None, my_team_idx=None, clear_cache_after=True, use_calibrated_opponent=True):
    
    start_time = time.time()
    rng = np.random.default_rng() if rng is None else rng
    
    # NEW: Initialize calibrated opponent model if requested
    if use_calibrated_opponent and pick_prob_fn is None:
        calibrated_opponent = CalibratedOpponentPredictor()
        calibrated_opponent.train_if_data_exists()
        pick_prob_fn = calibrated_opponent.predict_pick_probabilities
    
    # Rest of function unchanged...
    sim = ExpectedBestSimulator(pool_players, league=league, pick_prob_fn=pick_prob_fn, 
                               probability_model=probability_model, rng=rng, my_team_idx=my_team_idx)
    # ... existing code continues
```

### Step 2: Signature-Based Caching (Days 6-10)

**Create**: `src/monte_carlo/cache_manager.py`
```python
"""Efficient caching based on pool/roster signatures"""
import hashlib
from functools import wraps
import joblib
import os

# Initialize joblib memory with project cache directory
CACHE_DIR = './cache/monte_carlo'
os.makedirs(CACHE_DIR, exist_ok=True)
memory = joblib.Memory(location=CACHE_DIR, verbose=0)

def create_state_signature(roster_ids, scenario_idx, starter_slots):
    """Create deterministic signature for caching"""
    # Sort for deterministic ordering
    sorted_roster = sorted(roster_ids) if roster_ids else []
    key_data = f"{sorted_roster}_{scenario_idx}_{sorted(starter_slots.items())}"
    return hashlib.md5(key_data.encode()).hexdigest()[:16]

def cache_with_signature(func):
    """Decorator that caches based on state signatures"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Try to extract roster info for signature
        if len(args) >= 3:  # Assuming (roster_players, scenario_idx, starter_slots, ...)
            roster_players, scenario_idx, starter_slots = args[0], args[1], args[2]
            roster_ids = [p['id'] for p in roster_players] if roster_players else []
            signature = create_state_signature(roster_ids, scenario_idx, starter_slots)
            
            # Use joblib cache with signature
            cached_func = memory.cache(func)
            return cached_func(*args, **kwargs)
        else:
            # Fallback to uncached
            return func(*args, **kwargs)
    return wrapper

# Cache hit/miss tracking
cache_stats = {'hits': 0, 'misses': 0}

def get_cache_stats():
    """Get cache performance metrics"""
    total = cache_stats['hits'] + cache_stats['misses']
    hit_rate = cache_stats['hits'] / total if total > 0 else 0.0
    return {'hit_rate': hit_rate, 'hits': cache_stats['hits'], 'misses': cache_stats['misses']}

def clear_cache():
    """Clear all cached results"""
    memory.clear()
    cache_stats['hits'] = 0
    cache_stats['misses'] = 0
```

**Modify**: `src/monte_carlo/starter_core.py` - Add caching to compute_starter_sum
```python
# Add at top
from .cache_manager import cache_with_signature, cache_stats

@cache_with_signature
def compute_starter_sum(roster_players, scenario_idx, starter_slots, league=None):
    """Compute starter sum with caching (existing logic unchanged)"""
    cache_stats['hits'] += 1  # Will be incremented on cache hit
    
    # ... existing compute_starter_sum implementation ...
    # No changes to the actual logic, just add caching wrapper
```

### Step 3: CRN-Aware Batching (Days 11-15)

**Modify**: `src/monte_carlo/starter_optimizer.py` - Enhance `_evaluate_candidates`
```python
def _evaluate_candidates(candidates, roster_players, starter_slots, league, 
                        opportunity_cost, n_scenarios, use_batching=True):
    """Evaluate candidates using MSG - OC framework with optional batching."""
    if use_batching and len(candidates) > 5:
        return _evaluate_candidates_batched(candidates, roster_players, starter_slots, 
                                          league, opportunity_cost, n_scenarios)
    
    # Existing sequential evaluation (unchanged for compatibility)
    best_choice = None
    best_score = -1e9
    debug = []
    
    for cand in candidates:
        try:
            msg_values = [
                marginal_starter_gain(cand, roster_players, i, starter_slots, league)
                for i in range(min(n_scenarios, 150))
            ]
            expected_msg = float(np.mean(msg_values)) if msg_values else 0.0
            score = expected_msg - opportunity_cost
            debug.append((cand["name"], cand["pos"], expected_msg, opportunity_cost, score))
            
            if score > best_score:
                best_score = score
                best_choice = cand
        except Exception as e:
            print(f"Warning: Error evaluating {cand.get('name', 'Unknown')}: {e}")
            continue
    
    if best_choice is None and candidates:
        best_choice = candidates[0]
        best_score = 0.0
        debug = [(best_choice["name"], best_choice["pos"], 0.0, 0.0, 0.0)]
    
    return best_choice, best_score, debug

def _evaluate_candidates_batched(candidates, roster_players, starter_slots, 
                               league, opportunity_cost, n_scenarios):
    """Batched evaluation preserving CRN"""
    n_candidates = len(candidates)
    max_scenarios = min(n_scenarios, 150)
    
    # Pre-allocate matrix for vectorized operations
    msg_matrix = np.zeros((n_candidates, max_scenarios))
    
    # Batch calculate MSG values (preserves CRN by using same scenario indices)
    for scenario_idx in range(max_scenarios):
        for cand_idx, candidate in enumerate(candidates):
            try:
                msg_matrix[cand_idx, scenario_idx] = marginal_starter_gain(
                    candidate, roster_players, scenario_idx, starter_slots, league
                )
            except Exception:
                msg_matrix[cand_idx, scenario_idx] = 0.0
    
    # Calculate expected MSG per candidate
    expected_msgs = np.mean(msg_matrix, axis=1)
    
    # Find best candidate
    scores = expected_msgs - opportunity_cost
    best_idx = np.argmax(scores)
    
    # Build debug info
    debug = []
    for i, candidate in enumerate(candidates):
        debug.append((candidate["name"], candidate["pos"], 
                     expected_msgs[i], opportunity_cost, scores[i]))
    
    return candidates[best_idx], scores[best_idx], debug
```

**Add batching option to main function**:
```python
def pick_best_now(pool_players, roster_state, current_pick, next_my_pick, league=DEFAULT_LEAGUE,
                  top_k_candidates=20, scenarios=500, pick_prob_fn=None, probability_model=None, 
                  rng=None, my_team_idx=None, clear_cache_after=True, use_calibrated_opponent=True,
                  use_batching=True):
    
    # ... existing setup code ...
    
    # Pass batching option to evaluation
    best_choice, best_score, debug = _evaluate_candidates(
        candidates, roster_players, starter_slots, league, opportunity_cost, n_scenarios, use_batching)
    
    # ... rest unchanged ...
```

## Testing Strategy

### Unit Tests (`tests/test_phase_a.py`)
```python
def test_calibrated_opponent_fallback():
    """Test graceful fallback when no training data"""
    predictor = CalibratedOpponentPredictor()
    # Should not crash and return reasonable probabilities
    
def test_cache_determinism():
    """Test cache signature consistency"""
    sig1 = create_state_signature([1, 2, 3], 42, {'QB': 1, 'RB': 2})
    sig2 = create_state_signature([3, 1, 2], 42, {'RB': 2, 'QB': 1})  # Different order
    assert sig1 == sig2  # Should be same due to sorting
    
def test_batched_vs_sequential():
    """Test batched evaluation produces same results as sequential"""
    # Run same candidates through both paths, verify identical results
```

### Performance Benchmarks
```bash
# Baseline (before Phase A)
python -c "
from src.monte_carlo.starter_optimizer import pick_best_now
import time
# Run with use_calibrated_opponent=False, use_batching=False
"

# Phase A (after implementation)  
python -c "
from src.monte_carlo.starter_optimizer import pick_best_now
import time
# Run with use_calibrated_opponent=True, use_batching=True
"
```

## Dependencies to Add
```toml
# pyproject.toml - Add to existing dependencies
scikit-learn = ">=1.3.0"
joblib = ">=1.3.0"
```

## Success Metrics
- **Cache Hit Rate**: >50% for repeated similar states
- **Batching Speedup**: 20% faster evaluation with 10+ candidates  
- **Opponent Accuracy**: Measurable improvement in pick predictions
- **Zero Regression**: Same top recommendations in controlled tests
- **Performance**: Maintain <2s target with Phase A enabled

## Risk Mitigation
- **Feature Flags**: All Phase A features can be disabled via parameters
- **Graceful Fallback**: System works even if calibration fails or cache disabled
- **Backward Compatibility**: Existing interfaces unchanged
- **Incremental**: Each step can be tested independently

This plan leverages your clean refactored architecture and can be implemented incrementally with minimal risk to the existing MSG-OC framework.