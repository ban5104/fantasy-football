# Open Source Library Integration Plan for Monte Carlo Draft Optimizer

## Executive Summary

This document outlines a phased approach to integrate open source libraries into the existing Monte Carlo draft optimization system while preserving the core MSG-OC (Marginal Starter Gain - Opportunity Cost) decision framework and Common Random Numbers (CRN) variance reduction advantages.

**Core Principle**: Use OSS for infrastructure ("bricks"), keep sequential stochastic decision logic custom ("brain").

## Current System Strengths to Preserve

1. **MSG-OC Decision Framework**: Sequential stochastic optimization with recourse
2. **CRN Variance Reduction**: 40-60% improvement in estimation accuracy
3. **Pool/Roster Signatures**: Efficient state representation for caching
4. **Performance**: 4x parallel speedup, sub-2s decision times

## Phase A: Sharpen the Core (Immediate Value)

### 1. Calibrated Opponent Model

**Problem**: Current opponent prediction uses basic rank-weighted softmax
**Solution**: Train calibrated classifier on historical pick data

**Files to Modify**:
- `src/monte_carlo/opponent.py` - Add ML-based prediction
- `src/monte_carlo/probability_models.py` - Integration point
- New: `src/monte_carlo/opponent_calibration.py` - Training module

**Dependencies to Add**:
```toml
scikit-learn = ">=1.3.0"
```

**Implementation**:
```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

class CalibratedOpponentModel:
    def __init__(self):
        base_classifier = GradientBoostingClassifier(n_estimators=100)
        self.calibrated_model = CalibratedClassifierCV(base_classifier, method='isotonic')
    
    def fit(self, historical_data):
        # Features: round, team_needs, positional_runs, ADP_deltas, recent_picks
        features = self.extract_features(historical_data)
        targets = historical_data['actual_picks']
        self.calibrated_model.fit(features, targets)
    
    def predict_pick_probabilities(self, current_state):
        features = self.extract_features([current_state])
        return self.calibrated_model.predict_proba(features)[0]
```

**Integration**: Plug into existing `pick_prob_fn` parameter in `starter_optimizer.py`

### 2. Signature-Based Caching

**Problem**: Redundant SS calculations across similar states
**Solution**: Persistent caching keyed by pool/roster signatures

**Files to Modify**:
- `src/monte_carlo/starter_core.py` - Add caching decorators
- `src/monte_carlo/starter_optimizer.py` - Cache SS(now)/SS(next)
- New: `src/monte_carlo/cache_manager.py` - Signature-based cache

**Dependencies to Add**:
```toml
joblib = ">=1.3.0"
```

**Implementation**:
```python
from joblib import Memory
import hashlib

memory = Memory(location='./cache/monte_carlo', verbose=0)

def generate_pool_signature(available_players):
    """Create deterministic signature from available player pool"""
    player_ids = sorted(available_players)
    return hashlib.md5(str(player_ids).encode()).hexdigest()[:16]

def generate_roster_signature(roster_state):
    """Create deterministic signature from current roster"""
    sorted_roster = sorted([p['id'] for pos_list in roster_state.values() for p in pos_list])
    return hashlib.md5(str(sorted_roster).encode()).hexdigest()[:16]

@memory.cache
def cached_starter_sum(pool_sig, roster_sig, draw_seed, starter_slots):
    """Cache SS calculations by state signature"""
    return compute_starter_sum(pool_sig, roster_sig, draw_seed, starter_slots)
```

### 3. CRN-Aware Vectorized Batching

**Problem**: Per-draw calculations not fully vectorized
**Solution**: Batch SS calculations while preserving CRN

**Files to Modify**:
- `src/monte_carlo/starter_core.py` - Vectorize SS computations
- `src/monte_carlo/crn_manager.py` - Batch-aware sampling
- `src/monte_carlo/starter_optimizer.py` - Use vectorized calls

**Implementation**:
```python
def vectorized_starter_sums(candidates, roster_players, draw_indices, starter_slots):
    """Compute SS for multiple candidates/draws efficiently"""
    # Pre-allocate arrays for vectorized operations
    n_candidates = len(candidates)
    n_draws = len(draw_indices)
    
    # Vectorized projection sampling using existing CRN
    all_projections = np.zeros((n_candidates, n_draws))
    for i, candidate in enumerate(candidates):
        for j, draw_idx in enumerate(draw_indices):
            all_projections[i, j] = candidate['samples'][draw_idx]
    
    # Vectorized starter sum calculations
    return compute_starter_sums_batch(all_projections, roster_players, starter_slots)
```

## Phase B: Insight UX (Medium Value)

### 4. MSG-OC Telemetry and Visualization

**Problem**: Decision rationale not transparent to users
**Solution**: Rich telemetry showing survival curves, starter inclusion, MSG/OC breakdown

**Files to Modify**:
- `src/monte_carlo/starter_optimizer.py` - Enhanced return structure
- New: `src/monte_carlo/telemetry.py` - Visualization module
- New: `notebooks/draft_decision_analysis.ipynb` - Interactive analysis

**Dependencies to Add**:
```toml
arviz = ">=0.15.0"
plotly = ">=5.15.0"  # Already in pyproject.toml
```

**Implementation**:
```python
import arviz as az
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_decision_telemetry(msg_draws, oc_estimate, survival_probs, candidate):
    """Generate comprehensive decision analysis"""
    
    # Survival probability curve
    fig = make_subplots(rows=2, cols=2, 
                       subplot_titles=['Survival to Next Pick', 'MSG Distribution', 
                                     'MSG vs OC Breakdown', 'Starter Inclusion %'])
    
    # Plot 1: Survival curve
    fig.add_trace(go.Scatter(x=list(range(len(survival_probs))), 
                            y=survival_probs, name='Survival Probability'),
                 row=1, col=1)
    
    # Plot 2: MSG distribution 
    fig.add_trace(go.Histogram(x=msg_draws, name='MSG Distribution'),
                 row=1, col=2)
    
    # Plot 3: MSG vs OC
    fig.add_trace(go.Bar(x=['Expected MSG', 'Opportunity Cost', 'Net Score'],
                        y=[np.mean(msg_draws), oc_estimate, 
                           np.mean(msg_draws) - oc_estimate]),
                 row=2, col=1)
    
    return fig

def explain_decision(best_candidate, decision_metrics):
    """Generate human-readable decision rationale"""
    rationale = f"""
    RECOMMENDATION: {best_candidate['name']} ({best_candidate['pos']})
    
    KEY METRICS:
    • Expected MSG: {decision_metrics['expected_msg']:.1f} points
    • Opportunity Cost: {decision_metrics['opportunity_cost']:.1f} points  
    • Net Score: {decision_metrics['decision_score']:.1f} points
    • Survival to Next Pick: {decision_metrics['survival_prob']:.1%}
    • Starter Inclusion Rate: {decision_metrics['starter_inclusion']:.1%}
    
    RATIONALE: This player provides {decision_metrics['expected_msg']:.1f} points of 
    immediate starter value improvement, which exceeds the {decision_metrics['opportunity_cost']:.1f} 
    point cost of waiting for the next pick by {decision_metrics['decision_score']:.1f} points.
    """
    return rationale
```

### 5. Data-Driven Distribution Calibration

**Problem**: Static ±20% projection envelopes not realistic
**Solution**: Fit position-specific distributions from historical residuals

**Files to Modify**:
- `src/monte_carlo/probability.py` - Enhanced envelope generation
- New: `src/monte_carlo/distribution_fitting.py` - Historical calibration
- `data/` - Add historical projection residuals dataset

**Dependencies to Add**:
```toml
# PyMC already implied from scipy usage
```

**Implementation**:
```python
from scipy import stats
import pandas as pd

def fit_position_distributions(historical_residuals_df):
    """Fit position-specific projection error distributions"""
    position_params = {}
    
    for position in ['QB', 'RB', 'WR', 'TE']:
        pos_residuals = historical_residuals_df[
            historical_residuals_df['position'] == position
        ]['residual'].values
        
        # Fit multiple distributions and select best by AIC
        distributions = [stats.norm, stats.lognorm, stats.t]
        best_dist = None
        best_aic = np.inf
        
        for dist in distributions:
            params = dist.fit(pos_residuals)
            aic = 2 * len(params) - 2 * np.sum(dist.logpdf(pos_residuals, *params))
            if aic < best_aic:
                best_aic = aic
                best_dist = (dist, params)
        
        position_params[position] = best_dist
    
    return position_params

def generate_calibrated_envelope(base_projection, position, position_params):
    """Generate realistic projection envelope from fitted distributions"""
    dist, params = position_params[position]
    
    # Generate percentiles for envelope
    residual_samples = dist.rvs(*params, size=1000)
    low = base_projection + np.percentile(residual_samples, 10)
    high = base_projection + np.percentile(residual_samples, 90)
    
    return {'low': low, 'base': base_projection, 'high': high}
```

## Phase C: Scale When Needed (Future Value)

### 6. Ray-Based Parallelism with CRN Preservation

**Problem**: ProcessPoolExecutor limited to single machine
**Solution**: Ray distributed computing with deterministic seeding

**Files to Modify**:
- `src/monte_carlo/simulator.py` - Ray integration option
- `src/monte_carlo/crn_manager.py` - Distributed CRN support

**Dependencies to Add**:
```toml
ray = ">=2.5.0"  # Optional dependency
```

**Implementation**:
```python
import ray

@ray.remote
def distributed_candidate_scoring(candidates_batch, roster_state, pool_state, 
                                base_seed, draw_indices):
    """Score candidate batch with preserved CRN"""
    # Deterministic seeding: base_seed ⊕ batch_id ⊕ pool_signature
    batch_seed = base_seed ^ hash(str(pool_state)) ^ hash(str(draw_indices))
    rng = np.random.default_rng(batch_seed)
    
    # Process batch with local CRN
    scores = []
    for candidate in candidates_batch:
        msg_values = []
        for draw_idx in draw_indices:
            # Use deterministic seed per draw
            draw_seed = batch_seed ^ draw_idx
            msg = marginal_starter_gain(candidate, roster_state, draw_seed, starter_slots)
            msg_values.append(msg)
        scores.append(np.mean(msg_values))
    
    return scores

def scale_with_ray(candidates, roster_state, pool_state, n_scenarios):
    """Scale candidate evaluation across Ray cluster"""
    if not ray.is_initialized():
        ray.init()
    
    # Batch candidates for distribution
    batch_size = max(1, len(candidates) // ray.available_resources()['CPU'])
    batches = [candidates[i:i+batch_size] for i in range(0, len(candidates), batch_size)]
    
    # Distribute scoring
    futures = []
    for batch in batches:
        future = distributed_candidate_scoring.remote(
            batch, roster_state, pool_state, 42, range(n_scenarios)
        )
        futures.append(future)
    
    # Collect results
    all_scores = ray.get(futures)
    return np.concatenate(all_scores)
```

## Implementation Timeline

### Week 1-2: Phase A.1 - Calibrated Opponent Model
- Create `opponent_calibration.py` module
- Integrate with existing `pick_prob_fn` interface
- Test with historical data

### Week 3-4: Phase A.2 - Signature-Based Caching  
- Implement cache manager with pool/roster signatures
- Add caching decorators to SS calculations
- Performance testing and validation

### Week 5-6: Phase A.3 - CRN-Aware Batching
- Vectorize starter sum calculations
- Preserve CRN across batch operations
- Integration testing with existing simulator

### Week 7-8: Phase B.1 - MSG-OC Telemetry
- Create visualization module
- Enhance optimizer return structure
- Build interactive analysis notebook

## Success Metrics

### Phase A Targets:
- **Opponent Model**: >5% improvement in pick prediction accuracy
- **Caching**: >50% reduction in redundant SS calculations  
- **Vectorization**: >20% speedup in candidate scoring

### Phase B Targets:
- **Telemetry**: Decision rationale available in <0.5s
- **Calibration**: Projection accuracy improvement measurable via backtesting

### Phase C Targets:
- **Ray Scaling**: Linear speedup beyond 4 cores while preserving CRN variance reduction

## Risk Mitigation

1. **Preserve Existing Interfaces**: All changes maintain backward compatibility
2. **Incremental Testing**: Each phase includes comprehensive validation
3. **Performance Monitoring**: Continuous benchmarking against current system
4. **Rollback Strategy**: Feature flags allow disabling new components

## Dependencies Summary

```toml
# Phase A
scikit-learn = ">=1.3.0"
joblib = ">=1.3.0"

# Phase B  
arviz = ">=0.15.0"
plotly = ">=5.15.0"  # Already present

# Phase C (Optional)
ray = ">=2.5.0"
```

All dependencies are mature, well-maintained libraries with strong ecosystem support.