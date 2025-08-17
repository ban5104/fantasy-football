# Monte Carlo Draft Optimizer - Detailed Improvement Plan

## Executive Summary

The Monte Carlo draft optimizer has a **solid technical foundation** with proper variance reduction techniques (CRN, antithetic pairing) and efficient state management. However, critical **modeling and engineering gaps** affect real-world accuracy. This document provides a comprehensive action plan to address these issues.

## Critical Issues Identified

### 🚨 **URGENT Issues**

1. **Inconsistent Probability Systems**
   - Base system uses ESPN/ADP rankings with softmax (80/20 weighting)
   - Monte Carlo uses player "values" with different temperature scaling
   - **Impact**: Model decisions are based on inconsistent probability foundations

2. **State Drift Risk**
   - External picks may not properly synchronize with internal state
   - No validation of incoming pick events
   - **Impact**: Model's belief about available players can diverge from reality

### ⚠️ **HIGH Priority Issues**

3. **Weak Archetype Modeling**
   - Equal 0.25 distribution assumption for all archetypes
   - No smoothing for archetype updates
   - **Impact**: Unrealistic team behavior modeling

4. **Cache Inefficiency**
   - 0.92 cosine similarity threshold too restrictive
   - No age-weighting for cached values
   - **Impact**: Low cache hit rates, poor performance

5. **Missing Correlation Structure**
   - No position run modeling
   - Independent team decisions assumption
   - **Impact**: Misses real draft dynamics (reactive drafting, runs)

---

## Immediate Fixes (High-Impact, Low-Effort)

### 1. **Unify the Probability Pipeline (URGENT)**

**Problem**: Two different probability systems create model inconsistency.

**Solution**: Convert all sources into comparable scores before combining:
```
base_probs = w1 * softmax(transform(ESPN)) + w2 * softmax(transform(ADP)) + w3 * softmax(transform(proj))
```
Then apply roster/position penalties multiplicatively and renormalize.

**Result**: Monte Carlo rollouts and base pick model use identical probability foundation.

### 2. **State Validation & Synchronization (URGENT)**

**Implementation**:
- Validate incoming pick events: reject duplicates, warn on out-of-order picks
- Keep authoritative `draft_epoch` or sequence number
- Refuse to simulate until draft state and `draft_epoch` match
- Add reconciliation mechanism for state mismatches

### 3. **Smoothing for Archetype Posteriors (HIGH)**

**Current**: Simple replacement of archetype beliefs
**Fix**: Dirichlet/exponential smoothing
```
posterior ← (1-α) * posterior + α * likelihood_vector
```
Where α ≈ 0.15-0.25

**Benefit**: Prevents single early picks from over-determining team archetypes

### 4. **Improve Cache Efficiency (HIGH)**

**Changes**:
- Lower cosine tolerance to ~0.85 (from 0.92)
- Add age/visit_count tracking for cache entries
- Use age-weighted blending: `blended_value = λ*cached + (1-λ)*fresh`

### 5. **Add Simple Correlation Adjustments (MEDIUM)**

**Position Run Multiplier**:
- If last 2 picks were same position, boost that position's pick-probabilities by 1.25-1.5
- Decay over 2-4 subsequent picks
- Captures bulk of run/reactive behavior without heavyweight modeling

---

## Implementation Timeline (Realistic)

### Phase 1 (2-3 days): Core Fixes
- [ ] Unify probability pipeline across all components
- [ ] Add state validation and drift detection
- [ ] Run consistency tests between base and MC models
- [ ] Document and log all state changes

### Phase 2 (3-5 days): Smoothing & Optimization
- [ ] Implement Dirichlet smoothing for archetype updates (α ≈ 0.15-0.25)
- [ ] Lower cache tolerance to 0.85 and add age-weighting
- [ ] Test archetype learning behavior with contrived scenarios
- [ ] Measure cache hit rate improvements

### Phase 3 (1-2 weeks): Correlation & Validation
- [ ] Add position-run correlation modeling
- [ ] Full backtest suite on historical draft data
- [ ] Performance optimization and monitoring setup
- [ ] Calibrate all parameters based on historical data

**Total Realistic Timeline**: 2-4 weeks for complete implementation

---

## Validation Test Suite

Run each test and document results (pass/fail + metrics):

### 1. **Single-Team Need Test**
- **Setup**: Team A has 3 RBs already; compute P_t(QB) and P_t(RB) before/after
- **Expect**: P_t(QB) increases, P_t(RB) decreases
- **Metric**: Probability shift magnitude

### 2. **Archetype Posterior Shift Test**
- **Setup**: Uniform priors. Force Team B to pick WR three times
- **Expect**: Posterior mass shifts toward WR-focused archetype but retains smoothing
- **Metric**: Posterior distribution entropy

### 3. **Consistency Test (Unified Probabilities)**
- **Setup**: Compute top-10 pick probabilities via base model and MC average (same seeds)
- **Expect**: Distributions match within Monte Carlo noise
- **Metric**: Brier score / log-loss difference

### 4. **State-Drift Test**
- **Setup**: Inject out-of-order or duplicate pick events
- **Expect**: System rejects/reconciles and logs warnings
- **Metric**: Error detection rate

### 5. **Position-Run Sensitivity Test**
- **Setup**: Trigger two RB picks in a row artificially
- **Expect**: Next few picks show increased RB probabilities league-wide
- **Metric**: Correlation coefficient

### 6. **Backtest Calibration**
- **Metric**: Brier score for availability predictions vs historical drafts
- **Target**: Measurable improvement over ADP-only baseline

### 7. **Latency & Confidence Test**
- **Target**: Median latency < 2s, 95th percentile < 5s
- **Target**: >85% of decisions with SNR > 3 marked as CONFIDENT

---

## Medium-Term Improvements

### 1. **Estimate Archetype Mix from Data (MEDIUM)**
- Fit archetype priors and parameters using historical draft logs (MLE/EM)
- Replace equal 0.25 priors with data-derived distributions

### 2. **Model Team-to-Team Dependence (LONGER)**
- Add dependency model for position runs and reactive drafting
- Options: simple run-detection heuristic (fast) to conditional graphical model (heavy)

### 3. **Late-Round Top-K Expansion (MEDIUM)**
- Dynamically expand top-K after round X or when scarcity thresholds triggered
- Handle increased uncertainty in late rounds

---

## Recommended Parameter Starting Values

**Performance Parameters**:
- Cache cosine tolerance: **0.85** (from 0.92)
- Archetype smoothing α: **0.15-0.25**
- Position-run multiplier: **1.25** (decay over 2-4 picks)
- Initial CRN pairs: **50-200**, escalate to 500+ for close calls
- Top-K default: **40**, expand to 80 after round 10

**Tuning Notes**: All parameters should be calibrated against historical draft data

---

## Monitoring & Dashboards (Production Requirements)

### **Real-Time Monitoring**
- **Per-team roster table** (position counts) - live updates
- **Archetype posterior heatmap** - watch concentration evolution  
- **Delta pick-prob heatmap** (before vs after picks) - model reactivity
- **State drift alerts** - immediate notifications for pick mismatches

### **Performance Metrics**
- **Brier score tracking** (weekly) - prediction accuracy
- **EV-gap distribution** during live drafts - decision quality
- **Cache hit rate** - system efficiency
- **Latency percentiles** - user experience

---

## Risk Mitigation

### **Fallback Strategies**
1. **Probability System Failure**: Revert to simple ADP-based probabilities
2. **State Drift Detection**: Pause simulations, alert for manual reconciliation
3. **Performance Degradation**: Reduce simulation count, increase cache tolerance
4. **Archetype Learning Failure**: Reset to uniform priors

### **Data Quality Assurance**
- Validate all incoming ESPN/ADP data for completeness
- Cross-reference player IDs across all data sources
- Monitor for ranking volatility that might indicate data issues

---

## Success Metrics

### **Technical Metrics**
- Brier score improvement: Target 10-15% vs baseline
- Cache hit rate: Target >30% (vs current ~5%)
- Latency: 95th percentile < 2 seconds
- State synchronization: 99.9% pick validation success

### **User Experience Metrics**
- Decision confidence: >85% high-confidence decisions
- Availability prediction accuracy: <10% mean absolute error
- System uptime during drafts: 99.9%

---

## Conclusion

The Monte Carlo optimizer has strong technical foundations but requires systematic fixes to achieve production-level reliability. The prioritized approach focuses on:

1. **Immediate**: Fix probability inconsistency and state validation
2. **Short-term**: Improve archetype modeling and caching
3. **Medium-term**: Add correlation modeling and historical calibration

With these improvements, the system will provide significantly more accurate and reliable draft guidance while maintaining the sophisticated variance reduction techniques that give it an edge over simpler approaches.

**Next Steps**: Begin with Phase 1 implementation, focusing on probability unification as the highest-impact fix.