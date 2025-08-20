# 14-Round Monte Carlo Draft Strategy Enhancement Plan

## Executive Summary
Evolve the current 7-round Monte Carlo simulation to a comprehensive 14-round framework that naturally discovers optimal draft strategies balancing starting lineup maximization with roster depth value.

## Core Insight
**The Problem**: Current 7-round optimization maximizes starting lineup points but ignores the strategic value of bench depth. Taking a TE in round 7 might add 15 points to starters but costs multiple RB/WR depth pieces that provide injury insurance, bye week coverage, and upside.

**The Solution**: Extend to 14 rounds (excluding K/DST rounds) to capture the full trade-off between starter optimization and roster construction value.

## Key Concepts

### 1. Depth Quality Thresholds
Define backup quality based on projection gap from starters:
- **Excellent Backup** (≤15% gap): Season-ready, minimal downgrade
- **Useful Backup** (15-25% gap): Acceptable fill-in, some downgrade risk  
- **Depth Only** (>25% gap): Not worth roster spot normally

### 2. Natural Strategy Discovery
Instead of predetermined strategies, let patterns emerge naturally from simulations:
- Run simulations with variation in pick selection
- Detect what strategy patterns naturally occurred
- Group and analyze emergent strategies
- No forced bias, pure empirical discovery

### 3. Dual Value System
Evaluate rosters on two dimensions:
```
Total Value = Starter Points + (Bench Quality × Injury/Bye Factor)
```
- **Starter Points**: Optimal starting lineup projection
- **Bench Quality**: Count of Excellent + Useful backups
- **Trade-off Analysis**: Quantify starter points sacrificed vs depth gained

## Implementation Framework

### Phase 1: Extend Simulation Depth
- Change from 7 rounds to 14 rounds
- Rounds 15-16 remain reserved for K/DST
- Remove separate 7-round framework entirely

### Phase 2: Enhanced Roster Evaluation
```python
def evaluate_roster(roster):
    starter_points = calculate_optimal_lineup(roster)
    
    bench_metrics = {
        'excellent_backups': count_within_15_percent(roster),
        'useful_backups': count_15_to_25_percent(roster),
        'depth_only': count_over_25_percent(roster),
        'injury_resilience': calculate_replacement_value(roster),
        'flex_options': count_flex_worthy_bench(roster)
    }
    
    total_value = starter_points + (bench_quality * injury_factor)
    return starter_points, bench_metrics, total_value
```

### Phase 3: Pattern Detection
```python
def detect_draft_pattern(roster):
    # Analyze when positions were drafted
    te_round = get_round_drafted(roster, 'TE')
    qb_round = get_round_drafted(roster, 'QB')
    rb_wr_rounds_6_10 = count_rb_wr_in_rounds(roster, 6, 10)
    
    # Natural pattern classification
    if te_round >= 10: pattern = "punt_te"
    elif rb_wr_rounds_6_10 >= 4: pattern = "depth_loading"
    # ... etc
    
    return pattern
```

### Phase 4: Strategy Comparison Output
```
Emergent Strategy Analysis (14 rounds):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Top Naturally Occurring Patterns:
────────────────────────────────────────────────────
Pattern         | Starter | Bench Quality  | Total | Freq
               | Points  | Exc | Useful   | Value |
───────────────┼─────────┼─────┼──────────┼───────┼──────
Fill Starters  | 1,425   |  0  |    2     | 1,451 | 18%
Depth First    | 1,380   |  3  |    4     | 1,467 | 14%  ← Better total value
Balanced       | 1,405   |  2  |    3     | 1,460 | 12%

Trade-off Analysis:
• Depth First: -45 starter pts for +5 quality backups
• Worth it? 5 backups × 12.5 value = 62.5 pts > 45 pts lost ✓
```

## Critical Requirements

### 1. Strategy-Agnostic Simulation
- Don't predefine strategies
- Let patterns emerge naturally
- Discover rather than prescribe

### 2. Comprehensive Metrics
- Starting lineup points (current focus)
- Bench depth quality (new dimension)
- Total roster value (combined metric)
- Position-specific depth analysis

### 3. Clear Trade-off Visualization
Show exactly what you gain/lose:
- "Strategy A: -45 starter points, +3 excellent backups"
- "Strategy B: -20 starter points, +2 useful backups"
- Enable informed decision-making

## Expected Outcomes

### What This Reveals
1. **True cost of position reaching**: Quantify the depth sacrificed for marginal starter gains
2. **Optimal position timing**: When to draft each position for maximum total value
3. **Natural strategy emergence**: Which patterns consistently perform well
4. **Risk/reward profiles**: High floor vs high ceiling roster construction

### Decision Framework
Users can objectively evaluate:
> "I'm willing to sacrifice 45 starter points to gain 3 excellent backups because injury resilience is worth more than marginal TE upgrade"

## Technical Considerations

### Performance
- 14 rounds × 14 teams = 196 picks to simulate (vs 98 in 7-round)
- May need to optimize simulation count or use sampling
- Consider caching repeated scenarios

### Integration Points
- Maintain compatibility with `backup_draft.py` state export
- Keep monte_carlo_state.json format
- Preserve real-time draft integration

### Validation
- Backtest against historical drafts
- Verify depth metrics align with actual injury outcomes
- Ensure natural patterns match expert consensus strategies

## Migration Path

1. **Preserve Core Logic**: Keep probability engine, opponent modeling
2. **Extend Depth**: Change round limit from 7 to 14
3. **Add Metrics**: Implement depth quality evaluation
4. **Pattern Detection**: Add emergent strategy classification
5. **Enhanced Output**: Show trade-off analysis clearly

## Key Principles

1. **Discovery Over Prescription**: Let strategies emerge, don't force them
2. **Multi-Dimensional Value**: Starters + Depth = Total Value
3. **Quantified Trade-offs**: Make depth vs starters trade-off explicit
4. **Natural Patterns**: Group similar draft approaches that emerge
5. **Objective Comparison**: Enable data-driven strategy selection

---

This framework transforms the Monte Carlo engine from a "maximize 7-round starters" tool to a comprehensive "discover optimal 14-round roster construction" system that balances starting lineup points with strategic depth building.