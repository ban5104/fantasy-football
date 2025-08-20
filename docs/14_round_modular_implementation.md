# 14-Round Strategy - Modular Architecture Implementation

## Overview
Adapts the 14-round depth strategy to the clean modular architecture in `src/monte_carlo/`

## Architecture Design

### New Modules to Add

#### 1. `src/monte_carlo/depth.py`
**Purpose**: Evaluate bench quality and calculate depth value

```python
class DepthEvaluator:
    """Evaluates roster depth quality beyond starting lineup"""
    
    def __init__(self, projections_df):
        self.projections = projections_df
        self.thresholds = {
            'excellent': 0.15,  # ≤15% gap from starter
            'useful': 0.25,     # 15-25% gap
            'depth_only': 0.40  # >25% gap
        }
    
    def evaluate_depth(self, roster_players, starters):
        """
        Calculate bench quality metrics
        
        Returns:
            dict: excellent_backups, useful_backups, depth_value, 
                  injury_resilience, flex_options
        """
        bench = [p for p in roster_players if p not in starters]
        
        # Calculate quality tiers
        excellent = self._count_by_threshold(bench, starters, 'excellent')
        useful = self._count_by_threshold(bench, starters, 'useful')
        
        # Calculate total depth value
        injury_factor = 0.15  # 15% expected injury rate
        depth_value = (excellent * 15 + useful * 8) * injury_factor
        
        return {
            'excellent_backups': excellent,
            'useful_backups': useful,
            'depth_value': depth_value,
            'injury_resilience': self._calculate_replacement_value(bench),
            'flex_options': self._count_flex_worthy(bench)
        }
```

#### 2. `src/monte_carlo/pattern_detector.py`
**Purpose**: Discover emergent strategies from simulation results

```python
class PatternDetector:
    """Detects natural draft patterns without predetermined strategies"""
    
    def detect_pattern(self, draft_history):
        """
        Analyze draft history to classify emergent strategy
        
        Args:
            draft_history: List of (round, player, position) tuples
            
        Returns:
            str: Pattern name (e.g., 'depth_first', 'fill_starters', 'balanced')
        """
        # Analyze position timing
        te_round = self._get_position_round(draft_history, 'TE')
        qb_round = self._get_position_round(draft_history, 'QB')
        rb_wr_mid = self._count_positions_in_range(draft_history, 
                                                   ['RB', 'WR'], 6, 10)
        
        # Natural pattern classification (discovered, not prescribed)
        if te_round >= 10:
            return 'punt_te'
        elif rb_wr_mid >= 4:
            return 'depth_loading'
        elif qb_round <= 5:
            return 'early_qb'
        elif self._count_rb_first_3(draft_history) >= 2:
            return 'rb_heavy_start'
        else:
            return 'balanced'
    
    def analyze_patterns(self, simulation_results):
        """Group and analyze emergent patterns from multiple simulations"""
        pattern_groups = defaultdict(list)
        
        for result in simulation_results:
            pattern = self.detect_pattern(result['draft_history'])
            pattern_groups[pattern].append(result)
        
        return self._calculate_pattern_metrics(pattern_groups)
```

### Modified Modules

#### `src/monte_carlo/simulator.py`
**Changes**: Extend to 14 rounds, integrate depth evaluation

```python
class DraftSimulator:
    def __init__(self, projections_df, pick_probabilities, 
                 opponent_model, n_rounds=14):  # Changed from 7
        self.n_rounds = n_rounds
        self.depth_evaluator = DepthEvaluator(projections_df)
        # ... existing init
    
    def calculate_roster_value(self, roster_players):
        """Enhanced with dual value system"""
        # Calculate starter points (existing)
        starters, starter_points = self._calculate_optimal_lineup(roster_players)
        
        # NEW: Calculate depth value
        depth_metrics = self.depth_evaluator.evaluate_depth(
            roster_players, starters
        )
        
        # Dual value system
        total_value = starter_points + depth_metrics['depth_value']
        
        return {
            'total_value': total_value,
            'starter_points': starter_points,
            'depth_metrics': depth_metrics,
            'starters': starters
        }
```

#### `src/monte_carlo/__init__.py`
**Changes**: Update API to support 14-round mode

```python
def run_simulation(strategy='balanced', n_rounds=14, n_sims=100):
    """
    Run draft simulation with optional round depth
    
    Args:
        strategy: Strategy name or 'discover' for natural pattern detection
        n_rounds: Number of rounds to simulate (7 or 14)
        n_sims: Number of simulations
    """
    simulator = DraftSimulator(
        projections_df=projections,
        pick_probabilities=probabilities,
        opponent_model=opponent_model,
        n_rounds=n_rounds
    )
    
    if strategy == 'discover':
        # Natural strategy discovery mode
        results = simulator.run_discovery_mode(n_sims)
        pattern_detector = PatternDetector()
        patterns = pattern_detector.analyze_patterns(results)
        return patterns
    else:
        # Traditional strategy mode (for compatibility)
        return simulator.run_strategy(strategy, n_sims)
```

### Updated Runner

#### `monte_carlo_runner.py`
**Changes**: Add 14-round mode and pattern discovery

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['compare', 'discover', 'balanced', 
                                        'zero_rb', 'rb_heavy', 'hero_rb'])
    parser.add_argument('--rounds', type=int, default=14, 
                       help='Number of rounds (7 or 14)')
    parser.add_argument('--n-sims', type=int, default=100)
    
    args = parser.parse_args()
    
    if args.mode == 'discover':
        # Natural pattern discovery mode
        patterns = run_simulation(
            strategy='discover',
            n_rounds=args.rounds,
            n_sims=args.n_sims
        )
        display_discovered_patterns(patterns)
    elif args.mode == 'compare':
        # Compare with depth analysis
        compare_strategies_with_depth(n_rounds=args.rounds)
    else:
        # Run specific strategy
        results = run_simulation(
            strategy=args.mode,
            n_rounds=args.rounds,
            n_sims=args.n_sims
        )
        display_results_with_tradeoffs(results)
```

## Key Design Decisions

### 1. Modular Separation
- **Depth evaluation** is completely separate from simulation logic
- **Pattern detection** runs after simulations, not during
- **Existing modules** remain largely unchanged

### 2. Natural Discovery
- No predetermined 14-round strategies
- Patterns emerge from simulation data
- Classification happens post-hoc

### 3. Dual Value System
- Maintains backward compatibility with 7-round mode
- Adds depth dimension without breaking existing code
- Clear separation between starter and depth value

### 4. Progressive Enhancement
- Start with basic depth metrics
- Add pattern detection
- Enhance with trade-off analysis
- All while maintaining clean architecture

## Migration Path

1. **Phase 1**: Create depth.py with basic evaluation (30 min)
2. **Phase 2**: Extend simulator.py for 14 rounds (15 min)
3. **Phase 3**: Create pattern_detector.py for discovery (30 min)
4. **Phase 4**: Update runner with new modes (15 min)
5. **Phase 5**: Test with existing draft state (15 min)

## Usage Examples

```bash
# Discover natural patterns in 14-round drafts
python monte_carlo_runner.py discover --rounds 14

# Compare strategies with depth analysis
python monte_carlo_runner.py compare --rounds 14

# Run specific strategy for 14 rounds
python monte_carlo_runner.py balanced --rounds 14

# Backward compatible 7-round mode
python monte_carlo_runner.py balanced --rounds 7
```

## Output Format

```
Emergent Pattern Analysis (14 rounds):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Discovered Patterns (100 simulations):
────────────────────────────────────────────────
Pattern         | Starter | Bench Quality  | Total | Freq
               | Points  | Exc | Useful   | Value |
───────────────┼─────────┼─────┼──────────┼───────┼──────
Depth Loading  | 1,380   |  3  |    4     | 1,467 | 22%
Fill Starters  | 1,425   |  0  |    2     | 1,451 | 18%
Balanced       | 1,405   |  2  |    3     | 1,460 | 15%

Trade-off Analysis:
• Depth Loading: -45 starter pts for +5 quality backups
  → Net gain: +16 total value ✓
• Fill Starters: +45 starter pts but weak depth
  → Net loss: -16 total value ✗
```

## Benefits of This Architecture

1. **Clean Separation**: Each module has a single responsibility
2. **Natural Discovery**: Strategies emerge from data, not assumptions
3. **Extensible**: Easy to add new depth metrics or pattern detection
4. **Testable**: Each module can be tested independently
5. **Backward Compatible**: 7-round mode still works
6. **Performance**: Can optimize each module separately if needed

This implementation maintains the clean modular design while adding sophisticated 14-round depth analysis and natural strategy discovery.