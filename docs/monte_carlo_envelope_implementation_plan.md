# Monte Carlo Simulation with Envelope Data - Comprehensive Implementation Plan

## Executive Summary
This document outlines the complete implementation plan for enhancing the existing Monte Carlo fantasy football draft simulation system to incorporate LOW/BASE/HIGH envelope data with full statistical rigor, based on extensive research and incorporating expert feedback.

## Table of Contents
1. [Overview](#overview)
2. [Phase 1: Core Distribution Framework](#phase-1-core-distribution-framework)
3. [Phase 2: Dynamic Replacement Levels](#phase-2-dynamic-replacement-levels)
4. [Phase 3: Statistical Metrics](#phase-3-statistical-metrics)
5. [Phase 4: Convergence & Early Stopping](#phase-4-convergence--early-stopping)
6. [Phase 5: Correlation Modeling](#phase-5-correlation-modeling)
7. [Phase 6: Decision Rules & Output](#phase-6-decision-rules--output)
8. [Implementation Checklist](#implementation-checklist)
9. [Quick Reference Code](#quick-reference-code)

## Overview

### Goals
- Transform deterministic projections to probabilistic envelope-based sampling
- Eliminate bias through per-simulation replacement level calculations
- Add robust statistical metrics (VaR, CVaR, confidence intervals)
- Maintain clean output format with progressive disclosure
- Ensure reproducibility and computational efficiency

### Key Improvements
- **Beta-PERT Distribution**: Optimal for bounded three-point estimates (LOW/BASE/HIGH)
- **Common Random Numbers**: Reduce variance in strategy comparisons
- **Dynamic Replacement**: Calculate per-simulation to match sampled world
- **Empirical CIs**: Use percentile method for non-Gaussian distributions
- **Early Stopping**: Dual-rule convergence for efficiency

## Phase 1: Core Distribution Framework

### 1.1 Distribution Implementation (`src/monte_carlo/distributions.py`)

```python
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class EnvelopeDistribution:
    """Manages player projection distributions using Beta-PERT"""
    
    def __init__(self, seed: int = 42, concentration: float = 4.0):
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.concentration = concentration
        self.clip_counts = {}
        
    def sample_all_players_pert(self, 
                                players_df, 
                                n_sims: int = 1000) -> Dict[int, np.ndarray]:
        """
        Sample all players using Beta-PERT with common random numbers
        
        Returns:
            Dict mapping player_id to array of n_sims sampled values
        """
        samples = {}
        
        for player_id, row in players_df.iterrows():
            # Get envelope data, fallback to BASE ± 20% if not available
            if 'low' in row and 'high' in row:
                low, base, high = row['low'], row['base'], row['high']
            else:
                base = row.get('proj', row.get('base', 100))
                low = base * 0.8
                high = base * 1.2
            
            # Handle degenerate case
            if high == low:
                samples[player_id] = np.full(n_sims, low)
                continue
            
            # Beta-PERT parameters
            alpha = 1 + self.concentration * (base - low) / (high - low)
            beta = 1 + self.concentration * (high - base) / (high - low)
            
            # Sample from Beta distribution
            u = self.rng.beta(alpha, beta, n_sims)
            samples[player_id] = low + u * (high - low)
            
            # Sanity check
            assert np.all(samples[player_id] >= low - 1e-6), f"Sample below low for {player_id}"
            assert np.all(samples[player_id] <= high + 1e-6), f"Sample above high for {player_id}"
        
        return samples
    
    def get_substream_generator(self, stream_id: int):
        """Get independent RNG for parallel processing"""
        return np.random.Generator(np.random.PCG64(self.rng.bit_generator.jumped(stream_id)))
```

### 1.2 RNG Management

```python
# Central RNG configuration
class SimulationConfig:
    def __init__(self, config_path: str = "config/simulation.yaml"):
        self.seed = 42  # Default, override from config
        self.n_teams = 14
        self.n_rounds = 14
        self.quick_n_sims = 1000
        self.production_n_sims = 5000
        self.correlation_rho = 0.15
        self.correlation_bounds = (0.7, 1.4)
        self.convergence_tolerance = 0.05
        self.early_stop_se_threshold = 5.0
```

### 1.3 Unit Tests for Distribution

```python
def test_beta_pert_sampling():
    """Verify Beta-PERT produces valid samples"""
    dist = EnvelopeDistribution(seed=42)
    
    # Test data
    test_player = pd.DataFrame({
        'low': [80], 'base': [100], 'high': [120]
    }, index=[0])
    
    samples = dist.sample_all_players_pert(test_player, n_sims=10000)
    
    # Check bounds
    assert np.all(samples[0] >= 80), "Samples below low"
    assert np.all(samples[0] <= 120), "Samples above high"
    
    # Check mean near mode (for Beta-PERT with c=4)
    expected_mean = (80 + 4*100 + 120) / 6  # ~100
    assert abs(np.mean(samples[0]) - expected_mean) < 1.0, "Mean too far from expected"
    
    # Check reasonable variance
    cv = np.std(samples[0]) / np.mean(samples[0])
    assert 0.05 < cv < 0.15, f"CV out of range: {cv}"
```

## Phase 2: Dynamic Replacement Levels

### 2.1 Per-Simulation Replacement Calculation

```python
from collections import defaultdict
from typing import Dict, List

class ReplacementCalculator:
    """Calculate replacement levels per simulation"""
    
    STARTERS_PER_POS = {
        'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 
        'FLEX': 1, 'K': 1, 'DST': 1
    }
    
    def compute_replacement_levels(self, 
                                  sampled_vals: Dict[int, np.ndarray],
                                  sim_idx: int,
                                  players_df,
                                  n_teams: int = 14) -> Dict[str, float]:
        """
        Calculate replacement level for each position in simulation
        
        CRITICAL: FLEX replacement from combined RB/WR/TE pool
        """
        pos_values = defaultdict(list)
        
        # Group sampled values by position
        for player_id, values in sampled_vals.items():
            if player_id not in players_df.index:
                continue
            pos = players_df.loc[player_id, 'pos']
            pos_values[pos].append(values[sim_idx])
        
        replacement = {}
        
        # Standard positions
        for pos in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
            if pos not in pos_values or not pos_values[pos]:
                replacement[pos] = 0
                continue
                
            sorted_vals = sorted(pos_values[pos], reverse=True)
            rank = n_teams * self.STARTERS_PER_POS.get(pos, 1)
            
            # Ensure rank is within bounds
            rank_idx = min(rank - 1, len(sorted_vals) - 1)
            rank_idx = max(0, rank_idx)
            replacement[pos] = sorted_vals[rank_idx]
        
        # FLEX from combined pool (CRITICAL FIX)
        flex_pool = (pos_values.get('RB', []) + 
                    pos_values.get('WR', []) + 
                    pos_values.get('TE', []))
        
        if flex_pool:
            sorted_flex = sorted(flex_pool, reverse=True)
            flex_rank = n_teams * self.STARTERS_PER_POS['FLEX']
            flex_idx = min(flex_rank - 1, len(sorted_flex) - 1)
            flex_idx = max(0, flex_idx)
            replacement['FLEX'] = sorted_flex[flex_idx]
        else:
            replacement['FLEX'] = 0
        
        # Sanity checks
        for pos, val in replacement.items():
            assert not np.isnan(val), f"NaN replacement for {pos}"
            if pos_values.get(pos):
                assert val <= max(pos_values[pos]), f"Replacement > best for {pos}"
        
        return replacement
```

### 2.2 Bench Marginal Value Calculation

```python
def calculate_bench_marginal(bench_players: List[Dict],
                            sampled_vals: Dict[int, np.ndarray],
                            replacement: Dict[str, float],
                            sim_idx: int) -> Tuple[float, int, int]:
    """
    Calculate true marginal value vs replacement
    
    Returns:
        total_marginal: Sum of marginal values
        excellent_count: Backups within 15% of starter
        useful_count: Backups within 25% of starter
    """
    total_marginal = 0.0
    excellent_count = 0
    useful_count = 0
    
    for player in bench_players:
        player_id = player['id']
        player_val = sampled_vals[player_id][sim_idx]
        pos = player['pos']
        
        # Get appropriate replacement level
        if pos in ['RB', 'WR', 'TE']:
            # Use better of position-specific or FLEX
            repl = min(replacement.get(pos, 0), 
                      replacement.get('FLEX', 0))
        else:
            repl = replacement.get(pos, 0)
        
        # Marginal value calculation
        marginal = max(0, player_val - repl)
        total_marginal += marginal
        
        # Quality thresholds (vs starter, not replacement)
        starter_val = player.get('starter_comparison', player_val * 1.3)
        gap_pct = (starter_val - player_val) / starter_val if starter_val > 0 else 1.0
        
        if gap_pct <= 0.15:
            excellent_count += 1
        elif gap_pct <= 0.25:
            useful_count += 1
    
    return total_marginal, excellent_count, useful_count
```

## Phase 3: Statistical Metrics

### 3.1 Robust Statistical Calculations

```python
import scipy.stats as stats

class StatisticalAnalyzer:
    """Calculate robust statistics with proper confidence intervals"""
    
    @staticmethod
    def calculate_statistics(total_vals: np.ndarray) -> Dict:
        """
        Use percentile CIs for non-Gaussian distributions
        """
        n = len(total_vals)
        
        # Basic statistics
        mean_val = np.mean(total_vals)
        std_val = np.std(total_vals, ddof=1)
        
        # Percentile-based 95% CI (robust for non-normal)
        ci_95 = np.percentile(total_vals, [2.5, 97.5])
        
        # Value at Risk (empirical percentiles)
        var_5 = np.percentile(total_vals, 5)
        var_10 = np.percentile(total_vals, 10)
        
        # Conditional Value at Risk (expected shortfall)
        sorted_vals = np.sort(total_vals)
        worst_5pct = sorted_vals[:max(1, int(0.05 * n))]
        cvar_5 = np.mean(worst_5pct)
        worst_10pct = sorted_vals[:max(1, int(0.10 * n))]
        cvar_10 = np.mean(worst_10pct)
        
        # Standard error (for mean only, not tail metrics)
        se_mean = std_val / np.sqrt(n)
        
        # Additional metrics
        cv = std_val / mean_val if mean_val != 0 else float('inf')
        skewness = stats.skew(total_vals)
        kurtosis = stats.kurtosis(total_vals)
        
        return {
            'mean': mean_val,
            'std': std_val,
            'ci_95': ci_95,
            'ci_width': ci_95[1] - ci_95[0],
            'var_5': var_5,
            'var_10': var_10,
            'cvar_5': cvar_5,
            'cvar_10': cvar_10,
            'se_mean': se_mean,
            'cv': cv,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'n_sims': n
        }
```

### 3.2 Strategy Comparison with Common Random Numbers

```python
from itertools import combinations

def compare_strategies_common_rng(samples: Dict[int, np.ndarray],
                                 strategies: List[str],
                                 my_team_idx: int,
                                 simulator) -> Tuple[Dict, Dict]:
    """
    Compare strategies using same random draws for variance reduction
    
    CRITICAL: Each strategy uses SAME sampled values for each sim
    """
    results = {}
    detailed_results = {}
    
    # Run each strategy with common samples
    for strategy_name in strategies:
        total_vals = []
        starter_vals = []
        bench_marginals = []
        exc_counts = []
        use_counts = []
        
        n_sims = len(samples[list(samples.keys())[0]])
        
        for sim_idx in range(n_sims):
            # Simulate draft and evaluate with SAME samples
            roster = simulator.simulate_single_draft(
                my_team_idx, strategy_name, 
                samples, sim_idx
            )
            
            eval_result = simulator.evaluate_roster(
                roster, samples, sim_idx
            )
            
            total_vals.append(eval_result['total_value'])
            starter_vals.append(eval_result['starter_points'])
            bench_marginals.append(eval_result['bench_marginal'])
            exc_counts.append(eval_result['excellent_count'])
            use_counts.append(eval_result['useful_count'])
        
        results[strategy_name] = np.array(total_vals)
        detailed_results[strategy_name] = {
            'total_values': np.array(total_vals),
            'starter_points': np.array(starter_vals),
            'bench_marginals': np.array(bench_marginals),
            'excellent_counts': np.array(exc_counts),
            'useful_counts': np.array(use_counts)
        }
    
    # Head-to-head comparisons using paired samples
    comparisons = {}
    for s1, s2 in combinations(strategies, 2):
        # Direct comparison on same simulations
        wins = np.sum(results[s1] > results[s2])
        p_win = wins / len(results[s1])
        
        # Bootstrap CI for win probability
        n_bootstrap = 1000
        win_samples = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(results[s1]), len(results[s1]), replace=True)
            boot_wins = np.sum(results[s1][idx] > results[s2][idx])
            win_samples.append(boot_wins / len(idx))
        
        win_ci = np.percentile(win_samples, [2.5, 97.5])
        
        comparisons[f"{s1}_vs_{s2}"] = {
            'p_win': p_win,
            'ci_95': win_ci,
            'mean_diff': np.mean(results[s1] - results[s2]),
            'se_diff': np.std(results[s1] - results[s2]) / np.sqrt(len(results[s1]))
        }
    
    return detailed_results, comparisons
```

## Phase 4: Convergence & Early Stopping

### 4.1 Dual-Rule Convergence Detection

```python
class ConvergenceMonitor:
    """Monitor and detect convergence using dual rules"""
    
    def __init__(self, 
                tolerance_pct: float = 0.05,
                se_threshold: float = 5.0):
        self.tolerance_pct = tolerance_pct
        self.se_threshold = se_threshold
        self.convergence_history = []
    
    def check_convergence(self, 
                         total_vals: np.ndarray,
                         best_two_arrays: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> bool:
        """
        Dual rule for early stopping:
        1. CI width < 5% of mean
        2. SE of delta between best two < threshold
        """
        n = len(total_vals)
        mean_tv = np.mean(total_vals)
        se_tv = np.std(total_vals, ddof=1) / np.sqrt(n)
        ci_width = 2 * 1.96 * se_tv
        
        # Rule 1: Relative CI width
        rule1_met = ci_width < self.tolerance_pct * mean_tv
        
        # Rule 2: Absolute SE of difference
        rule2_met = False
        if best_two_arrays is not None:
            delta_vals = best_two_arrays[0] - best_two_arrays[1]
            se_delta = np.std(delta_vals, ddof=1) / np.sqrt(len(delta_vals))
            rule2_met = se_delta < self.se_threshold
        
        # Track convergence metrics
        self.convergence_history.append({
            'n_sims': n,
            'mean': mean_tv,
            'ci_width': ci_width,
            'ci_width_pct': ci_width / mean_tv if mean_tv != 0 else float('inf'),
            'rule1_met': rule1_met,
            'rule2_met': rule2_met,
            'converged': rule1_met or rule2_met
        })
        
        return rule1_met or rule2_met
    
    def adaptive_simulation(self, 
                          simulator,
                          samples: Dict,
                          strategies: List[str],
                          checkpoints: List[int] = [1000, 2000, 3000, 4000, 5000]):
        """
        Run simulations with early stopping at checkpoints
        """
        all_results = {s: [] for s in strategies}
        
        for checkpoint in checkpoints:
            # Run batch to checkpoint
            start_idx = len(all_results[strategies[0]])
            end_idx = checkpoint
            
            for sim_idx in range(start_idx, end_idx):
                for strategy in strategies:
                    result = simulator.run_single_sim(strategy, samples, sim_idx)
                    all_results[strategy].append(result)
            
            # Check convergence
            total_vals = {s: np.array([r['total'] for r in all_results[s]]) 
                         for s in strategies}
            
            # Get best two strategies
            mean_vals = {s: np.mean(v) for s, v in total_vals.items()}
            sorted_strategies = sorted(mean_vals.items(), key=lambda x: x[1], reverse=True)
            best_two = (total_vals[sorted_strategies[0][0]], 
                       total_vals[sorted_strategies[1][0]])
            
            if self.check_convergence(total_vals[strategies[0]], best_two):
                logger.info(f"Converged at {checkpoint} simulations")
                break
        
        return all_results, self.convergence_history
```

## Phase 5: Correlation Modeling

### 5.1 Team-Based Correlation with Validation

```python
class CorrelationModel:
    """Lightweight team correlation with monitoring"""
    
    def __init__(self, 
                base_rho: float = 0.15,
                qb_stack_rho: float = 0.25,
                bounds: Tuple[float, float] = (0.7, 1.4)):
        self.base_rho = base_rho
        self.qb_stack_rho = qb_stack_rho
        self.bounds = bounds
        self.clip_log = []
    
    def apply_team_correlation(self,
                              sampled_vals: Dict[int, np.ndarray],
                              sim_idx: int,
                              players_df,
                              rng) -> Dict[int, np.ndarray]:
        """
        Apply team multiplier with double clipping and logging
        """
        clip_count = 0
        total_adjustments = 0
        
        # Group players by team
        team_players = defaultdict(list)
        for player_id in sampled_vals.keys():
            if player_id in players_df.index:
                team = players_df.loc[player_id, 'team']
                team_players[team].append(player_id)
        
        # Apply correlation by team
        for team, player_ids in team_players.items():
            if len(player_ids) < 2:
                continue  # No correlation for single players
            
            # Check for QB-stack (QB + WR/TE on same team)
            has_qb = any(players_df.loc[pid, 'pos'] == 'QB' 
                        for pid in player_ids if pid in players_df.index)
            has_receiver = any(players_df.loc[pid, 'pos'] in ['WR', 'TE'] 
                             for pid in player_ids if pid in players_df.index)
            
            # Use enhanced correlation for QB stacks
            rho = self.qb_stack_rho if (has_qb and has_receiver) else self.base_rho
            
            # Sample team multiplier
            team_mult = rng.normal(1.0, rho)
            team_mult = np.clip(team_mult, self.bounds[0], self.bounds[1])
            
            # Apply to all players on team
            for player_id in player_ids:
                original = sampled_vals[player_id][sim_idx]
                
                # Apply multiplier
                correlated = original * team_mult
                
                # Clip to player's envelope bounds
                if player_id in players_df.index:
                    low = players_df.loc[player_id, 'low']
                    high = players_df.loc[player_id, 'high']
                    final = np.clip(correlated, low, high)
                    
                    if final != correlated:
                        clip_count += 1
                    
                    sampled_vals[player_id][sim_idx] = final
                    total_adjustments += 1
        
        # Log clipping frequency
        clip_rate = clip_count / total_adjustments if total_adjustments > 0 else 0
        self.clip_log.append({
            'sim_idx': sim_idx,
            'clip_count': clip_count,
            'total_adjustments': total_adjustments,
            'clip_rate': clip_rate
        })
        
        # Warning if excessive clipping
        if clip_rate > 0.1:
            logger.warning(f"High clipping rate in sim {sim_idx}: {clip_rate:.2%}")
        
        return sampled_vals
    
    def get_clipping_report(self) -> Dict:
        """Analyze clipping patterns for diagnostics"""
        if not self.clip_log:
            return {}
        
        clip_rates = [log['clip_rate'] for log in self.clip_log]
        return {
            'mean_clip_rate': np.mean(clip_rates),
            'max_clip_rate': np.max(clip_rates),
            'high_clip_sims': sum(1 for r in clip_rates if r > 0.1),
            'total_sims': len(clip_rates)
        }
```

## Phase 6: Decision Rules & Output

### 6.1 Enhanced Decision Framework

```python
class StrategyRecommender:
    """Make statistically-informed strategy recommendations"""
    
    def __init__(self, starter_tolerance: float = 15.0):
        self.starter_tolerance = starter_tolerance
    
    def make_recommendation(self,
                           detailed_results: Dict,
                           comparisons: Dict,
                           stats: Dict) -> Dict:
        """
        Smart recommendations using bench marginal and statistical confidence
        """
        strategies = list(detailed_results.keys())
        
        # Calculate key metrics for each strategy
        metrics = {}
        for strategy in strategies:
            result = detailed_results[strategy]
            stat = stats[strategy]
            
            metrics[strategy] = {
                'starter_mean': np.mean(result['starter_points']),
                'bench_marginal_mean': np.mean(result['bench_marginals']),
                'total_mean': stat['mean'],
                'total_ci': stat['ci_95'],
                'var_10': stat['var_10'],
                'cvar_10': stat['cvar_10'],
                'excellent_mean': np.mean(result['excellent_counts']),
                'useful_mean': np.mean(result['useful_counts'])
            }
        
        # Find best by different criteria
        best_starter = max(strategies, key=lambda s: metrics[s]['starter_mean'])
        best_total = max(strategies, key=lambda s: metrics[s]['total_mean'])
        best_risk_adjusted = max(strategies, key=lambda s: metrics[s]['var_10'])
        
        # Decision logic
        recommendations = []
        
        for strategy in strategies:
            score = 0
            reasons = []
            
            # Check starter points difference
            starter_diff = metrics[best_starter]['starter_mean'] - metrics[strategy]['starter_mean']
            
            if starter_diff <= self.starter_tolerance:
                # Within tolerance - consider other factors
                score += metrics[strategy]['total_mean']
                
                # Bench marginal value
                bench_advantage = (metrics[strategy]['bench_marginal_mean'] - 
                                 metrics[best_starter]['bench_marginal_mean'])
                if bench_advantage > 0:
                    score += bench_advantage
                    reasons.append(f"+{bench_advantage:.1f} bench value")
                
                # Win probability vs best
                vs_best_key = f"{strategy}_vs_{best_total}"
                if vs_best_key in comparisons:
                    p_win = comparisons[vs_best_key]['p_win']
                    if p_win > 0.6:
                        score += 10
                        reasons.append(f"{p_win:.0%} win prob")
                
                # Risk consideration
                if strategy == best_risk_adjusted:
                    score += 5
                    reasons.append("best downside protection")
            else:
                # Starter deficit too large
                score = metrics[strategy]['total_mean'] - starter_diff * 2
                reasons.append(f"-{starter_diff:.0f} starter deficit")
            
            recommendations.append({
                'strategy': strategy,
                'score': score,
                'reasons': reasons,
                'metrics': metrics[strategy]
            })
        
        # Sort and rank
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'recommendations': recommendations,
            'best_overall': recommendations[0]['strategy'],
            'confidence': self._calculate_confidence(recommendations, comparisons)
        }
    
    def _calculate_confidence(self, recommendations, comparisons):
        """Calculate confidence in recommendation"""
        if len(recommendations) < 2:
            return 1.0
        
        best = recommendations[0]['strategy']
        second = recommendations[1]['strategy']
        
        # Check head-to-head
        key = f"{best}_vs_{second}"
        if key in comparisons:
            return comparisons[key]['p_win']
        
        # Fallback to score difference
        score_diff = recommendations[0]['score'] - recommendations[1]['score']
        return min(0.5 + score_diff / 100, 1.0)
```

### 6.2 Output Formatter

```python
class OutputFormatter:
    """Format results with progressive disclosure"""
    
    def format_results(self,
                      detailed_results: Dict,
                      comparisons: Dict,
                      stats: Dict,
                      recommendation: Dict,
                      show_confidence: bool = False,
                      show_risk: bool = False) -> str:
        """
        Create formatted output table with optional details
        """
        output = []
        
        # Header
        n_sims = stats[list(stats.keys())[0]]['n_sims']
        output.append(f"14-Round Strategy Analysis (N={n_sims})")
        output.append("=" * 65)
        output.append("")
        
        # Main table
        output.append("Strategy      │ Starter │ Bench Quality │ Total  │ Risk")
        output.append("             │ Points  │ Exc │ Useful  │ Value  │ VaR₁₀")
        output.append("─" * 13 + "┼" + "─" * 9 + "┼" + "─" * 5 + "┼" + "─" * 9 + "┼" + "─" * 8 + "┼" + "─" * 6)
        
        # Data rows
        strategies = sorted(detailed_results.keys(), 
                          key=lambda s: stats[s]['mean'], 
                          reverse=True)
        
        for strategy in strategies:
            result = detailed_results[strategy]
            stat = stats[strategy]
            
            # Main values
            starter = np.mean(result['starter_points'])
            exc = np.mean(result['excellent_counts'])
            use = np.mean(result['useful_counts'])
            total = stat['mean']
            var10 = stat['var_10']
            
            # Format main row
            row = f"{strategy:12} │ {starter:6.0f}  │ {exc:3.1f} │ {use:6.1f}  │ {total:6.0f} │ {var10:5.0f}"
            
            # Add significance markers
            if strategy == recommendation['best_overall']:
                row += " **"
            elif recommendation['recommendations'][0]['score'] - \
                 next(r['score'] for r in recommendation['recommendations'] 
                      if r['strategy'] == strategy) < 10:
                row += " *"
            
            output.append(row)
            
            # Confidence intervals (if requested)
            if show_confidence:
                ci_row = f"{'':12} │[{stat['ci_95'][0]:5.0f},{stat['ci_95'][1]:5.0f}]│"
                exc_se = np.std(result['excellent_counts']) / np.sqrt(len(result['excellent_counts']))
                use_se = np.std(result['useful_counts']) / np.sqrt(len(result['useful_counts']))
                ci_row += f"[±{exc_se:.1f}]│[±{use_se:.1f}]   │" 
                ci_row += f"[{stat['ci_95'][0]:.0f},{stat['ci_95'][1]:.0f}]│"
                output.append(ci_row)
        
        output.append("")
        
        # Head-to-head comparisons
        if len(strategies) >= 2:
            best = strategies[0]
            second = strategies[1]
            key = f"{best}_vs_{second}"
            if key in comparisons:
                comp = comparisons[key]
                output.append(f"Head-to-Head: P({best} > {second}) = {comp['p_win']:.2f} "
                            f"[{comp['ci_95'][0]:.2f}, {comp['ci_95'][1]:.2f}]")
        
        # Decision
        output.append(f"Decision: {recommendation['best_overall']} "
                     f"(confidence: {recommendation['confidence']:.0%})")
        
        # Reasons
        best_rec = recommendation['recommendations'][0]
        if best_rec['reasons']:
            output.append(f"Reasons: {', '.join(best_rec['reasons'])}")
        
        # Risk metrics (if requested)
        if show_risk:
            output.append("")
            output.append("Risk Analysis:")
            for strategy in strategies[:3]:  # Top 3 only
                stat = stats[strategy]
                output.append(f"  {strategy:12}: VaR₅={stat['var_5']:.0f}, "
                            f"CVaR₅={stat['cvar_5']:.0f}, "
                            f"VaR₁₀={stat['var_10']:.0f}, "
                            f"CVaR₁₀={stat['cvar_10']:.0f}")
        
        return "\n".join(output)
```

## Implementation Checklist

### Phase 1: Core Distribution (Day 1) - ✅ COMPLETED
- [x] Create `distributions.py` with Beta-PERT implementation
- [x] Add envelope data loading to `probability.py`
- [x] Implement central RNG management
- [x] Unit test: Verify samples within bounds and reasonable mean
- [x] Performance test: 300 players × 1000 samples < 1 second

### Phase 2: Replacement Levels (Day 1) - ✅ COMPLETED
- [x] Create `replacement.py` module
- [x] Implement per-simulation calculation
- [x] Fix FLEX replacement from combined pool
- [x] Add bench marginal calculation
- [x] Unit test: Replacement monotonicity and sanity

### Phase 3: Statistical Framework (Day 2) - ❌ DEFERRED
- [ ] Create `statistics.py` with percentile CIs
- [ ] Implement VaR/CVaR calculations
- [ ] Add bootstrap confidence for comparisons
- [ ] Create convergence monitoring
- [ ] Unit test: CI coverage and statistical properties

### Phase 4: Strategy Comparison (Day 2) - 🔄 PARTIAL
- [x] Implement common random numbers
- [ ] Add head-to-head probability calculation
- [ ] Create recommendation engine
- [ ] Build output formatter
- [x] Integration test: Full simulation pipeline

### Phase 5: Optimization (Day 3) - ❌ DEFERRED
- [ ] Vectorize sampling operations
- [ ] Add parallel strategy evaluation
- [ ] Implement early stopping
- [ ] Add correlation model
- [ ] Performance test: 5000 sims < 2 minutes

### Phase 6: Integration (Day 3) - 🔄 PARTIAL
- [x] Update `monte_carlo_runner.py`
- [ ] Add progressive disclosure options
- [x] Test with live draft integration
- [ ] Create debug output capability
- [ ] Documentation and examples

## Quick Reference Code

### Main Simulation Loop
```python
# Initialize
config = SimulationConfig()
dist = EnvelopeDistribution(seed=config.seed)
replacement_calc = ReplacementCalculator()
correlator = CorrelationModel()
analyzer = StatisticalAnalyzer()
convergence = ConvergenceMonitor()

# Sample once for all strategies (common random numbers)
samples = dist.sample_all_players_pert(players_df, n_sims=config.production_n_sims)

# Apply correlation
for sim_idx in range(n_sims):
    samples = correlator.apply_team_correlation(samples, sim_idx, players_df, dist.rng)

# Evaluate strategies in parallel
from joblib import Parallel, delayed

def evaluate_strategy(strategy_name):
    return simulator.run_all_sims(strategy_name, samples)

results = Parallel(n_jobs=-1)(
    delayed(evaluate_strategy)(s) for s in strategies
)

# Compare and recommend
detailed_results, comparisons = compare_strategies_common_rng(samples, strategies, my_team_idx, simulator)
stats = {s: analyzer.calculate_statistics(detailed_results[s]['total_values']) 
         for s in strategies}
recommendation = recommender.make_recommendation(detailed_results, comparisons, stats)

# Format output
output = formatter.format_results(detailed_results, comparisons, stats, recommendation,
                                 show_confidence=True, show_risk=True)
print(output)
```

### Validation Tests
```python
def validate_implementation():
    """Run all validation checks"""
    
    # 1. Distribution sanity
    assert all(low <= samples <= high), "Samples out of bounds"
    assert abs(np.mean(samples) - expected_mean) < tolerance, "Mean check failed"
    
    # 2. Replacement monotonicity
    assert all(repl <= best_in_pos), "Replacement exceeds best player"
    assert not any(np.isnan(replacements)), "NaN in replacements"
    
    # 3. Convergence stability
    for n in [1000, 3000, 5000]:
        results_n = run_sims(n)
        assert check_convergence(results_n), f"Failed to converge at {n}"
    
    # 4. Common RNG reproducibility
    results1 = run_with_seed(42)
    results2 = run_with_seed(42)
    assert np.array_equal(results1, results2), "Not reproducible"
    
    # 5. Performance benchmarks
    start = time.time()
    run_sims(1000)
    assert time.time() - start < 30, "Performance target missed"
    
    print("✅ All validation checks passed")
```

## Conclusion

This comprehensive implementation plan transforms the Monte Carlo simulation system from deterministic projections to a sophisticated probabilistic framework while:

1. **Maintaining** the clean output format you desire
2. **Adding** robust statistical metrics and confidence intervals
3. **Ensuring** reproducibility through proper RNG management
4. **Optimizing** performance through vectorization and parallelization
5. **Providing** progressive disclosure for different user needs

The system will produce statistically sound strategy recommendations with quantified uncertainty, enabling better draft decisions based on rigorous analysis rather than point estimates.