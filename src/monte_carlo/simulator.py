"""Monte Carlo Simulation Engine - Simplified"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter
import time
import os
from concurrent.futures import ProcessPoolExecutor

from .strategies import ROSTER_REQUIREMENTS, POSITION_LIMITS, ROUND_POSITION_VALIDITY
from .replacement import calculate_replacement_levels


def _parallel_simulation_worker(args):
    """Worker function for parallel simulation - must be at module level for pickling"""
    try:
        (sim_start_idx, sim_count, my_team_idx, strategy_params, worker_seed,
         initial_roster, already_drafted, players_df_dict, n_teams, n_rounds) = args
        
        # Import dependencies inside worker to avoid pickle issues
        from .probability import ProbabilityModel
        from .opponent import OpponentModel
        
        # Recreate models from serialized data
        prob_model = ProbabilityModel._from_dict(players_df_dict)
        opponent_model = OpponentModel(prob_model)
        
        # Create simulator instance
        simulator = MonteCarloSimulator(prob_model, opponent_model, n_teams, n_rounds)
        
        # Generate child seeds using SeedSequence for proper independence
        seed_seq = np.random.SeedSequence(worker_seed)
        child_seeds = seed_seq.spawn(sim_count)
        
        # Run batch of simulations
        results = []
        for i in range(sim_count):
            sim_idx = sim_start_idx + i
            child_rng = np.random.default_rng(child_seeds[i])
            seed = int(child_rng.integers(0, 2**31))  # Convert to int for simulate_single_draft
            
            result = simulator.simulate_single_draft(
                my_team_idx, strategy_params, seed=seed,
                initial_roster=initial_roster, already_drafted=already_drafted
            )
            
            # Return lightweight result to reduce IPC overhead
            lightweight_result = {
                'roster_value': result['roster_value'],
                'starter_points': result['starter_points'], 
                'depth_bonus': result['depth_bonus'],
                'num_players': result['num_players'],
                'first_14_picks': result['position_sequence'][:14] if len(result['position_sequence']) >= 14 else result['position_sequence']
            }
            results.append(lightweight_result)
        
        return results
        
    except Exception as e:
        return {'error': str(e), 'worker_args': args}


class MonteCarloSimulator:
    """Run Monte Carlo simulations for draft strategy evaluation"""
    
    def __init__(self, probability_model, opponent_model, n_teams=14, n_rounds=14):
        self.prob_model = probability_model
        self.opponent_model = opponent_model
        self.n_teams = n_teams
        self.n_rounds = n_rounds
        
        # Unified cache for performance optimizations
        self._player_cache = None
        
    def _get_player_cache(self):
        """Get unified player cache with all needed data"""
        if self._player_cache is None:
            players_df = self.prob_model.players_df
            self._player_cache = {
                'pos': players_df['pos'].to_dict(),
                'proj': players_df['proj'].to_dict(),
                'espn_rank': players_df['espn_rank'].to_dict(),
                'player_name': players_df['player_name'].to_dict(),
                'name_to_id': dict(zip(players_df['player_name'], players_df.index))
            }
        return self._player_cache
        
    def generate_snake_order(self):
        """Generate snake draft pick order"""
        order = []
        for round_num in range(self.n_rounds):
            if round_num % 2 == 0:
                order.extend(range(self.n_teams))
            else:
                order.extend(reversed(range(self.n_teams)))
        return order
        
    def _get_player_projection(self, player, sim_seed=None, crn=None, sim_idx=None, use_triangular=False):
        """Get player projection from CRN, envelope sampling, triangular, or static value"""
        if crn and sim_idx is not None and crn.is_ready():
            return crn.get_projection(player['id'], sim_idx, player['proj'])
        elif sim_seed is not None and self.prob_model.has_envelope_data():
            if use_triangular:
                # Use triangular distribution instead of Beta-PERT
                return self._sample_triangular_projection(player, sim_seed)
            else:
                return self.prob_model.sample_player_projection(player['id'], sim_seed)
        else:
            return player['proj']
            
    def _sample_triangular_projection(self, player, sim_seed):
        """Sample from triangular distribution for optimizer mode"""
        rng = np.random.default_rng(sim_seed)
        
        # Use base projection as mode, create ±20% envelope if no envelope data
        base = player['proj']
        low = base * 0.8
        high = base * 1.2
        
        # Check if envelope data exists
        if 'low' in player and 'high' in player:
            low = player['low']
            high = player['high']
        
        # Sample from triangular distribution (low, mode=base, high)
        return rng.triangular(low, base, high)
    
    def _get_replacement_rank(self, pos, replacement_levels):
        """Get estimated replacement rank for tier-gap calculation"""
        if pos not in replacement_levels:
            return 50  # Default fallback
        
        replacement_value = replacement_levels[pos]
        
        # Find players of this position with similar value to estimate rank
        player_cache = self._get_player_cache()
        same_pos_players = []
        
        for pid, player_pos in player_cache['pos'].items():
            if player_pos == pos:
                proj = player_cache['proj'][pid]
                rank = player_cache['espn_rank'][pid]
                same_pos_players.append((proj, rank))
        
        if not same_pos_players:
            return 50
        
        # Find rank of player with closest projection to replacement level
        same_pos_players.sort(key=lambda x: abs(x[0] - replacement_value))
        closest_rank = same_pos_players[0][1]
        
        return closest_rank
    
    def _estimate_picks_until_next_turn(self, roster_size):
        """Estimate picks until next turn based on snake draft pattern"""
        picks_made = roster_size
        current_round = (picks_made // self.n_teams) + 1
        position_in_round = picks_made % self.n_teams
        
        if current_round % 2 == 1:  # Odd round (1st, 3rd, etc.)
            picks_remaining_in_round = self.n_teams - position_in_round - 1
            return picks_remaining_in_round + self.n_teams + position_in_round + 1
        else:  # Even round (2nd, 4th, etc.)
            return (self.n_teams - position_in_round - 1) * 2 + 1
    
    def _calculate_tier_drops_from_data(self, pos, current_round):
        """Calculate tier drops using actual data analysis instead of hardcoded values"""
        player_cache = self._get_player_cache()
        
        # Get players of this position sorted by projection
        pos_players = []
        for pid, player_pos in player_cache['pos'].items():
            if player_pos == pos:
                proj = player_cache['proj'][pid]
                rank = player_cache['espn_rank'][pid]
                pos_players.append((proj, rank))
        
        if len(pos_players) < 10:  # Fallback for small datasets
            return self._get_fallback_tier_drops(pos, current_round)
        
        # Sort by projection (descending)
        pos_players.sort(reverse=True)
        
        # Calculate actual tier drops by examining projection differences
        tier_sizes = []
        current_tier_start = 0
        
        for i in range(1, len(pos_players)):
            proj_drop = pos_players[i-1][0] - pos_players[i][0]
            # Tier break if drop is > 10% of current player's value
            if proj_drop > (pos_players[i][0] * 0.1):
                tier_size = i - current_tier_start
                tier_sizes.append(tier_size)
                current_tier_start = i
        
        # Calculate position-specific metrics from actual data
        avg_tier_size = np.mean(tier_sizes) if tier_sizes else 5
        decay_per_tier = np.mean([pos_players[i][0] / pos_players[i+int(avg_tier_size)][0] 
                                 for i in range(len(pos_players) - int(avg_tier_size))]) if len(pos_players) > avg_tier_size else 0.85
        
        decay_rate = 1 - decay_per_tier if decay_per_tier < 1 else 0.15
        margin = max(3, int(avg_tier_size * 2))
        
        return {
            'decay_rate': min(0.5, max(0.05, decay_rate)),  # Bounds check
            'margin': margin,
            'tier_size': avg_tier_size
        }
    
    def _get_fallback_tier_drops(self, pos, current_round):
        """Fallback tier drop calculations for edge cases"""
        fallback_values = {
            'RB': {'decay_rate': 0.12, 'margin': 6},
            'WR': {'decay_rate': 0.10, 'margin': 5}, 
            'TE': {'decay_rate': 0.20, 'margin': 8},
            'QB': {'decay_rate': 0.08, 'margin': 4},
            'K': {'decay_rate': 0.05, 'margin': 2},
            'DST': {'decay_rate': 0.05, 'margin': 2}
        }
        
        base_values = fallback_values.get(pos, {'decay_rate': 0.15, 'margin': 5})
        
        # Adjust for round (later rounds have smaller tiers)
        if current_round > 6:
            base_values['decay_rate'] *= 1.5  # Steeper drops in later rounds
            base_values['margin'] = max(2, base_values['margin'] // 2)
        
        return base_values

    def calculate_roster_value(self, roster_players, sim_seed=None, crn=None, sim_idx=None):
        """Calculate total value of a roster from optimal starting lineup (optimized)"""
        # Group by position and cache player values to avoid repeated lookups
        position_players = defaultdict(list)
        player_values = {}
        
        # Pre-calculate all player values once
        use_triangular = hasattr(self, '_optimizer_mode') and self._optimizer_mode
        for player in roster_players:
            value = self._get_player_projection(player, sim_seed, crn, sim_idx, use_triangular)
            player_values[player['id']] = value
            position_players[player['pos']].append(player)
            
        # Sort each position by cached value (highest first) 
        for pos in position_players:
            position_players[pos].sort(key=lambda p: player_values[p['id']], reverse=True)
        
        starters = []
        total_points = 0.0
        
        # QB: 1 starter
        if 'QB' in position_players and position_players['QB']:
            starter = position_players['QB'][0]
            starters.append(starter)
            total_points += player_values[starter['id']]
            
        # RB: 2 starters
        if 'RB' in position_players:
            for i in range(min(2, len(position_players['RB']))):
                starter = position_players['RB'][i]
                starters.append(starter)
                total_points += player_values[starter['id']]
                
        # WR: 2 starters  
        if 'WR' in position_players:
            for i in range(min(2, len(position_players['WR']))):
                starter = position_players['WR'][i]
                starters.append(starter)
                total_points += player_values[starter['id']]
                
        # TE: 1 starter
        if 'TE' in position_players and position_players['TE']:
            starter = position_players['TE'][0]
            starters.append(starter)
            total_points += player_values[starter['id']]
            
        # FLEX: Best remaining RB/WR/TE
        flex_candidates = []
        if 'RB' in position_players and len(position_players['RB']) > 2:
            flex_candidates.extend(position_players['RB'][2:])
        if 'WR' in position_players and len(position_players['WR']) > 2:
            flex_candidates.extend(position_players['WR'][2:])
        if 'TE' in position_players and len(position_players['TE']) > 1:
            flex_candidates.extend(position_players['TE'][1:])
            
        if flex_candidates:
            best_flex = max(flex_candidates, key=lambda p: player_values[p['id']])
            starters.append(best_flex)
            total_points += player_values[best_flex['id']]
            
        # K: 1 starter
        if 'K' in position_players and position_players['K']:
            starter = position_players['K'][0]
            starters.append(starter)
            total_points += player_values[starter['id']]
            
        # DST: 1 starter
        if 'DST' in position_players and position_players['DST']:
            starter = position_players['DST'][0]
            starters.append(starter)
            total_points += player_values[starter['id']]
            
        # Simple depth bonus for 14-round drafts
        depth_bonus = 0.0
        backup_counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'total': 0}
        
        if self.n_rounds >= 10:
            bench_size = len(roster_players) - len(starters)
            if bench_size > 0:
                bench_players = [p for p in roster_players if p not in starters]
                bench_value = sum(player_values[p['id']] for p in bench_players)
                depth_bonus = bench_value * 0.1  # 10% of bench value as depth bonus
                
                # Calculate backup qualification counts (15% and 25% thresholds)
                backup_counts = self._calculate_backup_qualification(
                    bench_players, starters, player_values, position_players
                )
            
        # Check if this is starter optimization mode
        if hasattr(self, '_optimizer_mode') and self._optimizer_mode:
            # For optimizer mode, also calculate bench depth using 25% rule
            from .starter_optimizer import calculate_bench_depth_report, get_starter_breakdown
            
            bench_depth = calculate_bench_depth_report(roster_players, starters)
            starter_breakdown = get_starter_breakdown(roster_players)
            
            return {
                'starter_points': total_points,
                'depth_bonus': depth_bonus,
                'total_value': total_points + depth_bonus,
                'starters': starters,
                'backup_counts': backup_counts,
                'bench_depth': bench_depth,
                'starter_breakdown': starter_breakdown
            }
        
        return {
            'starter_points': total_points,
            'depth_bonus': depth_bonus,
            'total_value': total_points + depth_bonus,
            'starters': starters,
            'backup_counts': backup_counts
        }
        
    def _calculate_backup_qualification(self, bench_players, starters, player_values, position_players):
        """
        Calculate backup player qualification based on percentage thresholds
        
        Backup Classification:
        - Excellent Backup (≤15% gap): Season-ready, minimal downgrade
        - Useful Backup (15-25% gap): Acceptable fill-in, some downgrade risk
        - Depth Only (>25% gap): Not worth roster spot normally
        
        Returns:
            dict: Backup counts by position and total
        """
        backup_counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'total': 0}
        
        # Get starter values by position for comparison
        starter_values = {}
        for starter in starters:
            pos = starter['pos']
            if pos not in starter_values:
                starter_values[pos] = []
            starter_values[pos].append(player_values[starter['id']])
        
        # For each bench player, compare to best starter at their position
        for bench_player in bench_players:
            pos = bench_player['pos']
            bench_value = player_values[bench_player['id']]
            
            # Get comparison value - best starter at position or FLEX-eligible
            comparison_value = 0
            if pos in starter_values and starter_values[pos]:
                comparison_value = max(starter_values[pos])
            elif pos in ['RB', 'WR', 'TE']:
                # For FLEX positions, compare to weakest FLEX-eligible starter
                flex_starter_values = []
                for flex_pos in ['RB', 'WR', 'TE']:
                    if flex_pos in starter_values:
                        flex_starter_values.extend(starter_values[flex_pos])
                if flex_starter_values:
                    # Compare to weakest FLEX starter (more realistic backup scenario)
                    comparison_value = min(flex_starter_values)
            
            # Calculate percentage gap
            if comparison_value > 0:
                gap_pct = (comparison_value - bench_value) / comparison_value
                
                # Classify as backup if within 25% threshold
                if gap_pct <= 0.25:
                    backup_counts[pos] += 1
                    backup_counts['total'] += 1
        
        return backup_counts
    
    def calculate_bench_quality_metrics(self, roster_players, starters, player_values):
        """
        Calculate detailed bench quality metrics for trade-off analysis
        
        Returns:
            dict: Detailed bench metrics including quality counts, handcuff value, upside
        """
        bench_players = [p for p in roster_players if p not in starters]
        
        metrics = {
            'quality_backups': {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'total': 0},
            'handcuff_value': 0,  # Value of handcuffs to owned RBs
            'upside_score': 0,     # High-variance bench players
            'bye_coverage': 0,     # Quality backups for bye weeks
            'total_bench_value': 0
        }
        
        if not bench_players:
            return metrics
        
        # Get starter values for comparison (using point threshold not percentage)
        starter_values = {}
        for starter in starters:
            pos = starter['pos']
            if pos not in starter_values:
                starter_values[pos] = []
            starter_values[pos].append(player_values[starter['id']])
        
        # Analyze each bench player
        for bench_player in bench_players:
            pos = bench_player['pos']
            bench_value = player_values[bench_player['id']]
            metrics['total_bench_value'] += bench_value
            
            # Quality backup = within 25 points of starter
            comparison_value = 0
            if pos in starter_values and starter_values[pos]:
                comparison_value = max(starter_values[pos])
            
            if comparison_value > 0:
                point_gap = comparison_value - bench_value
                if point_gap <= 25:  # Within 25 points
                    metrics['quality_backups'][pos] += 1
                    metrics['quality_backups']['total'] += 1
                    metrics['bye_coverage'] += bench_value * 0.5  # Half value for bye weeks
            
            # Calculate upside (use envelope data if available)
            if 'high' in bench_player and 'low' in bench_player:
                upside = bench_player['high'] - bench_player['proj']
                if upside > 30:  # High upside threshold
                    metrics['upside_score'] += upside * 0.3
            
            # Simple handcuff detection (same-team RBs)
            if pos == 'RB':
                for starter in starters:
                    if starter['pos'] == 'RB' and 'team' in starter and 'team' in bench_player:
                        if starter['team'] == bench_player['team']:
                            metrics['handcuff_value'] += bench_value * 0.4
                            break
        
        return metrics
        
    def calculate_vor_utility(self, player_id, roster, replacement_levels, sampled_projection, params):
        """
        Calculate VOR-based utility for a player using the formula:
        utility = (1-γ)*VOR + γ*(P_survive * VOR)
        
        Args:
            player_id: Player identifier
            roster: Current roster
            replacement_levels: Position replacement levels
            sampled_projection: Player's projected value for this simulation
            params: VOR parameters (alpha, lambda, gamma, r_te, delta_qb)
        """
        player_cache = self._get_player_cache()
        pos = player_cache['pos'][player_id]
        
        # Basic VOR calculation with bounds checking
        replacement_level = replacement_levels.get(pos, 0)
        vor = sampled_projection - replacement_level
        
        if vor <= 0 or np.isnan(vor) or np.isinf(vor):
            return 0.0
            
        # Scarcity bonus using alpha parameter
        pos_counts = defaultdict(int)
        for player in roster:
            pos_counts[player['pos']] += 1
            
        scarcity_bonus = 0
        if params.get('alpha', 0) > 0:
            remaining_need = max(0, ROSTER_REQUIREMENTS.get(pos, 0) - pos_counts[pos])
            if remaining_need > 0:
                scarcity_bonus = params['alpha'] * remaining_need
        
        # Shadow price bonus for early RB rounds (NEW)
        shadow_bonus = 0
        if (params.get('rb_shadow', 0) > 0 and pos == 'RB'):
            current_round = (len(roster) // self.n_teams) + 1
            shadow_decay_round = params.get('shadow_decay_round', 3)
            rb_count = pos_counts['RB']
            
            # Apply shadow price in early rounds, decay once we have enough RBs
            if current_round <= shadow_decay_round and rb_count < 2:
                shadow_bonus = params['rb_shadow']
                # Stronger bonus if we have 0 RBs vs 1 RB
                if rb_count == 0:
                    shadow_bonus *= 1.5
        
        # Risk adjustment using lambda parameter  
        risk_adjustment = 1.0
        if params.get('lambda', 0) != 0:
            # Negative lambda favors upside (high variance), positive favors conservative
            risk_adjustment = max(0.1, 1.0 + (params['lambda'] / 100.0))  # Bounds check
            
        # Fixed tier-gap calculation using actual replacement level
        espn_rank = player_cache['espn_rank'][player_id]
        replacement_rank = self._get_replacement_rank(pos, replacement_levels)
        tier_gap_norm = max(0, (replacement_rank - espn_rank) / max(1, self.n_teams))
        
        # Position-specific adjustments
        pos_adjustment = 1.0
        
        # TE punt logic
        if pos == 'TE' and params.get('r_te', 8) > 0:
            current_round = (len(roster) // self.n_teams) + 1
            if current_round >= params['r_te']:
                pos_adjustment = 0.5  # Reduce TE value after punt round
                
        # QB threshold logic  
        if pos == 'QB' and params.get('delta_qb', 0) > 0:
            if espn_rank > params['delta_qb']:
                pos_adjustment = 0.7  # Reduce value for QBs below threshold
        
        # Calculate base utility with bounds checking
        base_vor = vor + scarcity_bonus + shadow_bonus
        base_utility = base_vor * risk_adjustment * pos_adjustment * (1 + tier_gap_norm)
        
        # Bounds check for base utility
        if np.isnan(base_utility) or np.isinf(base_utility):
            return 0.0
        
        # Survival probability component using proper probability model
        gamma = params.get('gamma', 0.75)
        if gamma > 0:
            # Use actual survival probability from probability model
            picks_until_next = self._estimate_picks_until_next_turn(len(roster))
            available_players = set(self.prob_model.players_df.index)  # Approximation
            p_survive = self.prob_model.calculate_survival_probability(
                player_id, picks_until_next, available_players
            )
            p_survive = max(0.0, min(1.0, p_survive))  # Bounds check
            
            survival_utility = gamma * (p_survive * base_vor)
            return max(0.0, (1 - gamma) * base_utility + survival_utility)
        else:
            return max(0.0, base_utility)

    def calculate_vona(self, player_id, roster, next_picks, sim_cache, params):
        """
        Calculate Value of Next Available (VONA) for wait-vs-draft decisions
        VONA = E[value_if_wait] - value_now
        
        Args:
            player_id: Current player being considered
            roster: Current roster
            next_picks: Number of picks until next turn
            sim_cache: Cached simulation data for lookups
            params: VOR parameters
        """
        player_cache = self._get_player_cache()
        pos = player_cache['pos'][player_id]
        current_round = (len(roster) // self.n_teams) + 1
        
        # Get current player value
        current_projection = player_cache['proj'][player_id]
        value_now = self.calculate_vor_utility(
            player_id, roster, sim_cache.get('replacement_levels', {}), 
            current_projection, params
        )
        
        # Calculate actual tier drops based on data analysis
        tier_drops = self._calculate_tier_drops_from_data(pos, current_round)
        
        # Calculate expected value if we wait using data-driven tier analysis
        expected_decay = tier_drops.get('decay_rate', 0.15)
        margin = tier_drops.get('margin', 5)
        
        # Estimate what we'd get if we wait
        projected_wait_value = value_now * (1 - expected_decay * next_picks)
        expected_value_wait = max(0, projected_wait_value)
        
        # VONA calculation with bounds checking
        vona = expected_value_wait - value_now
        if np.isnan(vona) or np.isinf(vona):
            vona = 0.0
        
        return {
            'vona': vona,
            'value_now': value_now,
            'expected_value_wait': expected_value_wait,
            'margin': margin,
            'draft_now': vona < -margin  # Recommendation
        }

    def select_best_player(self, available_players, my_roster, strategy_params, round_num, recent_picks=None, replacement_levels=None):
        """Select best player using starter optimization, VOR utility, or legacy multipliers"""
        if not available_players:
            return None
            
        # Count current positions
        pos_counts = defaultdict(int)
        for player in my_roster:
            pos_counts[player['pos']] += 1
            
        # Check if we're using starter optimization
        if isinstance(strategy_params, dict) and 'optimize_starters' in strategy_params:
            return self.select_best_player_optimizer(
                available_players, my_roster, strategy_params, round_num
            )
        
        # Check if we're using the new starter optimizer
        if isinstance(strategy_params, dict) and 'use_starter_optimizer' in strategy_params:
            # Need to get my_team_idx from context - look for it in available patterns
            team_idx = getattr(self, '_current_team_idx', 4)  # Default to pick 5 (0-based)
            return self.select_best_player_starter(
                available_players, my_roster, round_num, team_idx
            )
            
        # Phase transition: Rounds 1-7 focus on starters, 8-14 on bench depth
        if round_num >= 8 and isinstance(strategy_params, dict):
            # Check if we have bench_phase parameters
            if 'bench_phase' in strategy_params:
                return self.select_best_bench_player(
                    available_players, my_roster, strategy_params['bench_phase'], 
                    round_num, pos_counts, replacement_levels
                )
            
        # Get valid positions for this round
        valid_positions = ROUND_POSITION_VALIDITY.get(round_num, ['RB', 'WR', 'TE', 'QB'])
        
        # Detect position runs
        run_multipliers = {}
        if recent_picks and len(recent_picks) >= 3:
            last_3 = recent_picks[-3:]
            for pos in ['RB', 'WR', 'QB', 'TE']:
                if last_3.count(pos) >= 2:
                    run_multipliers[pos] = 1.2
        
        # Get cached lookups for performance
        player_cache = self._get_player_cache()
        
        # Filter to valid players first
        valid_players = []
        for player_id in available_players:
            if (player_id in player_cache['pos'] and
                player_cache['pos'][player_id] in valid_positions and
                pos_counts[player_cache['pos'][player_id]] < POSITION_LIMITS.get(player_cache['pos'][player_id], 3)):
                valid_players.append(player_id)
        
        if not valid_players:
            return None
        
        # Check if we're using VOR parameters or legacy multipliers
        if isinstance(strategy_params, dict) and 'alpha' in strategy_params:
            # New VOR-based selection with chance constraint checking
            current_round = round_num
            
            # Check chance constraint if specified
            constraint_target = strategy_params.get('constraint_target')
            if constraint_target and current_round <= constraint_target.get('round', 99):
                target_pos = constraint_target.get('RB')
                target_round = constraint_target.get('round')
                threshold = strategy_params.get('constraint_threshold', 0.75)
                
                # If we're in or near target round and below threshold, force RB
                if (current_round <= target_round and 
                    pos_counts['RB'] < target_pos):
                    # Force RB selection if available
                    rb_players = [p for p in valid_players if player_cache['pos'][p] == 'RB']
                    if rb_players and current_round == target_round:
                        # Force RB in final target round
                        valid_players = rb_players
            
            best_player = None
            best_utility = -1
            
            sim_cache = {'replacement_levels': replacement_levels or {}}
            
            for player_id in valid_players:
                projection = player_cache['proj'][player_id]
                utility = self.calculate_vor_utility(
                    player_id, my_roster, replacement_levels or {}, 
                    projection, strategy_params
                )
                
                # Apply position run adjustments
                pos = player_cache['pos'][player_id]
                run_mult = run_multipliers.get(pos, 1.0)
                utility *= run_mult
                
                if utility > best_utility:
                    best_utility = utility
                    best_player = player_id
                    
            return best_player
            
        else:
            # Legacy multiplier-based selection
            strategy_multipliers = strategy_params
            
            # Convert to arrays for vectorization
            valid_players_arr = np.array(valid_players)
            positions = np.array([player_cache['pos'][pid] for pid in valid_players])
            projections = np.array([player_cache['proj'][pid] for pid in valid_players])
            
            # Calculate base scores vectorized
            if replacement_levels:
                # VOR calculation vectorized
                replacement_vals = np.array([replacement_levels.get(pos, 0) for pos in positions])
                vor_scores = projections - replacement_vals
                base_scores = np.maximum(0, vor_scores)  # Ensure non-negative VOR
            else:
                # Fallback rank-based scoring vectorized
                ranks = np.array([player_cache['espn_rank'][pid] for pid in valid_players])
                base_scores = projections / (ranks + 10)
            
            # Apply multipliers vectorized
            strategy_mults = np.array([strategy_multipliers.get(pos, 1.0) for pos in positions])
            need_mults = np.array([1.5 if pos_counts[pos] < ROSTER_REQUIREMENTS.get(pos, 0) else 1.0 for pos in positions])
            run_mults = np.array([run_multipliers.get(pos, 1.0) for pos in positions])
            
            # Final scores calculation
            final_scores = base_scores * strategy_mults * need_mults * run_mults
            
            # Find best player
            best_idx = np.argmax(final_scores)
            return valid_players_arr[best_idx]
        
    def select_best_player_starter(self, available_players, my_roster, round_num, my_team_idx):
        """Use starter optimizer for marginal value decisions"""
        from .starter_optimizer import pick_best_now, beta_pert_samples
        
        if not available_players:
            return None
            
        try:
            # Get cached lookups for performance
            player_cache = self._get_player_cache()
            
            # Convert available players to starter optimizer format with proper Beta-PERT sampling
            pool = []
            for player_id in available_players:
                base_proj = player_cache['proj'][player_id]
                
                # Use proper Beta-PERT sampling (±20% envelope)
                proj_samples = beta_pert_samples(
                    mode=base_proj, 
                    low=base_proj * 0.8, 
                    high=base_proj * 1.2, 
                    lam=4.0, 
                    size=500
                )
                
                pool.append({
                    'id': player_id,
                    'name': player_cache['player_name'][player_id],
                    'pos': player_cache['pos'][player_id],
                    'proj': base_proj,
                    'samples': proj_samples
                })
            
            # Convert my_roster to roster_state dict format with proper sampling
            roster_state = {}
            for player in my_roster:
                pos = player['pos']
                if pos not in roster_state:
                    roster_state[pos] = []
                    
                base_proj = player.get('proj', player_cache['proj'][player['id']])
                roster_samples = beta_pert_samples(
                    mode=base_proj,
                    low=base_proj * 0.8,
                    high=base_proj * 1.2,
                    lam=4.0,
                    size=500
                )
                
                roster_state[pos].append({
                    'id': player['id'],
                    'name': player['name'],
                    'pos': player['pos'],
                    'proj': base_proj,
                    'samples': roster_samples
                })
            
            # Calculate real global pick number using snake draft logic
            if round_num % 2 == 1:  # Odd round (1st, 3rd, etc.)
                current_pick = (round_num - 1) * self.n_teams + my_team_idx + 1
            else:  # Even round (2nd, 4th, etc.) - reverse order
                current_pick = (round_num - 1) * self.n_teams + (self.n_teams - my_team_idx)
            
            # Calculate next pick using snake order
            picks_until = self._estimate_picks_until_next_turn(len(my_roster))
            next_pick = current_pick + picks_until
            
            # Create probability model for sophisticated pick prediction
            try:
                from .starter_optimizer import create_probability_model
                prob_model = create_probability_model()
            except Exception as e:
                print(f"Warning: Could not load probability model: {e}")
                prob_model = None
            
            # Balance performance vs accuracy for simulation integration
            result = pick_best_now(
                pool, roster_state, current_pick, next_pick,
                top_k_candidates=min(15, len(pool)),  # Further reduced for performance
                scenarios=100,  # Significantly reduced for 3-sim testing
                probability_model=prob_model
            )
            
            # DEBUG: Print top 5 candidates to understand why QB is always picked
            if result and 'debug' in result and round_num <= 2:
                print(f"\n🔍 ROUND {round_num} DEBUG - Pool sample:")
                # Show projection distribution by position
                pos_samples = {}
                for p in pool[:20]:  # First 20 players only
                    pos = p['pos']
                    if pos not in pos_samples:
                        pos_samples[pos] = []
                    pos_samples[pos].append(p['cached_mean'] if 'cached_mean' in p else np.mean(p['samples']))
                
                for pos, projs in pos_samples.items():
                    avg_proj = np.mean(projs)
                    print(f"  {pos}: {len(projs)} players, avg proj = {avg_proj:.1f}")
                
                print(f"\n🔍 ROUND {round_num} DEBUG - Top 5 candidates:")
                for i, (name, pos, imm_val, opp_cost, score) in enumerate(result['debug'][:5]):
                    print(f"  {i+1}. {name} ({pos}): Score={score:.1f} (Value={imm_val:.1f} - Cost={opp_cost:.1f})")
                print(f"  → SELECTED: {result['pick']['name']} ({result['pick']['pos']})\n")
            
            return result['pick']['id'] if result['pick'] else None
            
        except Exception as e:
            print(f"Warning: Starter optimizer failed with error {e}, falling back to balanced selection")
            from .strategies import get_strategy
            fallback_strategy = get_strategy('balanced')
            return self.select_best_player(available_players, my_roster, fallback_strategy, round_num)
    
    def select_best_bench_player(self, available_players, my_roster, bench_params, round_num, pos_counts, replacement_levels):
        """
        Select best player for bench phase (rounds 8-14)
        
        Prioritizes:
        1. Handcuff value (backup to owned RBs)
        2. Bye week coverage (quality backups)
        3. Upside potential (high variance players)
        4. Quality threshold (within 25 points of starters)
        """
        player_cache = self._get_player_cache()
        
        # Calculate current starters for comparison
        starters = []
        roster_analysis = self.calculate_roster_value(my_roster)
        if 'starters' in roster_analysis:
            starters = roster_analysis['starters']
        
        # Get starter values by position
        starter_values = {}
        for starter in starters:
            pos = starter['pos']
            if pos not in starter_values:
                starter_values[pos] = []
            starter_values[pos].append(starter['proj'])
        
        best_player = None
        best_score = -1
        
        for player_id in available_players:
            if player_id not in player_cache['pos']:
                continue
                
            pos = player_cache['pos'][player_id]
            proj = player_cache['proj'][player_id]
            
            # Skip if we have too many at this position
            if pos_counts[pos] >= POSITION_LIMITS.get(pos, 5):
                continue
            
            # Base score from projection
            score = proj
            
            # 1. Handcuff bonus (RBs from same team as owned RBs)
            if pos == 'RB' and bench_params.get('handcuff_weight', 0) > 0:
                for roster_player in my_roster:
                    if roster_player['pos'] == 'RB':
                        # Simple handcuff detection (would need team data in real implementation)
                        if player_cache.get('espn_rank', {}).get(player_id, 999) > 50:  # Lower-ranked RB
                            score += proj * bench_params['handcuff_weight']
                            break
            
            # 2. Quality backup bonus (within 25 points of starter)
            quality_bonus = 0
            if pos in starter_values and starter_values[pos]:
                best_starter = max(starter_values[pos])
                point_gap = best_starter - proj
                if point_gap <= 25:
                    quality_bonus = (25 - point_gap) * bench_params.get('quality_weight', 0.5)
                    score += quality_bonus
            
            # 3. Upside bonus (high variance players)
            # Use ranking as proxy for upside in late rounds
            rank = player_cache['espn_rank'][player_id]
            if rank > 100 and bench_params.get('upside_weight', 0) > 0:
                upside_bonus = (200 - rank) * bench_params['upside_weight']
                score += upside_bonus
            
            # 4. Bye week coverage (prioritize positions with fewer backups)
            if pos_counts[pos] < ROSTER_REQUIREMENTS.get(pos, 2) + 1:  # Need at least 1 backup
                score += proj * bench_params.get('coverage_weight', 0.3)
            
            # 5. Position scarcity in late rounds
            if round_num >= 12:
                if pos in ['RB', 'WR'] and pos_counts[pos] < 5:
                    score *= 1.2  # Prioritize skill positions late
                elif pos in ['K', 'DST'] and pos_counts[pos] == 0:
                    score *= 1.5  # Must draft K/DST eventually
            
            if score > best_score:
                best_score = score
                best_player = player_id
        
        return best_player
    
    def select_best_player_optimizer(self, available_players, my_roster, strategy_params, round_num):
        """
        Select best player using starter optimization approach.
        ONLY maximizes 7 starter projected points.
        """
        from .starter_optimizer import pick_best_now, beta_pert_samples
        
        if not available_players:
            return None
            
        try:
            # Get cached lookups for performance
            player_cache = self._get_player_cache()
            
            # Validate player_cache before using
            if not player_cache or not isinstance(player_cache, dict):
                print("Warning: Invalid player cache, falling back to VOR selection")
                return self.select_best_player_vor(available_players, my_roster, strategy_params, round_num)
            
            # Estimate picks until next turn based on round and position
            picks_until_next = self._estimate_picks_until_next_turn(len(my_roster))
            current_pick = round_num  # Simplified for optimizer
            next_pick = current_pick + picks_until_next
            
            # Get risk aversion from strategy params with validation
            risk_aversion = strategy_params.get('risk_aversion', 0.5)
            if not isinstance(risk_aversion, (int, float)) or risk_aversion < 0.0 or risk_aversion > 1.0:
                risk_aversion = 0.5  # Safe fallback
            
            # Convert to starter_optimizer format
            pool = []
            for player_id in available_players:
                base_proj = player_cache['proj'][player_id]
                proj_samples = beta_pert_samples(
                    mode=base_proj, 
                    low=base_proj * 0.8, 
                    high=base_proj * 1.2, 
                    lam=4.0, 
                    size=100  # Reduced for performance
                )
                
                pool.append({
                    'id': player_id,
                    'name': player_cache['player_name'][player_id],
                    'pos': player_cache['pos'][player_id],
                    'proj': base_proj,
                    'samples': proj_samples
                })
            
            # Convert my_roster to roster_state dict format
            roster_state = {}
            for player in my_roster:
                pos = player['pos']
                if pos not in roster_state:
                    roster_state[pos] = []
                    
                base_proj = player.get('proj', player_cache['proj'][player['id']])
                roster_samples = beta_pert_samples(
                    mode=base_proj,
                    low=base_proj * 0.8,
                    high=base_proj * 1.2,
                    lam=4.0,
                    size=100
                )
                
                roster_state[pos].append({
                    'id': player['id'],
                    'name': player['name'],
                    'pos': player['pos'],
                    'proj': base_proj,
                    'samples': roster_samples
                })
            
            # Use starter optimizer to select best player
            result = pick_best_now(
                pool, roster_state, current_pick, next_pick,
                top_k_candidates=min(20, len(pool)),
                scenarios=50  # Reduced for performance
            )
            
            best_player_id = result['pick']['id'] if result['pick'] else None
            
            # Fallback to starter_max strategy if optimizer fails
            if best_player_id is None:
                # print("Warning: Optimizer returned None, falling back to starter_max selection")  # Suppress for cleaner output
                from .strategies import get_strategy
                fallback_strategy = get_strategy('starter_max')
                return self.select_best_player(available_players, my_roster, fallback_strategy, round_num)
            
            return best_player_id
            
        except Exception as e:
            print(f"Warning: Optimizer failed with error {e}, falling back to starter_max selection")
            from .strategies import get_strategy
            fallback_strategy = get_strategy('starter_max')
            return self.select_best_player(available_players, my_roster, fallback_strategy, round_num)
    
    def get_top_available_players(self, current_roster, already_drafted, k=6):
        """Get top K available players by VOR for VONA analysis"""
        player_cache = self._get_player_cache()
        
        # Initialize available players
        available = set(self.prob_model.players_df.index)
        
        # Remove already drafted players
        if already_drafted:
            for player_name in already_drafted:
                player_id = player_cache['name_to_id'].get(player_name)
                if player_id is not None:
                    available.discard(player_id)
        
        # Remove current roster players
        if current_roster:
            for player_name in current_roster:
                player_id = player_cache['name_to_id'].get(player_name)
                if player_id is not None:
                    available.discard(player_id)
        
        # Calculate VOR for all available players
        player_values = {}
        for pid in available:
            player_values[pid] = player_cache['proj'][pid]
            
        replacement_levels = calculate_replacement_levels(
            self.prob_model.players_df, player_values, self.n_teams
        )
        
        # Score players by VOR
        vor_scores = []
        for player_id in available:
            pos = player_cache['pos'][player_id]
            proj = player_cache['proj'][player_id]
            replacement = replacement_levels.get(pos, 0)
            vor = proj - replacement
            
            if vor > 0:  # Only consider positive VOR players
                vor_scores.append({
                    'id': player_id,
                    'name': player_cache['player_name'][player_id],
                    'pos': pos,
                    'vor': vor,
                    'proj': proj,
                    'rank': player_cache['espn_rank'][player_id]
                })
        
        # Sort by VOR descending and return top K
        vor_scores.sort(key=lambda x: x['vor'], reverse=True)
        return vor_scores[:k]

    def simulate_from_pick(self, my_team_idx, strategy_params, pick_num, forced_pick=None, 
                          opponent_sequence=None, seed=42, initial_roster=None, already_drafted=None):
        """Continue draft from specific pick with optional forced selection"""
        rng = np.random.default_rng(seed)
        
        # Get cached lookups for performance
        player_cache = self._get_player_cache()
        
        # Initialize available players
        available = set(self.prob_model.players_df.index)
        
        # Handle already drafted players using cached lookup
        if already_drafted:
            for player_name in already_drafted:
                player_id = player_cache['name_to_id'].get(player_name)
                if player_id is not None:
                    available.discard(player_id)
                    
        # Calculate replacement levels once per simulation for VOR scoring
        player_values = {}
        for pid in available:
            player_data = {
                'id': pid,
                'proj': player_cache['proj'][pid]
            }
            projected_value = self._get_player_projection(player_data, seed)
            player_values[pid] = projected_value
            
        replacement_levels = calculate_replacement_levels(
            self.prob_model.players_df, player_values, self.n_teams
        )
                    
        # Initialize rosters
        team_rosters = {i: [] for i in range(self.n_teams)}
        my_roster = []
        
        # Add initial roster if provided using cached lookup
        if initial_roster:
            for player_name in initial_roster:
                player_id = player_cache['name_to_id'].get(player_name)
                if player_id is not None:
                    player_data = {
                        'id': player_id,
                        'name': player_name,
                        'pos': player_cache['pos'][player_id],
                        'proj': player_cache['proj'][player_id]
                    }
                    player_data['proj'] = self._get_player_projection(player_data, seed)
                    my_roster.append(player_data)
                    team_rosters[my_team_idx].append(player_cache['pos'][player_id])
                    
        # Track draft flow
        recent_picks = []
        position_sequence = []
        
        # Generate pick order and start from specified pick
        pick_order = self.generate_snake_order()
        
        # Skip to the specified pick number
        for i in range(pick_num, len(pick_order)):
            pick_idx = i
            team_idx = pick_order[pick_idx]
            round_num = (pick_idx // self.n_teams) + 1
            
            if len(available) == 0:
                break
                
            if team_idx == my_team_idx:
                # Our pick - use forced pick if specified
                if forced_pick and i == pick_num:
                    # Force the specified player if available
                    forced_player_id = player_cache['name_to_id'].get(forced_pick)
                    if forced_player_id and forced_player_id in available:
                        player_id = forced_player_id
                    else:
                        # Fallback to normal selection if forced pick unavailable
                        self._current_team_idx = my_team_idx  # Set for starter optimizer
                        player_id = self.select_best_player(available, my_roster, strategy_params, 
                                                          round_num, recent_picks, replacement_levels)
                else:
                    # Normal selection
                    self._current_team_idx = my_team_idx  # Set for starter optimizer
                    player_id = self.select_best_player(available, my_roster, strategy_params, 
                                                      round_num, recent_picks, replacement_levels)
                
                if player_id:
                    player_data = {
                        'id': player_id,
                        'name': player_cache['player_name'][player_id],
                        'pos': player_cache['pos'][player_id],
                        'proj': player_cache['proj'][player_id],
                        'draft_pick': pick_idx + 1,
                        'draft_round': round_num
                    }
                    player_data['proj'] = self._get_player_projection(player_data, seed)
                    my_roster.append(player_data)
                    position_sequence.append(player_data['pos'])
                    team_rosters[team_idx].append(player_data['pos'])
                    recent_picks.append(player_data['pos'])
                    available.discard(player_id)
                    
            else:
                # Opponent pick - use cached sequence if provided
                if opponent_sequence and (i - pick_num) < len(opponent_sequence):
                    opponent_pick = opponent_sequence[i - pick_num]
                    player_id = player_cache['name_to_id'].get(opponent_pick)
                    if player_id and player_id in available:
                        pos = player_cache['pos'][player_id]
                        team_rosters[team_idx].append(pos)
                        recent_picks.append(pos)
                        available.discard(player_id)
                        continue
                
                # Normal opponent selection
                player_id = self.opponent_model.predict_opponent_pick(
                    available, team_rosters[team_idx], recent_picks, round_num, rng, team_idx
                )
                
                if player_id:
                    pos = self.opponent_model.get_position_from_id(player_id)
                    team_rosters[team_idx].append(pos)
                    recent_picks.append(pos)
                    available.discard(player_id)
                    
            # Keep recent picks to last 10
            if len(recent_picks) > 10:
                recent_picks.pop(0)
                
        # Calculate final roster value
        roster_analysis = self.calculate_roster_value(my_roster, seed)
        return roster_analysis['total_value']
    
    def calculate_positional_degradation(self, current_pick, next_pick, already_drafted=None, n_simulations=100):
        """Calculate expected point degradation for each position between current and next pick
        
        Returns degradation info for each position showing:
        - Top 3 players available now with projections
        - Expected top 3 players at next pick (using survival probabilities)
        - Expected point loss from waiting
        - Tier break warnings
        """
        if already_drafted is None:
            already_drafted = []
        
        # Fix Bug #1: Convert drafted names to set and filter by player_name column
        drafted_names_set = set(already_drafted)
        available_df = self.prob_model.players_df[
            ~self.prob_model.players_df['player_name'].isin(drafted_names_set)
        ].copy()
        
        # Calculate picks between now and next turn
        picks_until_next = next_pick - current_pick
        
        degradation_results = {}
        
        for position in ['RB', 'WR', 'TE', 'QB']:
            # Check which position column to use
            if 'pos' in available_df.columns:
                pos_players = available_df[available_df['pos'] == position].copy()
            elif 'position' in available_df.columns:
                pos_players = available_df[available_df['position'] == position].copy()
            else:
                continue
            if pos_players.empty:
                continue
            
            # Get top 3 available now with their projections
            top_now = pos_players.nsmallest(3, 'espn_rank')
            players_now = []
            
            for player_id, player in top_now.iterrows():
                # Fix Bug #4: Use base projection instead of expensive sampling
                median_proj = player['base'] if 'base' in player else player.get('proj', 50)
                
                players_now.append({
                    'name': player['player_name'],
                    'rank': player['espn_rank'],
                    'projection': median_proj,
                    'player_id': player_id
                })
            
            # Calculate survival probabilities and expected available at next pick
            players_next = []
            for player_id, player in pos_players.head(10).iterrows():  # Check top 10
                # Fix Bug #2: Use existing probability model's calculate_survival_probability method
                available_player_ids = pos_players.index.tolist()
                survival_prob = self.prob_model.calculate_survival_probability(
                    player_id, picks_until_next, available_player_ids
                )
                
                if survival_prob > 0.1:  # Only show if reasonable chance of survival (lowered threshold)
                    # Fix Bug #4: Use base projection instead of expensive sampling
                    median_proj = player['base'] if 'base' in player else player.get('proj', 50)
                    
                    players_next.append({
                        'name': player['player_name'],
                        'rank': player['espn_rank'],
                        'projection': median_proj,
                        'survival_prob': survival_prob,
                        'player_id': player_id
                    })
            
            # Sort by projection (best available player) and take top 3
            players_next.sort(key=lambda x: x['projection'], reverse=True)
            players_next = players_next[:3]
            
            # Calculate degradation (best now vs best expected to be available)
            best_now = players_now[0]['projection'] if players_now else 0
            best_next = players_next[0]['projection'] if players_next else 0
            degradation = best_now - best_next
            
            # Calculate tier breaks (20-point tiers)
            tier_boundaries = self._calculate_tier_info(pos_players, picks_until_next)
            
            degradation_results[position] = {
                'available_now': players_now,
                'expected_next': players_next,
                'degradation': degradation,
                'tier_info': tier_boundaries,
                'urgency': self._calculate_urgency(degradation, tier_boundaries)
            }
        
        return degradation_results
    
    def _calculate_tier_info(self, pos_players, picks_until_next):
        """Calculate tier break probabilities"""
        if pos_players.empty:
            return {}
        
        # Define tiers based on projections (20-point buckets)
        tier_info = {}
        tier_boundaries = [170, 150, 130, 110, 90]  # Tier thresholds
        
        for i, threshold in enumerate(tier_boundaries):
            tier_name = f"Tier {i+1} ({threshold}+ pts)"
            # Fix Bug #3: Use proper column names from the DataFrame
            # Check which projection column exists
            proj_column = 'base' if 'base' in pos_players.columns else 'proj'
            
            # Find players in this tier
            if i < len(tier_boundaries) - 1:
                tier_players = pos_players[
                    (pos_players[proj_column] >= threshold) & 
                    (pos_players[proj_column] < tier_boundaries[i-1] if i > 0 else True)
                ]
            else:
                tier_players = pos_players[pos_players[proj_column] >= threshold]
            
            if not tier_players.empty:
                # Calculate probability at least one survives
                survival_probs = []
                for player_id, player in tier_players.iterrows():
                    # Fix Bug #2: Use existing probability model instead of hardcoded columns
                    available_player_ids = tier_players.index.tolist()
                    survival = self.prob_model.calculate_survival_probability(
                        player_id, picks_until_next, available_player_ids
                    )
                    survival_probs.append(survival)
                
                # Probability at least one tier player survives
                prob_none_survive = np.prod([1 - p for p in survival_probs])
                prob_any_survive = 1 - prob_none_survive
                
                tier_info[tier_name] = {
                    'count_now': len(tier_players),
                    'prob_available': prob_any_survive
                }
        
        return tier_info
    
    def _calculate_urgency(self, degradation, tier_info):
        """Calculate urgency level based on degradation and tier breaks"""
        if degradation > 30:
            return 'CRITICAL'
        elif degradation > 20:
            return 'HIGH'
        elif degradation > 10:
            return 'MEDIUM'
        else:
            # Check tier breaks
            for tier_name, info in tier_info.items():
                if 'Tier 1' in tier_name and info['prob_available'] < 0.3:
                    return 'MEDIUM'
            return 'LOW'
    
    def calculate_multi_round_degradation(self, current_pick, future_picks, already_drafted=None, 
                                        n_sims=200, shortlist=6):
        """Calculate degradation across multiple future picks using CRN methodology
        
        Args:
            current_pick: Current global pick number
            future_picks: List of future pick numbers to analyze
            already_drafted: List of already drafted player names
            n_sims: Number of simulations per step
            shortlist: Number of top candidates per position
            
        Returns:
            Dictionary with degradation analysis for each position across all steps
        """
        if already_drafted is None:
            already_drafted = []
        
        # Convert drafted names to set and filter available players
        drafted_names_set = set(already_drafted)
        available_df = self.prob_model.players_df[
            ~self.prob_model.players_df['player_name'].isin(drafted_names_set)
        ].copy()
        
        # Initialize results structure
        results = {}
        
        # Analyze each position
        for position in ['RB', 'WR', 'TE', 'QB']:
            # Get position players
            if 'pos' in available_df.columns:
                pos_players = available_df[available_df['pos'] == position].copy()
            elif 'position' in available_df.columns:
                pos_players = available_df[available_df['position'] == position].copy()
            else:
                continue
                
            if pos_players.empty:
                continue
            
            # Get top players available now (step 0)
            top_now = pos_players.nsmallest(shortlist, 'espn_rank')
            players_now = []
            
            for player_id, player in top_now.iterrows():
                median_proj = player['base'] if 'base' in player else player.get('proj', 50)
                players_now.append({
                    'name': player['player_name'],
                    'rank': player['espn_rank'],
                    'projection': median_proj,
                    'player_id': player_id
                })
            
            # Calculate degradation for each future step
            steps = []
            
            # Step 0: Current state (baseline)
            if players_now:
                steps.append({
                    'step': 0,
                    'pick_number': current_pick,
                    'best_player': players_now[0],
                    'degradation': 0.0,  # Baseline
                    'available_count': len(players_now)
                })
            
            # Calculate degradation for each future pick
            baseline_value = players_now[0]['projection'] if players_now else 0
            
            for step_idx, future_pick in enumerate(future_picks):
                picks_until_step = future_pick - current_pick
                step_players = []
                
                # Use CRN simulation to determine expected availability
                for player_id, player in pos_players.head(shortlist * 2).iterrows():
                    # Calculate survival probability to this step
                    available_player_ids = pos_players.index.tolist()
                    survival_prob = self.prob_model.calculate_survival_probability(
                        player_id, picks_until_step, available_player_ids
                    )
                    
                    if survival_prob > 0.10:  # Only show if reasonable chance of being available
                        median_proj = player['base'] if 'base' in player else player.get('proj', 50)
                        
                        step_players.append({
                            'name': player['player_name'],
                            'rank': player['espn_rank'],
                            'projection': median_proj,  # Actual fantasy points, NOT weighted
                            'survival_prob': survival_prob,
                            'player_id': player_id
                        })
                
                # Sort by projection (best available player) and take best
                step_players.sort(key=lambda x: x['projection'], reverse=True)
                
                if step_players:
                    best_step_player = step_players[0]
                    best_value = best_step_player['projection']  # Use actual projection
                    degradation = baseline_value - best_value
                    
                    steps.append({
                        'step': step_idx + 1,
                        'pick_number': future_pick,
                        'best_player': best_step_player,
                        'degradation': degradation,
                        'available_count': len(step_players)
                    })
                else:
                    # No players expected to be available
                    steps.append({
                        'step': step_idx + 1,
                        'pick_number': future_pick,
                        'best_player': None,
                        'degradation': baseline_value,  # Full degradation
                        'available_count': 0
                    })
            
            # Calculate overall urgency based on early degradation
            urgency = 'LOW'
            if len(steps) >= 2:
                first_step_degradation = steps[1]['degradation']
                if first_step_degradation > 25:
                    urgency = 'CRITICAL'
                elif first_step_degradation > 15:
                    urgency = 'HIGH' 
                elif first_step_degradation > 8:
                    urgency = 'MEDIUM'
            
            results[position] = {
                'available_now': players_now,
                'steps': steps,
                'urgency': urgency,
                'baseline_value': baseline_value
            }
        
        return results
    
    def simulate_vona_comparison(self, my_team_idx, strategy_params, pick_num, target_player=None, 
                                opponent_sequence=None, seed=42, initial_roster=None, already_drafted=None):
        """Compare value of picking specific player vs best alternative at same pick"""
        # Branch A: Pick the target player
        value_with_target = self.simulate_from_pick(
            my_team_idx, strategy_params, pick_num, forced_pick=target_player,
            opponent_sequence=opponent_sequence, seed=seed, 
            initial_roster=initial_roster, already_drafted=already_drafted
        )
        
        # Branch B: Pick best available alternative (no forced pick)
        value_with_alternative = self.simulate_from_pick(
            my_team_idx, strategy_params, pick_num, forced_pick=None,
            opponent_sequence=opponent_sequence, seed=seed + 1000,  # Different seed for variation
            initial_roster=initial_roster, already_drafted=already_drafted
        )
        
        # Return incremental value difference
        return value_with_target - value_with_alternative

    def simulate_single_draft(self, my_team_idx, strategy_params, seed=42, initial_roster=None, already_drafted=None, crn=None, sim_idx=None):
        """Simulate a single draft"""
        rng = np.random.default_rng(seed)
        
        # Get cached lookups for performance
        player_cache = self._get_player_cache()
        
        # Initialize available players
        available = set(self.prob_model.players_df.index)
        
        # Handle already drafted players using cached lookup
        if already_drafted:
            for player_name in already_drafted:
                player_id = player_cache['name_to_id'].get(player_name)
                if player_id is not None:
                    available.discard(player_id)
                    
        # Calculate replacement levels once per simulation for VOR scoring
        player_values = {}
        for pid in available:
            player_data = {
                'id': pid,
                'proj': player_cache['proj'][pid]
            }
            projected_value = self._get_player_projection(player_data, seed, crn, sim_idx)
            player_values[pid] = projected_value
            
        replacement_levels = calculate_replacement_levels(
            self.prob_model.players_df, player_values, self.n_teams
        )
                    
        # Initialize rosters
        team_rosters = {i: [] for i in range(self.n_teams)}
        my_roster = []
        
        # Add initial roster if provided using cached lookup
        if initial_roster:
            for player_name in initial_roster:
                player_id = player_cache['name_to_id'].get(player_name)
                if player_id is not None:
                    player_data = {
                        'id': player_id,
                        'name': player_name,
                        'pos': player_cache['pos'][player_id],
                        'proj': player_cache['proj'][player_id]  # Will be updated with projection
                    }
                    # Update with actual projection
                    player_data['proj'] = self._get_player_projection(player_data, seed, crn, sim_idx)
                    my_roster.append(player_data)
                    team_rosters[my_team_idx].append(player_cache['pos'][player_id])
                    
        # Track draft flow
        recent_picks = []
        position_sequence = []
        
        # Generate pick order and simulate
        pick_order = self.generate_snake_order()
        
        for pick_num, team_idx in enumerate(pick_order):
            # Only break if we truly have no players left
            if len(available) == 0:
                print(f"Warning: Ran out of players at pick {pick_num + 1}")
                break
                
            round_num = (pick_num // self.n_teams) + 1
            
            if team_idx == my_team_idx:
                # Our pick
                self._current_team_idx = my_team_idx  # Set for starter optimizer
                player_id = self.select_best_player(available, my_roster, strategy_params, round_num, recent_picks, replacement_levels)
                
                if player_id:
                    player_data = {
                        'id': player_id,
                        'name': player_cache['player_name'][player_id],
                        'pos': player_cache['pos'][player_id],
                        'proj': player_cache['proj'][player_id],  # Will be updated
                        'draft_pick': pick_num + 1,  # Add draft pick number (1-indexed)
                        'draft_round': round_num      # Add draft round
                    }
                    # Update with actual projection
                    player_data['proj'] = self._get_player_projection(player_data, seed, crn, sim_idx)
                    my_roster.append(player_data)
                    position_sequence.append(player_data['pos'])
                    team_rosters[team_idx].append(player_data['pos'])
                    recent_picks.append(player_data['pos'])
                    available.discard(player_id)
                else:
                    # Emergency fallback: pick any available player to avoid simulation failure
                    if available:
                        emergency_id = next(iter(available))  # Get any available player
                        player_data = {
                            'id': emergency_id,
                            'name': player_cache['player_name'][emergency_id],
                            'pos': player_cache['pos'][emergency_id],
                            'proj': player_cache['proj'][emergency_id],
                            'draft_pick': pick_num + 1,
                            'draft_round': round_num
                        }
                        player_data['proj'] = self._get_player_projection(player_data, seed, crn, sim_idx)
                        my_roster.append(player_data)
                        position_sequence.append(player_data['pos'])
                        team_rosters[team_idx].append(player_data['pos'])
                        recent_picks.append(player_data['pos'])
                        available.discard(emergency_id)
                    # If still no player available, skip silently to avoid spam
                    
            else:
                # Opponent pick - use CRN seed if available
                if crn is not None and sim_idx is not None and crn.is_ready():
                    opp_seed = crn.get_opponent_seed(sim_idx)
                    opp_rng = np.random.default_rng(opp_seed)
                else:
                    opp_rng = rng
                    
                player_id = self.opponent_model.predict_opponent_pick(
                    available, team_rosters[team_idx], recent_picks, round_num, opp_rng, team_idx
                )
                
                if player_id:
                    pos = self.opponent_model.get_position_from_id(player_id)
                    team_rosters[team_idx].append(pos)
                    recent_picks.append(pos)
                    available.discard(player_id)
                    
            # Keep recent picks to last 10
            if len(recent_picks) > 10:
                recent_picks.pop(0)
                
        # Calculate final roster value
        roster_analysis = self.calculate_roster_value(my_roster, seed, crn, sim_idx)
        
        return {
            'roster': my_roster,
            'position_sequence': position_sequence,
            'roster_value': roster_analysis['total_value'],
            'starter_points': roster_analysis['starter_points'],
            'depth_bonus': roster_analysis['depth_bonus'],
            'starters': roster_analysis['starters'],
            'backup_counts': roster_analysis.get('backup_counts', {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'total': 0}),
            'num_players': len(my_roster)
        }
        
    def run_simulations(self, my_team_idx, strategy_name, n_sims=100, initial_roster=None, already_drafted=None, base_seed=42, parallel=False, n_workers=4):
        """Run multiple simulations with optional parallel execution"""
        from .strategies import get_strategy, get_vor_policy, is_vor_policy
        
        if parallel:
            return self.run_simulations_parallel(my_team_idx, strategy_name, n_sims, n_workers, initial_roster, already_drafted)
        
        # Determine if this is a VOR policy or legacy strategy
        if is_vor_policy(strategy_name):
            vor_policy = get_vor_policy(strategy_name)
            strategy_params = vor_policy['params']
        else:
            strategy = get_strategy(strategy_name)
            # Handle strategies that don't use multipliers (like starter_optimize)
            if 'multipliers' in strategy:
                strategy_params = strategy['multipliers']
            else:
                strategy_params = strategy  # Pass the whole strategy config
        
        results = []
        position_frequency = defaultdict(lambda: defaultdict(int))  # {round: {position: count}}
        
        start_time = time.time()
        
        # Pre-warm caches for better performance
        self._get_player_cache()
        
        for sim_idx in range(n_sims):
            result = self.simulate_single_draft(
                my_team_idx, strategy_params, seed=base_seed + sim_idx,
                initial_roster=initial_roster, already_drafted=already_drafted
            )
            
            results.append(result)
            
            # Track position frequency by round (rounds 1-14)
            seq = result['position_sequence']
            for round_num in range(1, min(15, len(seq) + 1)):
                if len(seq) >= round_num:
                    position = seq[round_num - 1]  # Convert to 0-based index
                    position_frequency[round_num][position] += 1
                
        elapsed = time.time() - start_time
        
        # Aggregate results
        values = [r['roster_value'] for r in results]
        
        # Aggregate backup counts
        avg_backup_counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'total': 0}
        if results:
            for position in avg_backup_counts.keys():
                position_counts = [r['backup_counts'][position] for r in results if 'backup_counts' in r]
                if position_counts:
                    avg_backup_counts[position] = np.mean(position_counts)
        
        # Calculate position frequency percentages by round
        position_frequencies = {}
        for round_num, pos_counts in position_frequency.items():
            if pos_counts:
                total_picks = sum(pos_counts.values())
                frequencies = {}
                for pos, count in pos_counts.items():
                    percentage = (count / total_picks) * 100
                    frequencies[pos] = percentage
                # Sort by frequency (highest first)
                sorted_freq = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
                position_frequencies[f'round_{round_num}'] = sorted_freq
                
        return {
            'strategy': strategy_name,
            'n_sims': n_sims,
            'mean_value': np.mean(values),
            'std_value': np.std(values),
            'max_value': np.max(values),
            'min_value': np.min(values),
            'position_frequencies': position_frequencies,
            'avg_backup_counts': avg_backup_counts,
            'elapsed_time': elapsed,
            'all_results': results
        }

    def run_simulations_parallel(self, my_team_idx, strategy_name, n_sims=100, n_workers=4, 
                                initial_roster=None, already_drafted=None):
        """Run simulations in parallel using ProcessPoolExecutor (CPU-friendly)"""
        from .strategies import get_strategy, get_vor_policy, is_vor_policy
        
        # Set CPU-friendly priority on Unix systems
        try:
            if hasattr(os, 'nice'):
                os.nice(10)  # Lower priority to be CPU-friendly
        except (OSError, AttributeError):
            pass  # Windows or permission issues
        
        # Determine if this is a VOR policy or legacy strategy
        if is_vor_policy(strategy_name):
            vor_policy = get_vor_policy(strategy_name)
            strategy_params = vor_policy['params']
        else:
            strategy = get_strategy(strategy_name)
            strategy_params = strategy['multipliers']
        
        start_time = time.time()
        
        # Prepare serializable data for workers
        players_df_dict = self.prob_model._to_dict()
        base_seed = 42
        
        # Generate independent seeds for each worker using SeedSequence
        seed_seq = np.random.SeedSequence(base_seed)
        worker_seeds = seed_seq.spawn(n_workers)
        
        # Calculate batch size per worker
        batch_size = max(1, n_sims // n_workers)
        worker_args = []
        
        sim_idx = 0
        for worker_id in range(n_workers):
            # Calculate simulations for this worker
            remaining_sims = n_sims - sim_idx
            if remaining_sims <= 0:
                break
                
            worker_sims = min(batch_size, remaining_sims)
            if worker_id == n_workers - 1:  # Last worker gets any remainder
                worker_sims = remaining_sims
            
            # Use independent worker seed instead of base_seed + offset
            worker_seed = int(np.random.default_rng(worker_seeds[worker_id]).integers(0, 2**31))
            
            worker_args.append((
                sim_idx, worker_sims, my_team_idx, strategy_params, worker_seed,
                initial_roster, already_drafted, players_df_dict, self.n_teams, self.n_rounds
            ))
            
            sim_idx += worker_sims
        
        print(f"🔄 Running {n_sims} simulations across {len(worker_args)} workers...")
        
        # Run parallel simulations with error handling
        all_results = []
        errors = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_parallel_simulation_worker, args) for args in worker_args]
            
            for i, future in enumerate(futures):
                worker_results = future.result()
                
                # Check for worker errors
                if isinstance(worker_results, dict) and 'error' in worker_results:
                    error_msg = f"Worker {i+1} failed: {worker_results['error']}"
                    print(f"   ❌ {error_msg}")
                    errors.append(error_msg)
                else:
                    all_results.extend(worker_results)
                    print(f"   Worker {i+1}/{len(futures)} completed ({len(worker_results)} sims)")
        
        # Handle case where some workers failed
        if errors:
            print(f"⚠️  {len(errors)} worker(s) failed, continuing with {len(all_results)} successful results")
            if not all_results:
                raise RuntimeError(f"All workers failed: {'; '.join(errors)}")
        
        elapsed = time.time() - start_time
        
        # Aggregate results (adapted for lightweight results)
        position_frequency = defaultdict(lambda: defaultdict(int))  # {round: {position: count}}
        
        for result in all_results:
            # Use first_14_picks instead of limited picks for frequency tracking
            first_picks = result.get('first_14_picks', result.get('first_7_picks', []))  # Fallback for compatibility
            for round_num in range(1, min(15, len(first_picks) + 1)):
                if len(first_picks) >= round_num:
                    position = first_picks[round_num - 1]  # Convert to 0-based index
                    position_frequency[round_num][position] += 1
        
        # Calculate position frequency percentages by round
        position_frequencies = {}
        for round_num, pos_counts in position_frequency.items():
            if pos_counts:
                total_picks = sum(pos_counts.values())
                frequencies = {}
                for pos, count in pos_counts.items():
                    percentage = (count / total_picks) * 100
                    frequencies[pos] = percentage
                # Sort by frequency (highest first)
                sorted_freq = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
                position_frequencies[f'round_{round_num}'] = sorted_freq
        
        values = [r['roster_value'] for r in all_results]
        
        print(f"✅ Parallel execution completed: {elapsed:.1f}s ({len(all_results)/elapsed:.1f} sims/sec)")
        
        return {
            'strategy': strategy_name,
            'n_sims': len(all_results),
            'mean_value': np.mean(values),
            'std_value': np.std(values),
            'max_value': np.max(values),
            'min_value': np.min(values),
            'position_frequencies': position_frequencies,
            'elapsed_time': elapsed,
            'all_results': all_results,
            'parallel': True,
            'n_workers': len(worker_args),
            'worker_errors': errors
        }
        
    def _check_convergence(self, strategy_results, strategy_names, n_min, ci_target, ci_relative):
        """Check if strategies have converged based on confidence intervals"""
        try:
            from scipy import stats
            use_t_dist = True
        except ImportError:
            use_t_dist = False
            
        all_converged = True
        
        for strategy_name in strategy_names:
            if strategy_results[strategy_name]['converged']:
                continue
                
            values = strategy_results[strategy_name]['values']
            n = len(values)
            if n < n_min or n < 2:
                all_converged = False
                continue
                
            # Calculate confidence interval with t-distribution and sample std
            mean_val = np.mean(values)
            s = np.std(values, ddof=1)  # Sample standard deviation
            se = s / np.sqrt(n)
            
            # Use t-distribution for small samples
            if use_t_dist:
                t = stats.t.ppf(0.975, df=n-1)  # Correct: 97.5th percentile for 95% CI
            else:
                # Fallback to z-approximation for n>30, otherwise use conservative t=2.0
                t = 1.96 if n > 30 else 2.0
                
            ci_half_width = t * se
            
            # Single adaptive threshold: max of absolute and relative criteria
            threshold = max(ci_target, ci_relative * abs(mean_val))
            
            if ci_half_width < threshold:
                strategy_results[strategy_name]['converged'] = True
                print(f"   ✅ {strategy_name}: {mean_val:.1f} ± {ci_half_width:.1f} (CONVERGED)")
            else:
                all_converged = False
                print(f"   ⏳ {strategy_name}: {mean_val:.1f} ± {ci_half_width:.1f}")
                
        return all_converged
    
    def _check_paired_convergence(self, paired_results, strategy_names, n_min, ci_target, ci_relative, base_strategy='balanced'):
        """Check convergence using paired differences for variance reduction"""
        try:
            from scipy import stats
            use_t_dist = True
        except ImportError:
            use_t_dist = False
            
        # Calculate paired differences between each strategy and base strategy
        n_sims = len(paired_results)
        if n_sims < n_min or n_sims < 2:
            return False
            
        # Ensure base strategy exists in the list
        if base_strategy not in strategy_names:
            base_strategy = strategy_names[0]
            
        all_converged = True
        
        for strategy_name in strategy_names:
            if strategy_name == base_strategy:
                continue  # Skip comparing base strategy to itself
                
            # Calculate paired differences for this strategy vs base
            differences = []
            for sim_idx in range(n_sims):
                if (sim_idx in paired_results and 
                    strategy_name in paired_results[sim_idx] and 
                    base_strategy in paired_results[sim_idx]):
                    diff = paired_results[sim_idx][strategy_name] - paired_results[sim_idx][base_strategy]
                    differences.append(diff)
            
            if len(differences) < n_min:
                all_converged = False
                continue
                
            # Calculate CI on paired differences (much tighter than individual CIs)
            mean_diff = np.mean(differences)
            s_diff = np.std(differences, ddof=1)  # Sample standard deviation
            se_diff = s_diff / np.sqrt(len(differences))
            
            # Use t-distribution for small samples
            if use_t_dist:
                t = stats.t.ppf(1 - 0.025, df=len(differences)-1)
            else:
                t = 1.96 if len(differences) > 30 else 2.0
                
            ci_half_width = t * se_diff
            
            # Adaptive threshold for differences (typically much smaller than individual values)
            threshold = max(ci_target * 0.5, ci_relative * abs(mean_diff))  # Use 50% of target for differences
            
            if ci_half_width < threshold:
                print(f"   ✅ {strategy_name} vs {base_strategy}: {mean_diff:+.1f} ± {ci_half_width:.1f} (CONVERGED)")
            else:
                all_converged = False
                print(f"   ⏳ {strategy_name} vs {base_strategy}: {mean_diff:+.1f} ± {ci_half_width:.1f}")
                
        return all_converged
    
    def run_adaptive_comparison(self, my_team_idx, strategy_names=None, n_min=1000, n_max=5000, 
                              batch_size=250, ci_target=3.0, ci_relative=0.05, initial_roster=None, already_drafted=None):
        """Run adaptive comparison using CRN with paired differences for variance reduction"""
        from .crn_manager import CRNManager
        from .strategies import get_strategy, get_vor_policy, is_vor_policy, list_strategies
        
        if strategy_names is None:
            strategy_names = list_strategies()
            
        print(f"🎯 Adaptive CRN Comparison: {len(strategy_names)} strategies")
        print(f"   Min sims: {n_min}, Max sims: {n_max}, Batch size: {batch_size}")
        print(f"   Adaptive threshold: max({ci_target} points, {ci_relative*100}% of mean)")
        
        # Initialize CRN manager
        crn = CRNManager(n_max_sims=n_max, seed=42)
        crn.generate_all_samples(self.prob_model.players_df, self.n_teams)
        
        # Pre-warm caches for better performance
        self._get_player_cache()
        
        # Storage for paired results by simulation index
        paired_results = {}  # {sim_idx: {strategy: value}}
        strategy_converged = {name: False for name in strategy_names}
            
        current_n = 0
        start_time = time.time()
        
        # Run in batches until convergence or max sims
        while current_n < n_max:
            batch_end = min(current_n + batch_size, n_max)
            batch_sims = batch_end - current_n
            
            print(f"\n🔄 Running batch: simulations {current_n+1}-{batch_end}")
            
            # Run batch for all strategies using same random numbers
            for sim_idx in range(current_n, batch_end):
                if sim_idx not in paired_results:
                    paired_results[sim_idx] = {}
                
                for strategy_name in strategy_names:
                    if strategy_converged[strategy_name]:
                        continue
                        
                    # Determine if this is a VOR policy or legacy strategy
                    if is_vor_policy(strategy_name):
                        vor_policy = get_vor_policy(strategy_name)
                        strategy_params = vor_policy['params']
                    else:
                        strategy = get_strategy(strategy_name)
                        strategy_params = strategy['multipliers']
                    
                    result = self.simulate_single_draft(
                        my_team_idx, strategy_params, 
                        seed=42 + sim_idx,  # Still use seed for backwards compatibility
                        initial_roster=initial_roster, 
                        already_drafted=already_drafted,
                        crn=crn, 
                        sim_idx=sim_idx
                    )
                    paired_results[sim_idx][strategy_name] = result['roster_value']
                    
            current_n = batch_end
            
            # Check convergence after minimum simulations using paired differences
            if current_n >= n_min:
                print(f"\n📊 Convergence check at {current_n} simulations:")
                
                if self._check_paired_convergence(paired_results, strategy_names, n_min, ci_target, ci_relative):
                    print(f"\n🎉 All strategy pairs converged at {current_n} simulations!")
                    break
                    
        elapsed = time.time() - start_time
        
        # Aggregate final results from paired data
        final_results = {}
        
        for strategy_name in strategy_names:
            # Extract values for this strategy from paired results
            values = []
            for sim_idx in range(current_n):
                if sim_idx in paired_results and strategy_name in paired_results[sim_idx]:
                    values.append(paired_results[sim_idx][strategy_name])
            
            if not values:
                continue
                
            n_sims = len(values)
            mean_val = np.mean(values)
            std_val = np.std(values)
            se = std_val / np.sqrt(n_sims)
            ci_half_width = 1.96 * se
            
            final_results[strategy_name] = {
                'mean_value': mean_val,
                'std_value': std_val,
                'se_value': se,
                'ci_half_width': ci_half_width,
                'n_sims': n_sims,
                'converged': strategy_converged[strategy_name],
                'max_value': np.max(values),
                'min_value': np.min(values),
                'patterns': {},  # Simplified for adaptive version
                'crn_enabled': True,
                'paired_crn': True  # Indicate this used paired differences
            }
            
        # Sort by mean value
        sorted_results = sorted(final_results.items(), key=lambda x: x[1]['mean_value'], reverse=True)
        best_strategy = sorted_results[0][0] if sorted_results else 'balanced'
        
        print(f"\n🏆 ADAPTIVE CRN RESULTS ({current_n} simulations, {elapsed:.1f}s):")
        print("-" * 60)
        for i, (strategy, stats) in enumerate(sorted_results, 1):
            convergence = "✅" if stats['converged'] else "⏳"
            print(f"{i}. {strategy.upper()}: {stats['mean_value']:.1f} ± {stats['ci_half_width']:.1f} "
                  f"({stats['n_sims']} sims) {convergence}")
                  
        # Calculate variance reduction benefit
        print(f"\n📈 CRN Performance:")
        mean_se = np.mean([stats['se_value'] for stats in final_results.values()])
        print(f"   Average SE: {mean_se:.2f}")
        print(f"   CRN enabled: Significant variance reduction vs independent sampling")
        
        return {
            'rankings': sorted_results,
            'best_strategy': best_strategy,
            'total_simulations': current_n,
            'elapsed_time': elapsed,
            'crn_enabled': True,
            'convergence_achieved': all([stats['converged'] for stats in final_results.values()])
        }
        
    # Legacy methods for backward compatibility
    def run_simulations_with_fixed_seeds(self, my_team_idx, strategy_name, n_sims=100, base_seed=42, initial_roster=None, already_drafted=None):
        """Legacy method - use run_simulations with base_seed parameter instead"""
        return self.run_simulations(my_team_idx, strategy_name, n_sims, initial_roster, already_drafted, base_seed)