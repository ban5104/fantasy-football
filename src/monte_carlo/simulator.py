"""
Monte Carlo Simulation Engine for Fantasy Football Draft
Pure simulation logic - no data loading or strategy definitions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import time

from .strategies import ROSTER_REQUIREMENTS, POSITION_LIMITS, ROUND_POSITION_VALIDITY
from .depth import DepthEvaluator


class MonteCarloSimulator:
    """Run Monte Carlo simulations for draft strategy evaluation"""
    
    def __init__(self, 
                 probability_model,
                 opponent_model,
                 n_teams: int = 14,
                 n_rounds: int = 14):  # Default to 14 rounds for full draft
        """
        Initialize simulator
        
        Args:
            probability_model: ProbabilityModel instance
            opponent_model: OpponentModel instance
            n_teams: Number of teams in league
            n_rounds: Number of rounds to simulate
        """
        self.prob_model = probability_model
        self.opponent_model = opponent_model
        self.n_teams = n_teams
        self.n_rounds = n_rounds
        
        # Initialize depth evaluator for 14+ round analysis
        if hasattr(probability_model, 'players_df'):
            self.depth_evaluator = DepthEvaluator(probability_model.players_df)
        else:
            self.depth_evaluator = None
        
    def generate_snake_order(self) -> List[int]:
        """Generate snake draft pick order"""
        order = []
        for round_num in range(self.n_rounds):
            if round_num % 2 == 0:
                order.extend(range(self.n_teams))
            else:
                order.extend(reversed(range(self.n_teams)))
        return order
        
    def calculate_roster_value(self, roster_players: List[dict], 
                              include_depth: bool = None) -> Dict:
        """
        Calculate total value of a roster with dual value system
        
        Args:
            roster_players: List of player dictionaries with 'pos' and 'proj' keys
            include_depth: Whether to include depth analysis (auto-detect if None)
            
        Returns:
            Dictionary with starter_points, depth_value, total_value, and metrics
        """
        # Auto-detect depth mode based on number of rounds and roster size
        if include_depth is None:
            include_depth = (self.n_rounds >= 10 and len(roster_players) >= 10 
                           and self.depth_evaluator is not None)
        # Calculate optimal starting lineup
        starters, starter_points = self._calculate_optimal_lineup(roster_players)
        
        # Calculate consistency score
        if len(roster_players) > 0:
            projections = [p['proj'] for p in roster_players]
            avg_proj = np.mean(projections)
            std_proj = np.std(projections)
            coefficient_of_variation = std_proj / avg_proj if avg_proj > 0 else 1.0
            consistency_score = starter_points / (1 + coefficient_of_variation)
            composite_value = 0.8 * starter_points + 0.2 * consistency_score
        else:
            consistency_score = 0.0
            composite_value = starter_points
        
        # Initialize return value
        result = {
            'starter_points': starter_points,
            'starters': starters,
            'consistency_score': consistency_score,
            'composite_value': composite_value,
            'total_value': starter_points,  # Will be updated if depth included
            'depth_metrics': None
        }
        
        # Add depth analysis for 14-round drafts
        if include_depth and self.depth_evaluator:
            depth_metrics = self.depth_evaluator.evaluate_depth(roster_players, starters)
            depth_value = depth_metrics['depth_value']
            
            result.update({
                'depth_metrics': depth_metrics,
                'depth_value': depth_value,
                'total_value': starter_points + depth_value
            })
            
        return result
        
    def _calculate_optimal_lineup(self, roster_players: List[dict]) -> Tuple[List[dict], float]:
        """
        Calculate optimal starting lineup from roster
        
        Args:
            roster_players: List of player dictionaries
            
        Returns:
            Tuple of (starters_list, total_points)
        """
        # Group by position
        position_players = defaultdict(list)
        for player in roster_players:
            position_players[player['pos']].append(player)
            
        # Sort each position by value (highest first)
        for pos in position_players:
            position_players[pos].sort(key=lambda p: p['proj'], reverse=True)
            
        starters = []
        total_points = 0.0
        
        # QB: 1 starter
        if 'QB' in position_players and position_players['QB']:
            starter = position_players['QB'][0]
            starters.append(starter)
            total_points += starter['proj']
            
        # RB: 2 starters
        if 'RB' in position_players:
            for i in range(min(2, len(position_players['RB']))):
                starter = position_players['RB'][i]
                starters.append(starter)
                total_points += starter['proj']
                
        # WR: 2 starters  
        if 'WR' in position_players:
            for i in range(min(2, len(position_players['WR']))):
                starter = position_players['WR'][i]
                starters.append(starter)
                total_points += starter['proj']
                
        # TE: 1 starter
        if 'TE' in position_players and position_players['TE']:
            starter = position_players['TE'][0]
            starters.append(starter)
            total_points += starter['proj']
            
        # FLEX: Best remaining RB/WR/TE
        flex_candidates = []
        if 'RB' in position_players and len(position_players['RB']) > 2:
            flex_candidates.extend(position_players['RB'][2:])
        if 'WR' in position_players and len(position_players['WR']) > 2:
            flex_candidates.extend(position_players['WR'][2:])
        if 'TE' in position_players and len(position_players['TE']) > 1:
            flex_candidates.extend(position_players['TE'][1:])
            
        if flex_candidates:
            best_flex = max(flex_candidates, key=lambda p: p['proj'])
            starters.append(best_flex)
            total_points += best_flex['proj']
            
        # K: 1 starter
        if 'K' in position_players and position_players['K']:
            starter = position_players['K'][0]
            starters.append(starter)
            total_points += starter['proj']
            
        # DST: 1 starter
        if 'DST' in position_players and position_players['DST']:
            starter = position_players['DST'][0]
            starters.append(starter)
            total_points += starter['proj']
            
        return starters, total_points
        
    def select_best_player(self,
                          available_players: Set[int],
                          my_roster: List[dict],
                          strategy_multipliers: dict,
                          round_num: int,
                          recent_picks: Optional[List[str]] = None) -> Optional[int]:
        """
        Select best player for our team based on strategy
        
        Args:
            available_players: Set of available player IDs
            my_roster: Current roster (list of player dicts)
            strategy_multipliers: Position multipliers from strategy
            round_num: Current round (1-based)
            recent_picks: Recent picks for position run detection
            
        Returns:
            Player ID to draft, or None if no valid options
        """
        if not available_players:
            return None
            
        # Count current positions
        pos_counts = defaultdict(int)
        for player in my_roster:
            pos_counts[player['pos']] += 1
            
        # Get valid positions for this round
        valid_positions = ROUND_POSITION_VALIDITY.get(round_num, 
                                                      ['RB', 'WR', 'TE', 'QB'])
        
        # Detect position runs (2+ same position in last 3 picks)
        run_multipliers = {}
        if recent_picks and len(recent_picks) >= 3:
            last_3 = recent_picks[-3:]
            for pos in ['RB', 'WR', 'QB', 'TE']:
                if last_3.count(pos) >= 2:
                    run_multipliers[pos] = 1.2  # Boost probability
        
        # Score each available player
        best_score = -np.inf
        best_player_id = None
        
        players_df = self.prob_model.players_df
        
        for player_id in available_players:
            if player_id not in players_df.index:
                continue
                
            player_data = players_df.loc[player_id]
            pos = player_data['pos']
            
            # Skip if position not valid for round
            if pos not in valid_positions:
                continue
                
            # Skip if at position limit
            if pos_counts[pos] >= POSITION_LIMITS.get(pos, 3):
                continue
                
            # Calculate score
            # Base score from projections and rank
            proj = player_data['proj']
            rank = player_data['espn_rank']
            base_score = proj / (rank + 10)
            
            # Apply strategy multiplier
            strategy_mult = strategy_multipliers.get(pos, 1.0)
            
            # Apply need multiplier (prefer unfilled positions)
            need_mult = 1.0
            if pos_counts[pos] < ROSTER_REQUIREMENTS.get(pos, 0):
                need_mult = 1.5  # Boost for needed positions
                
            # Apply run multiplier
            run_mult = run_multipliers.get(pos, 1.0)
                
            # Final score
            score = base_score * strategy_mult * need_mult * run_mult
            
            if score > best_score:
                best_score = score
                best_player_id = player_id
                
        return best_player_id
        
    def simulate_single_draft(self,
                             my_team_idx: int,
                             strategy_multipliers: dict,
                             seed: int = 42,
                             initial_roster: Optional[List[str]] = None,
                             already_drafted: Optional[Set[str]] = None) -> dict:
        """
        Simulate a single draft
        
        Args:
            my_team_idx: Our team index (0-based)
            strategy_multipliers: Position multipliers from strategy
            seed: Random seed for reproducibility
            initial_roster: Optional starting roster (for live drafts)
            already_drafted: Optional set of already drafted player names
            
        Returns:
            Dictionary with simulation results
        """
        rng = np.random.default_rng(seed)
        
        # Initialize available players
        players_df = self.prob_model.players_df
        available = set(players_df.index)
        
        # Handle already drafted players
        if already_drafted:
            for player_name in already_drafted:
                # Find and remove player by name
                mask = players_df['player_name'] == player_name
                if mask.any():
                    player_id = players_df[mask].index[0]
                    available.discard(player_id)
                    
        # Initialize rosters
        team_rosters = {i: [] for i in range(self.n_teams)}
        my_roster = []
        
        # Add initial roster if provided
        if initial_roster:
            for player_name in initial_roster:
                mask = players_df['player_name'] == player_name
                if mask.any():
                    player_id = players_df[mask].index[0]
                    player_data = {
                        'id': player_id,
                        'name': player_name,
                        'pos': players_df.loc[player_id, 'pos'],
                        'proj': players_df.loc[player_id, 'proj']
                    }
                    my_roster.append(player_data)
                    team_rosters[my_team_idx].append(players_df.loc[player_id, 'pos'])
                    
        # Track draft flow
        recent_picks = []
        position_sequence = []
        
        # Generate pick order
        pick_order = self.generate_snake_order()
        
        # Simulate each pick
        for pick_num, team_idx in enumerate(pick_order):
            if not available:
                break
                
            round_num = (pick_num // self.n_teams) + 1
            
            if team_idx == my_team_idx:
                # Our pick
                player_id = self.select_best_player(
                    available, my_roster, strategy_multipliers, round_num, recent_picks
                )
                
                if player_id:
                    player_data = {
                        'id': player_id,
                        'name': players_df.loc[player_id, 'player_name'],
                        'pos': players_df.loc[player_id, 'pos'],
                        'proj': players_df.loc[player_id, 'proj']
                    }
                    my_roster.append(player_data)
                    position_sequence.append(player_data['pos'])
                    team_rosters[team_idx].append(player_data['pos'])
                    recent_picks.append(player_data['pos'])
                    available.discard(player_id)
                    
            else:
                # Opponent pick
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
                
        # Calculate final roster value with depth analysis
        roster_analysis = self.calculate_roster_value(my_roster)
        
        return {
            'roster': my_roster,
            'position_sequence': position_sequence,
            'roster_value': roster_analysis['total_value'],  # For backward compatibility
            'starter_points': roster_analysis['starter_points'],
            'consistency_score': roster_analysis['consistency_score'],
            'composite_value': roster_analysis['composite_value'],
            'depth_value': roster_analysis.get('depth_value', 0.0),
            'depth_metrics': roster_analysis.get('depth_metrics'),
            'starters': roster_analysis['starters'],
            'num_players': len(my_roster)
        }
        
    def run_simulations(self,
                       my_team_idx: int,
                       strategy_name: str,
                       n_sims: int = 100,
                       initial_roster: Optional[List[str]] = None,
                       already_drafted: Optional[Set[str]] = None) -> dict:
        """
        Run multiple simulations and aggregate results
        
        Args:
            my_team_idx: Our team index (0-based)
            strategy_name: Name of strategy to use
            n_sims: Number of simulations to run
            initial_roster: Optional starting roster
            already_drafted: Optional already drafted players
            
        Returns:
            Dictionary with aggregated results
        """
        from .strategies import get_strategy
        
        strategy = get_strategy(strategy_name)
        strategy_multipliers = strategy['multipliers']
        
        results = []
        position_patterns = defaultdict(list)
        
        start_time = time.time()
        
        for sim_idx in range(n_sims):
            # Run simulation
            result = self.simulate_single_draft(
                my_team_idx,
                strategy_multipliers,
                seed=42 + sim_idx,
                initial_roster=initial_roster,
                already_drafted=already_drafted
            )
            
            results.append(result)
            
            # Track position patterns
            seq = result['position_sequence']
            if len(seq) >= 2:
                position_patterns['2_round'].append('-'.join(seq[:2]))
            if len(seq) >= 3:
                position_patterns['3_round'].append('-'.join(seq[:3]))
            if len(seq) >= 4:
                position_patterns['4_round'].append('-'.join(seq[:4]))
                
        elapsed = time.time() - start_time
        
        # Aggregate results
        values = [r['roster_value'] for r in results]
        
        # Calculate confidence intervals
        confidence_intervals = {
            '95': (np.percentile(values, 2.5), np.percentile(values, 97.5)),
            '50': (np.percentile(values, 25), np.percentile(values, 75)),
            'median': np.median(values)
        }
        
        # Find most common patterns
        pattern_frequencies = {}
        for pattern_type, patterns in position_patterns.items():
            if patterns:
                from collections import Counter
                counts = Counter(patterns)
                pattern_frequencies[pattern_type] = counts.most_common(5)
                
        return {
            'strategy': strategy_name,
            'n_sims': n_sims,
            'mean_value': np.mean(values),
            'std_value': np.std(values),
            'max_value': np.max(values),
            'min_value': np.min(values),
            'confidence_intervals': confidence_intervals,
            'pattern_frequencies': pattern_frequencies,
            'elapsed_time': elapsed,
            'all_results': results
        }