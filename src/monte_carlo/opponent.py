"""Opponent Behavior Model - Simplified"""

import numpy as np
from typing import Dict, List, Set, Optional
from collections import defaultdict


class OpponentModel:
    """Model opponent draft behavior combining rankings and roster needs"""
    
    ROSTER_REQUIREMENTS = {
        'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'K': 1, 'DST': 1
    }
    
    POSITION_LIMITS = {
        'QB': 2, 'RB': 5, 'WR': 5, 'TE': 2, 'K': 1, 'DST': 1
    }
    
    def __init__(self, probability_model):
        self.prob_model = probability_model
        
    def get_round_weights(self, round_num):
        """Get ranking vs need weights based on round"""
        if round_num <= 2:
            return (0.90, 0.10)  # 90% rankings, 10% needs
        elif round_num <= 4:
            return (0.60, 0.40)
        elif round_num <= 6:
            return (0.40, 0.60)
        else:
            return (0.20, 0.80)  # 20% rankings, 80% needs
            
    def calculate_roster_need(self, roster, position):
        """Calculate how much a team needs a specific position"""
        pos_counts = defaultdict(int)
        for pos in roster:
            pos_counts[pos] += 1
            
        current_count = pos_counts[position]
        required = self.ROSTER_REQUIREMENTS.get(position, 0)
        limit = self.POSITION_LIMITS.get(position, 3)
        
        if current_count >= limit:
            return 0.1  # Already at limit
        if current_count >= required:
            return 0.5 if position in ['RB', 'WR'] else 0.3  # Some depth value
            
        # Position needed
        deficit = required - current_count
        if deficit >= 2:
            return 2.5  # Urgent need
        elif deficit == 1:
            return 1.8  # Strong need
        else:
            return 1.0  # Neutral
            
    def detect_position_run(self, recent_picks, position, window=8):
        """Detect if there's a run on a specific position"""
        if len(recent_picks) < 3:
            return False
            
        recent_window = recent_picks[-window:] if len(recent_picks) >= window else recent_picks
        pos_count = sum(1 for p in recent_window if p == position)
        
        run_thresholds = {'QB': 3, 'RB': 5, 'WR': 5, 'TE': 3}
        threshold = run_thresholds.get(position, 4)
        return pos_count >= threshold
        
    def calculate_opponent_probabilities(self, available_players, team_roster, recent_picks, round_num, team_idx=None):
        """Calculate adjusted pick probabilities for an opponent"""
        if self.prob_model.players_df is None:
            raise ValueError("Probability model must be loaded first")
            
        base_probs = self.prob_model.get_pick_probabilities(available_players)
        ranking_weight, need_weight = self.get_round_weights(round_num)
        
        adjusted_probs = {}
        players_df = self.prob_model.players_df
        
        for player_id, base_prob in base_probs.items():
            if player_id not in players_df.index:
                continue
                
            position = players_df.loc[player_id, 'pos']
            need_mult = self.calculate_roster_need(team_roster, position)
            
            # Check for position run
            run_mult = 1.3 if self.detect_position_run(recent_picks, position) else 1.0
                
            # Combine factors
            adjusted_prob = (
                ranking_weight * base_prob +
                need_weight * base_prob * need_mult * run_mult
            )
            
            adjusted_probs[player_id] = adjusted_prob
            
        # Renormalize
        total = sum(adjusted_probs.values())
        if total > 0:
            return {pid: p/total for pid, p in adjusted_probs.items()}
        else:
            return base_probs
            
    def predict_opponent_pick(self, available_players, team_roster, recent_picks, round_num, rng=None, team_idx=None):
        """Predict which player an opponent will pick"""
        if not available_players:
            return None
            
        probs = self.calculate_opponent_probabilities(available_players, team_roster, recent_picks, round_num, team_idx)
        
        if not probs:
            return None
            
        if rng is None:
            rng = np.random.default_rng()
            
        player_ids = list(probs.keys())
        probabilities = np.array(list(probs.values()))
        probabilities = probabilities / probabilities.sum()  # Ensure sum = 1
        
        return rng.choice(player_ids, p=probabilities)
        
    def get_position_from_id(self, player_id):
        """Helper to get position from player ID"""
        if self.prob_model.players_df is None or player_id not in self.prob_model.players_df.index:
            return 'FLEX'
        return self.prob_model.players_df.loc[player_id, 'pos']