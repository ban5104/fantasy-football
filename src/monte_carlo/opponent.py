"""
Opponent Behavior Model for Fantasy Football Draft
Models how opponents draft based on rankings and roster needs
"""

import numpy as np
from typing import Dict, List, Set, Optional
from collections import defaultdict

# Personality archetypes for opponent diversity
PERSONALITY_ARCHETYPES = {
    'conservative': {'ranking_mult': 1.2, 'need_mult': 0.8},
    'aggressive': {'ranking_mult': 0.8, 'need_mult': 1.2},
    'balanced': {'ranking_mult': 1.0, 'need_mult': 1.0}
}


class OpponentModel:
    """Model opponent draft behavior combining rankings and roster needs"""
    
    # Standard roster requirements
    ROSTER_REQUIREMENTS = {
        'QB': 1,
        'RB': 2,
        'WR': 2,
        'TE': 1,
        'FLEX': 1,  # Best remaining RB/WR/TE
        'K': 1,
        'DST': 1
    }
    
    # Maximum reasonable roster sizes
    POSITION_LIMITS = {
        'QB': 2,
        'RB': 5,
        'WR': 5,
        'TE': 2,
        'K': 1,
        'DST': 1
    }
    
    def __init__(self, probability_model):
        """
        Initialize opponent model
        
        Args:
            probability_model: ProbabilityModel instance for base probabilities
        """
        self.prob_model = probability_model
        
        # Assign random personalities to each team (excluding ourselves)
        self.team_personalities = {}
        archetype_names = list(PERSONALITY_ARCHETYPES.keys())
        for team_idx in range(14):  # Assuming 14 teams
            personality = np.random.choice(archetype_names)
            self.team_personalities[team_idx] = personality
        
    def get_round_weights(self, round_num: int) -> tuple:
        """
        Get ranking vs need weights based on round
        
        Early rounds: Follow rankings closely
        Later rounds: Fill roster needs
        
        Args:
            round_num: Current round (1-based)
            
        Returns:
            Tuple of (ranking_weight, need_weight)
        """
        if round_num <= 2:
            return (0.90, 0.10)  # 90% rankings, 10% needs
        elif round_num <= 4:
            return (0.60, 0.40)  # 60% rankings, 40% needs
        elif round_num <= 6:
            return (0.40, 0.60)  # 40% rankings, 60% needs
        else:
            return (0.20, 0.80)  # 20% rankings, 80% needs
            
    def calculate_roster_need(self, roster: List[str], position: str) -> float:
        """
        Calculate how much a team needs a specific position
        
        Args:
            roster: List of positions already drafted
            position: Position to check need for
            
        Returns:
            Need multiplier (0.1 = filled, 1.0 = neutral, 2.0+ = urgent need)
        """
        # Count current positions
        pos_counts = defaultdict(int)
        for pos in roster:
            pos_counts[pos] += 1
            
        current_count = pos_counts[position]
        required = self.ROSTER_REQUIREMENTS.get(position, 0)
        limit = self.POSITION_LIMITS.get(position, 3)
        
        # Already at limit
        if current_count >= limit:
            return 0.1
            
        # Position filled
        if current_count >= required:
            # Still some value for depth, but reduced
            if position in ['RB', 'WR']:
                return 0.5  # Flex eligibility keeps value
            else:
                return 0.3  # Less valuable depth
                
        # Position needed
        deficit = required - current_count
        if deficit >= 2:
            return 2.5  # Urgent need
        elif deficit == 1:
            return 1.8  # Strong need
        else:
            return 1.0  # Neutral
            
    def detect_position_run(self, recent_picks: List[str], 
                          position: str, 
                          window: int = 8) -> bool:
        """
        Detect if there's a run on a specific position
        
        Args:
            recent_picks: List of recent position picks
            position: Position to check for run
            window: How many recent picks to consider
            
        Returns:
            True if position run detected
        """
        if len(recent_picks) < 3:
            return False
            
        # Look at last 'window' picks
        recent_window = recent_picks[-window:] if len(recent_picks) >= window else recent_picks
        pos_count = sum(1 for p in recent_window if p == position)
        
        # Run thresholds by position
        run_thresholds = {
            'QB': 3,   # 3+ QBs in window
            'RB': 5,   # 5+ RBs in window
            'WR': 5,   # 5+ WRs in window
            'TE': 3,   # 3+ TEs in window
        }
        
        threshold = run_thresholds.get(position, 4)
        return pos_count >= threshold
        
    def calculate_opponent_probabilities(self,
                                        available_players: Set[int],
                                        team_roster: List[str],
                                        recent_picks: List[str],
                                        round_num: int,
                                        team_idx: Optional[int] = None) -> Dict[int, float]:
        """
        Calculate adjusted pick probabilities for an opponent
        
        Args:
            available_players: Set of available player IDs
            team_roster: Opponent's current roster (list of positions)
            recent_picks: Recent picks across all teams (for run detection)
            round_num: Current round number (1-based)
            team_idx: Team index for personality-based adjustments
            
        Returns:
            Dictionary mapping player ID to adjusted probability
        """
        if not self.prob_model.players_df is not None:
            raise ValueError("Probability model must be loaded first")
            
        # Get base probabilities
        base_probs = self.prob_model.get_pick_probabilities(available_players)
        
        # Get round-based weights
        ranking_weight, need_weight = self.get_round_weights(round_num)
        
        # Apply personality modifiers
        if team_idx is not None and team_idx in self.team_personalities:
            personality = self.team_personalities[team_idx]
            archetype = PERSONALITY_ARCHETYPES[personality]
            ranking_weight *= archetype['ranking_mult']
            need_weight *= archetype['need_mult']
        
        # Calculate adjusted probabilities
        adjusted_probs = {}
        players_df = self.prob_model.players_df
        
        for player_id, base_prob in base_probs.items():
            if player_id not in players_df.index:
                continue
                
            position = players_df.loc[player_id, 'pos']
            
            # Calculate need multiplier
            need_mult = self.calculate_roster_need(team_roster, position)
            
            # Check for position run
            run_mult = 1.0
            if self.detect_position_run(recent_picks, position):
                run_mult = 1.3  # Slight boost during runs
                
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
            # Fallback to base probabilities
            return base_probs
            
    def predict_opponent_pick(self,
                             available_players: Set[int],
                             team_roster: List[str],
                             recent_picks: List[str],
                             round_num: int,
                             rng: Optional[np.random.Generator] = None,
                             team_idx: Optional[int] = None) -> Optional[int]:
        """
        Predict which player an opponent will pick
        
        Args:
            available_players: Set of available player IDs
            team_roster: Opponent's current roster
            recent_picks: Recent picks for run detection
            round_num: Current round
            rng: Random number generator for sampling
            team_idx: Team index for personality-based adjustments
            
        Returns:
            Player ID of predicted pick, or None if no players available
        """
        if not available_players:
            return None
            
        # Get adjusted probabilities
        probs = self.calculate_opponent_probabilities(
            available_players, team_roster, recent_picks, round_num, team_idx
        )
        
        if not probs:
            return None
            
        # Sample from distribution
        if rng is None:
            rng = np.random.default_rng()
            
        player_ids = list(probs.keys())
        probabilities = list(probs.values())
        
        # Ensure probabilities sum to 1 (handle floating point errors)
        probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()
        
        chosen_id = rng.choice(player_ids, p=probabilities)
        
        return chosen_id
        
    def get_position_from_id(self, player_id: int) -> str:
        """Helper to get position from player ID"""
        if self.prob_model.players_df is None:
            return 'FLEX'
        if player_id not in self.prob_model.players_df.index:
            return 'FLEX'
        return self.prob_model.players_df.loc[player_id, 'pos']