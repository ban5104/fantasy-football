"""
Monte Carlo Draft Simulation System
Clean API for fantasy football draft analysis
"""

from .probability import ProbabilityModel
from .opponent import OpponentModel  
from .simulator import MonteCarloSimulator
from .strategies import STRATEGIES, get_strategy, list_strategies
from .depth import DepthEvaluator
from .pattern_detector import PatternDetector

import json
import os
from typing import Optional, Dict, List, Set

# Module-level configuration
CONFIG = {
    'base_path': '/Users/ben/projects/fantasy-football-draft-spreadsheet-draft-pick-odds',
    'espn_weight': 0.8,
    'adp_weight': 0.2,
    'temperature': 5.0,
    'n_teams': 14,
    'n_rounds': 14,
    'n_sims': 100
}


class DraftSimulator:
    """High-level API for draft simulations"""
    
    def __init__(self, **kwargs):
        """
        Initialize draft simulator
        
        Args:
            **kwargs: Override CONFIG defaults (n_teams, n_rounds, espn_weight, adp_weight, etc.)
        """
        # Merge CONFIG with kwargs
        config = {**CONFIG, **kwargs}
        
        self.n_teams = config['n_teams']
        self.n_rounds = config['n_rounds']
        
        # Initialize models
        self.prob_model = ProbabilityModel(config['espn_weight'], config['adp_weight'])
        self.opponent_model = OpponentModel(self.prob_model)
        self.simulator = MonteCarloSimulator(
            self.prob_model, 
            self.opponent_model,
            self.n_teams,
            self.n_rounds
        )
        
        # Load data
        self.prob_model.load_data()
        
    def load_draft_state(self, state_file: Optional[str] = None) -> Optional[dict]:
        """
        Load draft state from JSON file
        
        Args:
            state_file: Path to state file (default: data/draft/monte_carlo_state.json)
            
        Returns:
            Draft state dictionary or None if not found
        """
        if state_file is None:
            state_file = os.path.join(CONFIG['base_path'], 'data/draft/monte_carlo_state.json')
            
        if not os.path.exists(state_file):
            return None
            
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            return state
        except Exception as e:
            print(f"Error loading draft state: {e}")
            return None
            
    def run_strategy_comparison(self, 
                               my_team_idx: int,
                               n_sims: int = 100) -> Dict:
        """
        Compare all strategies for a given draft position
        
        Args:
            my_team_idx: Team index (0-based)
            n_sims: Simulations per strategy
            
        Returns:
            Comparison results
        """
        results = {}
        
        for strategy_name in list_strategies():
            print(f"Testing {strategy_name} strategy...")
            
            result = self.simulator.run_simulations(
                my_team_idx,
                strategy_name,
                n_sims
            )
            
            results[strategy_name] = {
                'mean_value': result['mean_value'],
                'std_value': result['std_value'],
                'patterns': result.get('pattern_frequencies', {})
            }
            
        # Sort by mean value
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]['mean_value'],
            reverse=True
        )
        
        return {
            'rankings': sorted_results,
            'best_strategy': sorted_results[0][0] if sorted_results else 'balanced'
        }
        
    def get_recommendations(self,
                           my_team_idx: int,
                           strategy: str = 'balanced',
                           current_roster: Optional[List[str]] = None,
                           already_drafted: Optional[Set[str]] = None,
                           n_sims: int = 100) -> Dict:
        """
        Get draft recommendations for current state
        
        Args:
            my_team_idx: Team index (0-based)
            strategy: Strategy name
            current_roster: Current roster (player names)
            already_drafted: All drafted players
            n_sims: Number of simulations
            
        Returns:
            Recommendations with expected values
        """
        # Run simulations
        result = self.simulator.run_simulations(
            my_team_idx,
            strategy,
            n_sims,
            initial_roster=current_roster,
            already_drafted=already_drafted
        )
        
        # Analyze next best picks
        # This would need more sophisticated analysis
        # For now, return simulation summary
        
        return {
            'strategy': strategy,
            'expected_value': result['mean_value'],
            'confidence': result['std_value'],
            'simulations_run': n_sims,
            'common_patterns': result.get('pattern_frequencies', {})
        }
        
    def discover_natural_patterns(self,
                                 my_team_idx: int,
                                 n_sims: int = 100,
                                 current_roster: Optional[List[str]] = None,
                                 already_drafted: Optional[Set[str]] = None) -> Dict:
        """
        Run pattern discovery mode to identify emergent strategies
        
        Args:
            my_team_idx: Team index (0-based)
            n_sims: Number of simulations
            current_roster: Current roster (player names)
            already_drafted: All drafted players
            
        Returns:
            Pattern analysis with discovered strategies
        """
        # Run simulations with balanced strategy as baseline
        result = self.simulator.run_simulations(
            my_team_idx,
            'balanced',  # Use balanced as baseline for natural discovery
            n_sims,
            initial_roster=current_roster,
            already_drafted=already_drafted
        )
        
        # Analyze patterns
        pattern_detector = PatternDetector()
        pattern_analysis = pattern_detector.analyze_patterns(result['all_results'])
        
        return {
            'pattern_analysis': pattern_analysis,
            'baseline_strategy': 'balanced',
            'discovery_mode': True,
            'n_sims': n_sims,
            'my_team_idx': my_team_idx
        }


# Convenience functions for easy use
def quick_simulation(my_pick_number: int = 5,
                    strategy: str = 'balanced',
                    n_sims: int = 100,
                    n_rounds: int = 14) -> Dict:
    """
    Quick simulation for a draft position
    
    Args:
        my_pick_number: Your pick number (1-based, e.g., 5 for 5th pick)
        strategy: Strategy name
        n_sims: Number of simulations
        n_rounds: Number of rounds to simulate
        
    Returns:
        Simulation results
    """
    sim = DraftSimulator(n_rounds=n_rounds)
    my_team_idx = my_pick_number - 1  # Convert to 0-based
    
    return sim.simulator.run_simulations(
        my_team_idx,
        strategy,
        n_sims
    )
    

def compare_all_strategies(my_pick_number: int = 5,
                          n_sims: int = 100,
                          n_rounds: int = 14) -> Dict:
    """
    Compare all strategies for your draft position
    
    Args:
        my_pick_number: Your pick number (1-based)
        n_sims: Simulations per strategy
        n_rounds: Number of rounds to simulate
        
    Returns:
        Strategy comparison results
    """
    sim = DraftSimulator(n_rounds=n_rounds)
    my_team_idx = my_pick_number - 1
    
    return sim.run_strategy_comparison(my_team_idx, n_sims)


def discover_patterns(my_pick_number: int = 5,
                     n_sims: int = 100,
                     n_rounds: int = 14) -> Dict:
    """
    Discover natural draft patterns for your draft position
    
    Args:
        my_pick_number: Your pick number (1-based)
        n_sims: Number of simulations
        n_rounds: Number of rounds to simulate
        
    Returns:
        Pattern discovery results
    """
    sim = DraftSimulator(n_rounds=n_rounds)
    my_team_idx = my_pick_number - 1
    
    return sim.discover_natural_patterns(my_team_idx, n_sims)