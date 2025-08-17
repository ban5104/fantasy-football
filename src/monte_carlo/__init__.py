"""Monte Carlo Draft Simulation System - Simplified"""

from .probability import ProbabilityModel
from .opponent import OpponentModel  
from .simulator import MonteCarloSimulator
from .strategies import get_strategy, list_strategies

import json
import os


class DraftSimulator:
    """Simplified API for draft simulations"""
    
    def __init__(self, n_teams=14, n_rounds=14, espn_weight=0.8, adp_weight=0.2):
        """Initialize with simple defaults"""
        self.n_teams = n_teams
        self.n_rounds = n_rounds
        
        self.prob_model = ProbabilityModel(espn_weight, adp_weight)
        self.opponent_model = OpponentModel(self.prob_model)
        self.simulator = MonteCarloSimulator(self.prob_model, self.opponent_model, n_teams, n_rounds)
        self.prob_model.load_data()
        
    def load_draft_state(self, state_file=None):
        """Load draft state from JSON file"""
        if state_file is None:
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            state_file = os.path.join(base_path, 'data/draft/monte_carlo_state.json')
            
        if not os.path.exists(state_file):
            return None
            
        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except:
            return None
            
    def run_strategy_comparison(self, my_team_idx, n_sims=100):
        """Compare all strategies for a draft position"""
        results = {}
        
        for strategy_name in list_strategies():
            print(f"Testing {strategy_name} strategy...")
            result = self.simulator.run_simulations(my_team_idx, strategy_name, n_sims)
            results[strategy_name] = {
                'mean_value': result['mean_value'],
                'std_value': result['std_value']
            }
            
        sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_value'], reverse=True)
        return {
            'rankings': sorted_results,
            'best_strategy': sorted_results[0][0] if sorted_results else 'balanced'
        }
        
    def get_recommendations(self, my_team_idx, strategy='balanced', current_roster=None, already_drafted=None, n_sims=100):
        """Get draft recommendations for current state"""
        result = self.simulator.run_simulations(my_team_idx, strategy, n_sims, 
                                               initial_roster=current_roster, already_drafted=already_drafted)
        return {
            'strategy': strategy,
            'expected_value': result['mean_value'],
            'confidence': result['std_value'],
            'simulations_run': n_sims
        }


def quick_simulation(my_pick_number=5, strategy='balanced', n_sims=100):
    """Quick simulation for a draft position"""
    sim = DraftSimulator()
    my_team_idx = my_pick_number - 1
    return sim.simulator.run_simulations(my_team_idx, strategy, n_sims)