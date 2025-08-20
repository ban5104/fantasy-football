#!/usr/bin/env python3
"""
Simplified Monte Carlo State Loader - Reduces complexity while maintaining functionality
"""

import json
import os

def reload_draft_state(config):
    """Reload CONFIG from backup draft state - simplified version."""
    state_file = '../data/draft/monte_carlo_state.json'
    
    if not os.path.exists(state_file):
        print("❌ No backup draft state found - start backup draft script first!")
        return False
    
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        # Simple validation - check required keys exist and are valid types
        required = {
            'my_team_idx': int,
            'current_global_pick': int, 
            'my_current_roster': list
        }
        
        for key, expected_type in required.items():
            if key not in state or not isinstance(state[key], expected_type):
                print(f"❌ Invalid state file - {key} missing or wrong type")
                return False
        
        # Range validation
        if not (0 <= state['my_team_idx'] < 14) or state['current_global_pick'] < 0:
            print(f"❌ Invalid values: team={state['my_team_idx']}, pick={state['current_global_pick']}")
            return False
        
        # Update config
        config.update({
            'my_team_idx': state['my_team_idx'],
            'current_global_pick': state['current_global_pick'],
            'my_current_roster': state['my_current_roster']
        })
        
        print(f"✅ Reloaded state: Team #{config['my_team_idx']+1}, Pick #{config['current_global_pick']+1}")
        print(f"   Roster ({len(config['my_current_roster'])}): {config['my_current_roster']}")
        return True
        
    except (json.JSONDecodeError, Exception) as e:
        print(f"❌ Error loading state: {e}")
        return False

# Example usage:
if __name__ == "__main__":
    CONFIG = {
        'my_team_idx': 7,
        'current_global_pick': 0,
        'my_current_roster': []
    }
    
    reload_draft_state(CONFIG)
    print(f"Updated CONFIG: {CONFIG}")