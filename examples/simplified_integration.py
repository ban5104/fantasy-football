#!/usr/bin/env python3
"""
Complete example showing simplified Monte Carlo integration
Demonstrates how all simplified components work together
"""

import sys
import os
import json
import yaml
from typing import List, Dict, Any, Optional

class SimplifiedDraftTracker:
    """Streamlined draft tracker with essential functionality only."""
    
    def __init__(self, force_dynamic_vbd: Optional[bool] = None):
        self.picks = []
        self.current_pick = 1
        self.my_draft_position = None
        self.output_dir = "data/draft"
        
        # Load config and team names in one step
        self.config = self._load_complete_config()
        self.team_names = self.config['team_names']
        self.dynamic_vbd_enabled = self._get_dynamic_vbd_setting(force_dynamic_vbd)
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _load_complete_config(self) -> Dict[str, Any]:
        """Load complete configuration with fallbacks."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'league-config.yaml')
        
        # Default config
        config = {
            'team_names': [f"Team {i}" for i in range(1, 15)],
            'dynamic_vbd': {'enabled': False},
            'basic_settings': {'teams': 14}
        }
        
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f) or {}
            config.update(file_config)
            
            # Ensure team names
            if len(config.get('team_names', [])) < 14:
                num_teams = config.get('basic_settings', {}).get('teams', 14)
                config['team_names'] = [f"Team {i}" for i in range(1, num_teams + 1)]
            else:
                config['team_names'] = config['team_names'][:14]
                
        except Exception as e:
            print(f"⚠️ Using default config: {e}")
        
        return config
    
    def _get_dynamic_vbd_setting(self, force_setting: Optional[bool]) -> bool:
        """Get Dynamic VBD setting with precedence."""
        if force_setting is not None:
            return force_setting
        return self.config.get('dynamic_vbd', {}).get('enabled', False)
    
    def select_draft_position(self) -> None:
        """Simplified team selection."""
        print("\n🏈 SELECT YOUR DRAFT POSITION")
        print("=" * 40)
        
        for i, name in enumerate(self.team_names, 1):
            print(f"  {i:2d}. {name}")
        
        while True:
            try:
                pos = int(input(f"\nSelect position (1-{len(self.team_names)}): "))
                if 1 <= pos <= len(self.team_names):
                    self.my_draft_position = pos
                    print(f"✅ You are: {self.team_names[pos-1]} (Pick #{pos})")
                    return
                print(f"❌ Position must be between 1 and {len(self.team_names)}")
            except (ValueError, KeyboardInterrupt):
                print("❌ Invalid input or cancelled")
                return
    
    def export_monte_carlo_state(self) -> None:
        """Simplified Monte Carlo export."""
        if not self.my_draft_position:
            return
        
        try:
            my_team_name = self.team_names[self.my_draft_position - 1]
            my_picks = [p for p in self.picks if p['team_name'] == my_team_name]
            
            state = {
                'my_team_idx': self.my_draft_position - 1,
                'current_global_pick': self.current_pick - 1,
                'my_current_roster': [p['player_name'] for p in my_picks],
                'team_name': my_team_name,
                'total_teams': len(self.team_names)
            }
            
            state_file = os.path.join(self.output_dir, "monte_carlo_state.json")
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            print(f"📡 Monte Carlo state exported")
            
        except Exception as e:
            print(f"⚠️ Export error: {e}")

def reload_monte_carlo_state(config: Dict[str, Any]) -> bool:
    """Simplified Monte Carlo state reload."""
    state_file = 'data/draft/monte_carlo_state.json'
    
    if not os.path.exists(state_file):
        print("❌ No state file found")
        return False
    
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        # Essential validation
        required = ['my_team_idx', 'current_global_pick', 'my_current_roster']
        if not all(key in state for key in required):
            print("❌ Invalid state file")
            return False
        
        # Update config
        config.update({
            'my_team_idx': state['my_team_idx'],
            'current_global_pick': state['current_global_pick'],
            'my_current_roster': state['my_current_roster']
        })
        
        print(f"✅ State reloaded: Team #{config['my_team_idx']+1}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Draft tracker usage
    tracker = SimplifiedDraftTracker()
    tracker.select_draft_position()
    tracker.export_monte_carlo_state()
    
    # Monte Carlo usage
    monte_carlo_config = {
        'my_team_idx': 7,
        'current_global_pick': 0,
        'my_current_roster': []
    }
    
    if reload_monte_carlo_state(monte_carlo_config):
        print(f"Ready for Monte Carlo with: {monte_carlo_config}")