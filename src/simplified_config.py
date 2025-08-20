#!/usr/bin/env python3
"""
Simplified configuration loading - combines team names and config loading
"""

import yaml
import os
from typing import Dict, List, Any, Optional

def load_league_config(config_path: str = None) -> Dict[str, Any]:
    """Load complete league configuration with fallbacks."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'league-config.yaml')
    
    config = {
        'team_names': [f"Team {i}" for i in range(1, 15)],
        'dynamic_vbd': {'enabled': False},
        'basic_settings': {'teams': 14}
    }
    
    try:
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f) or {}
        
        # Merge configurations
        config.update(file_config)
        
        # Ensure team names
        team_names = config.get('team_names', [])
        if len(team_names) < 14:
            num_teams = config.get('basic_settings', {}).get('teams', 14)
            config['team_names'] = [f"Team {i}" for i in range(1, num_teams + 1)]
        else:
            config['team_names'] = team_names[:14]
        
        print(f"✅ Loaded config with {len(config['team_names'])} teams")
        
    except Exception as e:
        print(f"⚠️ Using default config due to error: {e}")
    
    return config

def get_dynamic_vbd_setting(config: Dict[str, Any], force_setting: Optional[bool] = None) -> bool:
    """Get Dynamic VBD setting with precedence: force > config > default."""
    if force_setting is not None:
        return force_setting
    
    return config.get('dynamic_vbd', {}).get('enabled', False)

# Example usage:
if __name__ == "__main__":
    config = load_league_config()
    print(f"Team names: {config['team_names'][:3]}...")
    print(f"Dynamic VBD: {get_dynamic_vbd_setting(config)}")
    print(f"Force enabled: {get_dynamic_vbd_setting(config, True)}")