# Simplified Monte Carlo State Reload Cell
import json
import os

def reload_draft_state():
    """Simplified state reload with essential validation only."""
    state_file = '../data/draft/monte_carlo_state.json'
    
    if not os.path.exists(state_file):
        print("❌ No backup draft state found - start backup draft script first!")
        return False
    
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        # Essential validation only
        required_keys = ['my_team_idx', 'current_global_pick', 'my_current_roster']
        if not all(key in state for key in required_keys):
            print(f"❌ Missing required keys in state file")
            return False
        
        # Basic type/range checks
        if (not isinstance(state['my_team_idx'], int) or 
            not isinstance(state['current_global_pick'], int) or 
            not isinstance(state['my_current_roster'], list) or
            not (0 <= state['my_team_idx'] < 14) or
            state['current_global_pick'] < 0):
            print("❌ Invalid state data")
            return False
        
        # Update CONFIG
        CONFIG.update({
            'my_team_idx': state['my_team_idx'],
            'current_global_pick': state['current_global_pick'],
            'my_current_roster': state['my_current_roster']
        })
        
        team_name = state.get('team_name', f'Team {CONFIG["my_team_idx"]+1}')
        print(f"✅ State reloaded: {team_name} (#{CONFIG['my_team_idx']+1})")
        print(f"   Pick #{CONFIG['current_global_pick']+1} | Roster: {CONFIG['my_current_roster']}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading state: {e}")
        return False

# Auto-reload on execution
try:
    reload_draft_state()
except NameError:
    print("⚠️ CONFIG not loaded - run configuration cell first")