#!/usr/bin/env python3
"""
Monte Carlo Position Pattern Discovery - Optimized 7-Round Strategy
Uses real ESPN/ADP 80/20 probabilities to discover optimal draft strategies
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import time
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'n_teams': 14,
    'rounds': 7,  # Focus on first 7 rounds where strategy matters most
    'my_team_idx': 4,  # Pick #5 (0-based indexing) - INTERNAL USE ONLY
    'n_sims': 100,  # Reduced for testing (change to 500 for production)
    'top_k': 100,  # Top 100 players (7 rounds * 14 teams = 98 picks)
    'espn_weight': 0.8,
    'adp_weight': 0.2,
    'temperature': 5.0,  # Softmax temperature for probability distribution
    
    # Enhancement 1: Dynamic Opponent Behavior
    'need_modifier': 0.3,    # Probability reduction when position filled
    'want_modifier': 1.5,    # Probability increase when position needed
    'run_modifier': 1.3,     # Probability increase during position runs
    'run_threshold': 3,      # Number of picks to trigger run behavior
    
    # Enhancement 3: Contingency Planning
    'qb_run_threshold': 3,   # QB picks in window to trigger QB panic
    'tier_break_threshold': 0.5,  # Score drop to detect tier break
}

# Fix 5: Configuration indexing validation and conversion functions
def validate_config():
    """Validate configuration parameters with clear indexing rules"""
    # INDEXING RULES:
    # - my_team_idx: 0-based internally (0 to n_teams-1)
    # - Display: 1-based for user communication (1 to n_teams)
    
    assert 0 <= CONFIG['my_team_idx'] < CONFIG['n_teams'], \
        f"my_team_idx must be 0-{CONFIG['n_teams']-1} (0-based), got {CONFIG['my_team_idx']}"
    
    assert CONFIG['n_teams'] > 0, f"n_teams must be positive, got {CONFIG['n_teams']}"
    assert CONFIG['rounds'] > 0, f"rounds must be positive, got {CONFIG['rounds']}"
    assert CONFIG['n_sims'] > 0, f"n_sims must be positive, got {CONFIG['n_sims']}"
    assert CONFIG['top_k'] > 0, f"top_k must be positive, got {CONFIG['top_k']}"
    
    # Weights must sum to 1.0
    total_weight = CONFIG['espn_weight'] + CONFIG['adp_weight']
    assert abs(total_weight - 1.0) < 0.001, \
        f"ESPN and ADP weights must sum to 1.0, got {total_weight}"

def get_display_pick_number():
    """Convert 0-based team index to 1-based display pick number"""
    return CONFIG['my_team_idx'] + 1

def set_team_pick_number(pick_number_1_based):
    """Set team position using 1-based pick number (user-friendly)"""
    assert 1 <= pick_number_1_based <= CONFIG['n_teams'], \
        f"Pick number must be 1-{CONFIG['n_teams']}, got {pick_number_1_based}"
    CONFIG['my_team_idx'] = pick_number_1_based - 1

# Roster requirements (for value calculation)
STARTER_REQUIREMENTS = {
    'QB': 1,
    'RB': 2,
    'WR': 2,
    'TE': 1,
    'FLEX': 1,  # Best remaining RB/WR/TE
    'K': 1,
    'DST': 1
}

# Position value multipliers (based on scarcity and importance)
POSITION_VALUES = {
    'QB': 1.0,   # QBs score more but are replaceable
    'RB': 1.2,   # RB scarcity premium
    'WR': 1.1,   # WR depth but need quality
    'TE': 1.15,  # Elite TE advantage
    'K': 0.5,    # Kickers don't matter in first 7 rounds
    'DST': 0.6,  # DST don't matter in first 7 rounds
}

# Enhancement 2: Strategy Space Exploration
DRAFT_STRATEGIES = {
    'balanced': {'RB': 1.0, 'WR': 1.0, 'TE': 1.0, 'QB': 1.0},
    'zero_rb': {'RB': 0.4, 'WR': 1.4, 'TE': 1.2, 'QB': 1.1},
    'rb_heavy': {'RB': 1.6, 'WR': 0.8, 'TE': 0.9, 'QB': 0.9},
    'hero_rb': {'RB': 1.3, 'WR': 1.1, 'TE': 1.0, 'QB': 0.8},
}

def validate_monte_carlo_state(state):
    """Validate Monte Carlo state JSON with specific error messages"""
    if not isinstance(state, dict):
        raise ValueError("State must be a dictionary")
    
    # Check required fields
    required_fields = ['my_team_idx', 'current_global_pick', 'my_current_roster', 'all_drafted']
    for field in required_fields:
        if field not in state:
            raise ValueError(f"Missing required field: '{field}'")
    
    # Validate team index
    team_idx = state['my_team_idx']
    if not isinstance(team_idx, int):
        raise ValueError(f"'my_team_idx' must be an integer, got {type(team_idx).__name__}")
    if not (0 <= team_idx <= 13):
        raise ValueError(f"'my_team_idx' must be 0-13 for 14-team league, got {team_idx}")
    
    # Validate current pick
    current_pick = state['current_global_pick']
    if not isinstance(current_pick, int):
        raise ValueError(f"'current_global_pick' must be an integer, got {type(current_pick).__name__}")
    if current_pick < 0:
        raise ValueError(f"'current_global_pick' must be non-negative, got {current_pick}")
    
    # Validate roster arrays
    my_roster = state['my_current_roster']
    if not isinstance(my_roster, list):
        raise ValueError(f"'my_current_roster' must be a list, got {type(my_roster).__name__}")
    
    all_drafted = state['all_drafted']
    if not isinstance(all_drafted, list):
        raise ValueError(f"'all_drafted' must be a list, got {type(all_drafted).__name__}")
    
    # Performance bounds validation
    max_draft_size = 224  # 14 teams × 16 rounds
    if len(all_drafted) > max_draft_size:
        raise ValueError(f"Draft list too large: {len(all_drafted)} players (max {max_draft_size})")
    
    # Check for reasonable roster size
    if len(my_roster) > 16:  # Standard fantasy roster limit
        raise ValueError(f"My roster too large: {len(my_roster)} players (max 16)")
    
    # Consistency check
    if len(my_roster) > len(all_drafted):
        raise ValueError("My roster cannot have more players than total drafted")
    
    return True

def load_monte_carlo_state(base_path):
    """Load current draft state from backup_draft.py integration with enhanced security"""
    import os
    
    # Path validation and security
    try:
        base_path = os.path.abspath(base_path)
        if not os.path.isdir(base_path):
            print(f"⚠️ Base path is not a valid directory: {base_path}")
            return None
        
        state_file = os.path.join(base_path, 'data/draft/monte_carlo_state.json')
        state_file = os.path.abspath(state_file)
        
        # Prevent directory traversal attacks
        if not state_file.startswith(base_path):
            print("⚠️ Security error: State file path outside base directory")
            return None
            
    except (OSError, ValueError) as e:
        print(f"⚠️ Path validation error: {e}")
        return None
    
    # Check if state file exists
    if not os.path.exists(state_file):
        print("📡 No Monte Carlo state found - using default configuration")
        return None
    
    # Load and validate state file
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
            
    except FileNotFoundError:
        print("📡 No Monte Carlo state found - using default configuration")
        return None
        
    except json.JSONDecodeError as e:
        print(f"⚠️ Corrupted JSON in state file: {e}")
        print("   Please check the state file or delete it to start fresh")
        return None
        
    except PermissionError:
        print("⚠️ Permission denied reading state file")
        return None
        
    except OSError as e:
        print(f"⚠️ File system error reading state file: {e}")
        return None
    
    # Validate state structure and content
    try:
        validate_monte_carlo_state(state)
        
    except ValueError as e:
        print(f"⚠️ Invalid state file format: {e}")
        print("   Please regenerate the state file from backup_draft.py")
        return None
    
    # Performance warning for large drafts
    draft_size = len(state['all_drafted'])
    if draft_size > 150:
        print(f"⚠️ Large draft detected ({draft_size} players) - performance may be impacted")
    
    # Success - display state information
    print(f"📡 Loaded draft state from backup_draft.py:")
    print(f"   Current pick: #{state['current_global_pick'] + 1}")
    print(f"   Your team: {state.get('team_name', 'Unknown')}")
    print(f"   Your roster: {len(state['my_current_roster'])} players")
    print(f"   Total drafted: {len(state['all_drafted'])} players")
    
    # Update CONFIG with live draft state
    CONFIG['my_team_idx'] = state['my_team_idx']
    
    return state

def softmax(ranks, temperature=5.0):
    """Convert ranks to probabilities using softmax with temperature and error handling"""
    try:
        # Fix 4: Add error handling for numpy operations
        if len(ranks) == 0:
            return np.array([])
        
        # Lower rank = better, so negate for softmax
        scores = -np.array(ranks, dtype=np.float64) / temperature
        
        # Clip extreme values to prevent overflow
        scores = np.clip(scores, -500, 500)
        
        exp_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
        total = exp_scores.sum()
        
        if total == 0 or np.isnan(total) or np.isinf(total):
            # Fallback to uniform distribution
            return np.ones(len(ranks)) / len(ranks)
        
        return exp_scores / total
    except Exception as e:
        print(f"Warning: Softmax calculation failed ({e}), using uniform distribution")
        return np.ones(len(ranks)) / len(ranks)

def load_player_data():
    """Load and merge ESPN rankings, ADP data, and projections with file existence checks"""
    import os
    print("Loading player data...")
    
    # Fix 1: File path dependencies with existence checks and fallbacks
    base_path = '/Users/ben/projects/fantasy-football-draft-spreadsheet-draft-pick-odds'
    espn_file = os.path.join(base_path, 'data/espn_projections_20250814.csv')
    adp_file = os.path.join(base_path, 'data/fantasypros_adp_20250815.csv') 
    proj_file = os.path.join(base_path, 'data/rankings_top300_20250814.csv')
    
    # Load Monte Carlo state if available (from backup_draft.py)
    monte_carlo_state = load_monte_carlo_state(base_path)
    
    # Load ESPN rankings with fallback
    try:
        if os.path.exists(espn_file):
            espn_df = pd.read_csv(espn_file)
        else:
            print(f"Warning: ESPN file not found at {espn_file}, using fallback data")
            espn_df = pd.DataFrame({'player_name': [], 'overall_rank': [], 'position': [], 'team': []})
    except Exception as e:
        print(f"Error loading ESPN data: {e}, using fallback")
        espn_df = pd.DataFrame({'player_name': [], 'overall_rank': [], 'position': [], 'team': []})
    
    espn_df['espn_rank'] = espn_df.get('overall_rank', 300)
    
    # Load ADP data with fallback
    try:
        if os.path.exists(adp_file):
            adp_df = pd.read_csv(adp_file)
        else:
            print(f"Warning: ADP file not found at {adp_file}, using fallback data")
            adp_df = pd.DataFrame({'PLAYER': [], 'RANK': []})
    except Exception as e:
        print(f"Error loading ADP data: {e}, using fallback")
        adp_df = pd.DataFrame({'PLAYER': [], 'RANK': []})
    
    adp_df['adp_rank'] = adp_df.get('RANK', 300)
    adp_df['player_name'] = adp_df.get('PLAYER', '')
    
    # Load projections with fallback
    try:
        if os.path.exists(proj_file):
            proj_df = pd.read_csv(proj_file)
        else:
            print(f"Warning: Projections file not found at {proj_file}, using fallback data")
            proj_df = pd.DataFrame({'PLAYER': [], 'FANTASY_PTS': [], 'POSITION': []})
    except Exception as e:
        print(f"Error loading projections data: {e}, using fallback")
        proj_df = pd.DataFrame({'PLAYER': [], 'FANTASY_PTS': [], 'POSITION': []})
    
    proj_df['player_name'] = proj_df.get('PLAYER', '').astype(str).str.replace(r'\s+[A-Z]{2,3}$', '', regex=True).str.strip()
    proj_df['proj'] = proj_df.get('FANTASY_PTS', 100).fillna(100)
    
    # Merge datasets
    merged = espn_df[['player_name', 'position', 'espn_rank', 'team']].merge(
        adp_df[['player_name', 'adp_rank']], 
        on='player_name', 
        how='outer'
    )
    
    merged = merged.merge(
        proj_df[['player_name', 'proj', 'POSITION']], 
        on='player_name', 
        how='left'
    )
    
    # Clean positions
    merged['pos'] = merged['position'].fillna(merged['POSITION']).fillna('FLEX')
    merged['pos'] = merged['pos'].str.extract(r'([A-Z]+)')[0]
    
    # Fill missing values
    merged['espn_rank'] = merged['espn_rank'].fillna(300)
    merged['adp_rank'] = merged['adp_rank'].fillna(300)
    merged['proj'] = merged['proj'].fillna(50)
    
    # Create unique player_id
    merged['player_id'] = range(len(merged))
    
    # Prepare final dataframe
    players_df = merged[['player_id', 'player_name', 'pos', 'proj', 'espn_rank', 'adp_rank']].copy()
    players_df.columns = ['player_id', 'name', 'pos', 'proj', 'espn_rank', 'adp_rank']
    players_df = players_df.dropna(subset=['name'])
    players_df = players_df.set_index('player_id')
    
    # Apply position value adjustments
    players_df['adj_proj'] = players_df.apply(
        lambda row: row['proj'] * POSITION_VALUES.get(row['pos'], 1.0), 
        axis=1
    )
    
    # Fix 4: Error handling for numpy operations
    # Pre-calculate probabilities for efficiency with error handling
    try:
        # ESPN probabilities
        espn_probs = softmax(players_df['espn_rank'].values, CONFIG['temperature'])
        players_df['espn_prob'] = espn_probs
    except Exception as e:
        print(f"Warning: ESPN softmax failed ({e}), using uniform distribution")
        players_df['espn_prob'] = 1.0 / len(players_df)
    
    try:
        # ADP probabilities
        adp_probs = softmax(players_df['adp_rank'].values, CONFIG['temperature'])
        players_df['adp_prob'] = adp_probs
    except Exception as e:
        print(f"Warning: ADP softmax failed ({e}), using uniform distribution")
        players_df['adp_prob'] = 1.0 / len(players_df)
    
    # Combined probability (80/20 weighted)
    try:
        players_df['pick_prob'] = (
            CONFIG['espn_weight'] * players_df['espn_prob'] + 
            CONFIG['adp_weight'] * players_df['adp_prob']
        )
    except Exception as e:
        print(f"Warning: Probability combination failed ({e}), using uniform distribution")
        players_df['pick_prob'] = 1.0 / len(players_df)
    
    # Filter out already-drafted players if Monte Carlo state is available
    if monte_carlo_state:
        drafted_players = set(monte_carlo_state['all_drafted'])
        
        # Performance bounds: limit drafted players list size
        max_draft_size = 224  # 14 teams × 16 rounds
        if len(drafted_players) > max_draft_size:
            print(f"⚠️ Unusually large draft list detected: {len(drafted_players)} players")
            print("   Truncating to prevent performance issues")
            # Keep only the first max_draft_size players (should be chronological)
            drafted_players = set(list(monte_carlo_state['all_drafted'])[:max_draft_size])
        
        if drafted_players:
            pre_filter_count = len(players_df)
            players_df = players_df[~players_df['name'].isin(drafted_players)]
            filtered_count = pre_filter_count - len(players_df)
            print(f"🔒 Filtered out {filtered_count} already-drafted players")
            
            # Additional performance check
            if filtered_count > 200:
                print("   Large number of drafted players may impact simulation speed")
    
    return players_df

def get_snake_draft_order(n_teams, rounds):
    """Generate snake draft pick order"""
    order = []
    for r in range(rounds):
        if r % 2 == 0:
            order.extend(range(n_teams))
        else:
            order.extend(reversed(range(n_teams)))
    return np.array(order, dtype=np.int32)

# Enhancement 1: Dynamic Opponent Behavior
def get_team_roster_needs(roster_positions):
    """Calculate what positions a team needs based on starter requirements"""
    pos_counts = defaultdict(int)
    for pos, _ in roster_positions:
        pos_counts[pos] += 1
    
    needs = {}
    for pos, required in STARTER_REQUIREMENTS.items():
        if pos == 'FLEX':
            continue  # FLEX is handled separately
        current = pos_counts.get(pos, 0)
        if current < required:
            needs[pos] = CONFIG['want_modifier']
        else:
            needs[pos] = CONFIG['need_modifier']
    
    return needs

def detect_position_run(recent_picks, position):
    """Detect if there's a run on a specific position with bounds checking"""
    # Fix 2: Add bounds checking before array access
    if not recent_picks or len(recent_picks) < CONFIG['run_threshold']:
        return False
    
    # Safe slice with bounds checking
    safe_slice_start = max(0, len(recent_picks) - 10)
    recent_slice = recent_picks[safe_slice_start:]
    recent_pos_count = sum(1 for pos in recent_slice if pos == position)
    return recent_pos_count >= CONFIG['run_threshold']

def apply_dynamic_modifiers(pick_probs, available_indices, top_players, team_rosters, recent_picks, current_round):
    """Apply dynamic probability modifiers based on team needs and draft context"""
    modified_probs = pick_probs.copy()
    
    for i, player_idx in enumerate(available_indices):
        pos = top_players.loc[player_idx, 'pos']
        
        # Round-based modifier (early rounds favor value, late rounds fill needs)
        round_modifier = 1.0
        if current_round <= 3:
            # Early rounds: slightly reduce kicker/defense probability
            if pos in ['K', 'DST']:
                round_modifier = 0.1
        elif current_round >= 6:
            # Late rounds: increase need-based picking
            round_modifier = 1.2
        
        # Position run modifier
        run_modifier = 1.0
        if detect_position_run(recent_picks, pos):
            run_modifier = CONFIG['run_modifier']
        
        # Apply modifiers
        modified_probs[i] *= round_modifier * run_modifier
    
    # Renormalize
    return modified_probs / modified_probs.sum()

def detect_contingencies(drafted_by_round, recent_picks, current_round):
    """Detect draft contingencies that require strategy changes"""
    contingencies = []
    
    # Fix 2: QB run detection with bounds checking
    if recent_picks and len(recent_picks) >= 3:  # Need minimum history
        safe_slice_start = max(0, len(recent_picks) - 8)
        recent_slice = recent_picks[safe_slice_start:]
        qb_count = sum(1 for pos in recent_slice if pos == 'QB')
        if qb_count >= CONFIG['qb_run_threshold']:
            contingencies.append('qb_run')
    
    # Position scarcity by round
    if current_round <= 4:
        rb_count = sum(1 for round_picks in drafted_by_round.values() 
                      for pos in round_picks if pos == 'RB')
        if rb_count > (current_round * CONFIG['n_teams'] * 0.4):  # >40% RBs taken
            contingencies.append('rb_scarce')
    
    return contingencies

def compute_roster_value_fast(roster_positions, position_projections):
    """Fast roster value calculation using position counts"""
    value = 0.0
    pos_counts = defaultdict(int)
    
    # Count positions and sum projections
    for pos, proj in roster_positions:
        pos_counts[pos] += 1
    
    # Calculate starter values
    for pos, proj_list in position_projections.items():
        if pos in ['QB', 'TE', 'K', 'DST']:
            # Single starter positions
            if len(proj_list) > 0:
                value += proj_list[0]
        elif pos == 'RB':
            # 2 RB starters
            value += sum(proj_list[:2])
        elif pos == 'WR':
            # 2 WR starters
            value += sum(proj_list[:2])
    
    # FLEX calculation (best remaining RB/WR/TE)
    flex_candidates = []
    if len(position_projections['RB']) > 2:
        flex_candidates.extend(position_projections['RB'][2:])
    if len(position_projections['WR']) > 2:
        flex_candidates.extend(position_projections['WR'][2:])
    if len(position_projections['TE']) > 1:
        flex_candidates.extend(position_projections['TE'][1:])
    
    if flex_candidates:
        value += max(flex_candidates)
    
    return value

def select_best_player_greedy(available_indices, my_roster_positions, top_players, strategy='balanced', contingencies=None):
    """Greedy selection with strategy weights and contingency handling"""
    if not available_indices:
        return None
    
    # Count current positions
    pos_counts = defaultdict(int)
    for pos, _ in my_roster_positions:
        pos_counts[pos] += 1
    
    # Round-based strategy (don't take QB/K/DST early)
    current_round = len(my_roster_positions) + 1
    
    # Position limits and timing
    if current_round <= 5:
        # First 5 rounds: focus on RB/WR/TE
        valid_positions = ['RB', 'WR', 'TE']
        if current_round >= 4 and pos_counts['QB'] == 0:
            valid_positions.append('QB')  # Can take QB round 4+
    else:
        # Rounds 6-7: Any position
        valid_positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
    
    # Position limits (don't draft too many of one position)
    pos_limits = {'QB': 1, 'RB': 4, 'WR': 4, 'TE': 2, 'K': 1, 'DST': 1}
    
    # Get strategy weights
    strategy_weights = DRAFT_STRATEGIES.get(strategy, DRAFT_STRATEGIES['balanced'])
    
    # Enhancement 3: Contingency handling
    contingency_weights = {}
    if contingencies:
        if 'qb_run' in contingencies and pos_counts['QB'] == 0:
            contingency_weights['QB'] = 2.0  # Panic draft QB
        if 'rb_scarce' in contingencies:
            contingency_weights['RB'] = 1.8  # Reach for RB
    
    # Filter candidates
    candidates = []
    for idx in available_indices:
        pos = top_players.loc[idx, 'pos']
        if pos in valid_positions and pos_counts[pos] < pos_limits.get(pos, 3):
            # Score based on ESPN rank (lower is better) with position adjustment
            rank = top_players.loc[idx, 'espn_rank']
            proj = top_players.loc[idx, 'proj']
            
            # Apply strategy weight
            strategy_mult = strategy_weights.get(pos, 1.0)
            
            # Apply contingency weight if applicable
            contingency_mult = contingency_weights.get(pos, 1.0)
            
            # Combine all factors
            score = (proj * strategy_mult * contingency_mult) / (rank + 10)
            candidates.append((idx, score))
    
    if not candidates:
        # Fallback: best available by ESPN rank
        best_idx = min(available_indices, key=lambda i: top_players.loc[i, 'espn_rank'])
    else:
        # Select best candidate by score
        best_idx = max(candidates, key=lambda x: x[1])[0]
    
    return best_idx

def simulate_draft_fast(sim_idx, top_players, pick_order, rng, strategy='balanced'):
    """Enhanced draft simulation with dynamic behavior and contingency planning"""
    # Fix 3: Memory efficiency - top_players already filtered and passed in
    available_indices = set(top_players.index)
    
    # Pre-calculate normalized probabilities for top players
    pick_probs = top_players['pick_prob'].values
    pick_probs = pick_probs / pick_probs.sum()
    
    my_roster_positions = []  # List of (position, projection) tuples
    my_position_sequence = []  # Just positions for pattern analysis
    
    # Enhancement 1: Track team rosters and recent picks
    team_rosters = {i: [] for i in range(CONFIG['n_teams'])}
    recent_picks = []  # Last 10 picks for run detection
    drafted_by_round = defaultdict(list)  # Track positions by round
    
    for pick_num, team_idx in enumerate(pick_order):
        if not available_indices:
            break
        
        current_round = (pick_num // CONFIG['n_teams']) + 1
        
        # Enhancement 3: Detect contingencies
        contingencies = detect_contingencies(drafted_by_round, recent_picks, current_round)
        
        if team_idx == CONFIG['my_team_idx']:
            # MY PICK: Enhanced strategy with contingency handling
            best_idx = select_best_player_greedy(
                list(available_indices), 
                my_roster_positions, 
                top_players,
                strategy=strategy,
                contingencies=contingencies
            )
            
            if best_idx:
                pos = top_players.loc[best_idx, 'pos']
                proj = top_players.loc[best_idx, 'adj_proj']
                my_roster_positions.append((pos, proj))
                my_position_sequence.append(pos)
                team_rosters[team_idx].append(pos)
                recent_picks.append(pos)
                drafted_by_round[current_round].append(pos)
                # Fix 2: Use discard instead of remove to avoid KeyErrors
                available_indices.discard(best_idx)
        
        else:
            # OPPONENT PICK: Enhanced with dynamic behavior
            available_list = list(available_indices)
            if available_list:
                # Get base probabilities for available players
                avail_probs = np.array([
                    top_players.loc[idx, 'pick_prob'] 
                    for idx in available_list
                ])
                
                # Enhancement 1: Apply dynamic modifiers
                avail_probs = apply_dynamic_modifiers(
                    avail_probs, available_list, top_players, 
                    team_rosters, recent_picks, current_round
                )
                
                # Sample pick
                chosen_idx = rng.choice(available_list, p=avail_probs)
                chosen_pos = top_players.loc[chosen_idx, 'pos']
                team_rosters[team_idx].append(chosen_pos)
                recent_picks.append(chosen_pos)
                drafted_by_round[current_round].append(chosen_pos)
                # Fix 2: Use discard instead of remove to avoid KeyErrors
                available_indices.discard(chosen_idx)
        
        # Fix 2: Keep recent_picks to last 10 with bounds checking
        while len(recent_picks) > 10:
            if recent_picks:  # Additional safety check
                recent_picks.pop(0)
            else:
                break
    
    # Calculate final roster value
    position_projections = defaultdict(list)
    for pos, proj in my_roster_positions:
        position_projections[pos].append(proj)
    
    # Sort projections by value for each position
    for pos in position_projections:
        position_projections[pos].sort(reverse=True)
    
    roster_value = compute_roster_value_fast(my_roster_positions, position_projections)
    
    return {
        'position_sequence': my_position_sequence,
        'roster_value': roster_value,
        'strategy': strategy,
        'contingencies': contingencies if team_idx == CONFIG['my_team_idx'] else [],
        'pattern_2': '-'.join(my_position_sequence[:2]) if len(my_position_sequence) >= 2 else None,
        'pattern_3': '-'.join(my_position_sequence[:3]) if len(my_position_sequence) >= 3 else None,
        'pattern_4': '-'.join(my_position_sequence[:4]) if len(my_position_sequence) >= 4 else None,
        'pattern_7': '-'.join(my_position_sequence[:7]) if len(my_position_sequence) >= 7 else None,
    }

# Enhancement 2: Strategy Space Exploration
def run_strategy_comparison():
    """Run simulations across all strategies to find optimal approach"""
    # Fix 5: Validate configuration before running
    validate_config()
    
    print("=" * 70)
    print("🏈 ENHANCED MONTE CARLO STRATEGY COMPARISON")
    print("=" * 70)
    print(f"Position: Pick #{get_display_pick_number()} in {CONFIG['n_teams']}-team league")
    print(f"Simulations per strategy: {CONFIG['n_sims']}")
    print(f"Total simulations: {CONFIG['n_sims'] * len(DRAFT_STRATEGIES)}")
    print(f"Enhancements: Dynamic Opponent Behavior + Contingency Planning")
    print("")
    
    # Fix 3: Memory efficiency - Load data once globally
    players_df = load_player_data()
    print(f"✓ Loaded {len(players_df)} players")
    print(f"✓ Using top {CONFIG['top_k']} players for simulation")
    
    # Pre-filter top players once for all simulations
    top_players = players_df.nsmallest(CONFIG['top_k'], 'espn_rank')
    print("")
    
    # Generate pick order
    pick_order = get_snake_draft_order(CONFIG['n_teams'], CONFIG['rounds'])
    
    strategy_results = {}
    overall_start_time = time.time()
    
    # Run simulations for each strategy
    for strategy_idx, strategy_name in enumerate(DRAFT_STRATEGIES.keys()):
        print(f"🎯 Testing {strategy_name.upper()} strategy...")
        strategy_weights = DRAFT_STRATEGIES[strategy_name]
        print(f"   Weights: {', '.join([f'{k}={v:.1f}' for k, v in strategy_weights.items()])}")
        
        results = []
        round_positions = defaultdict(lambda: defaultdict(int))
        pattern_values = defaultdict(list)
        contingency_counts = defaultdict(int)
        
        start_time = time.time()
        
        for sim_idx in range(CONFIG['n_sims']):
            if sim_idx % 100 == 0 and sim_idx > 0:
                elapsed = time.time() - start_time
                rate = sim_idx / elapsed
                eta = (CONFIG['n_sims'] - sim_idx) / rate
                print(f"     Progress: {sim_idx}/{CONFIG['n_sims']} ({sim_idx/CONFIG['n_sims']*100:.0f}%) - ETA: {eta:.0f}s")
            
            # Run single simulation with strategy
            rng = np.random.default_rng(42 + sim_idx + strategy_idx * 1000)
            result = simulate_draft_fast(sim_idx, top_players, pick_order, rng, strategy=strategy_name)
            results.append(result)
            
            # Track contingencies
            for contingency in result.get('contingencies', []):
                contingency_counts[contingency] += 1
            
            # Track round-by-round positions
            for round_num, pos in enumerate(result['position_sequence'], 1):
                round_positions[round_num][pos] += 1
            
            # Track pattern values
            for pattern_key in ['pattern_2', 'pattern_3', 'pattern_4', 'pattern_7']:
                if result[pattern_key]:
                    pattern_values[result[pattern_key]].append(result['roster_value'])
        
        elapsed = time.time() - start_time
        avg_value = np.mean([r['roster_value'] for r in results])
        std_value = np.std([r['roster_value'] for r in results])
        
        print(f"   ✅ Completed in {elapsed:.1f}s - Avg Value: {avg_value:.1f} ± {std_value:.1f}")
        
        strategy_results[strategy_name] = {
            'results': results,
            'avg_value': avg_value,
            'std_value': std_value,
            'round_positions': dict(round_positions),
            'pattern_values': dict(pattern_values),
            'contingency_counts': dict(contingency_counts),
            'elapsed_time': elapsed
        }
        print("")
    
    total_elapsed = time.time() - overall_start_time
    print(f"🏆 STRATEGY COMPARISON COMPLETE ({total_elapsed:.1f}s total)")
    print("=" * 70)
    
    return analyze_strategy_comparison(strategy_results)

def run_monte_carlo_strategy_discovery(strategy='balanced'):
    """Single strategy simulation runner"""
    # Fix 5: Validate configuration before running
    validate_config()
    
    print("=" * 70)
    print("🏈 MONTE CARLO 7-ROUND STRATEGY DISCOVERY")
    print("=" * 70)
    print(f"Position: Pick #{get_display_pick_number()} in {CONFIG['n_teams']}-team league")
    print(f"Strategy: {strategy.upper()}")
    print(f"Simulations: {CONFIG['n_sims']}")
    print(f"Strategy Focus: First {CONFIG['rounds']} rounds")
    print(f"Probability Model: {CONFIG['espn_weight']*100:.0f}% ESPN + {CONFIG['adp_weight']*100:.0f}% ADP")
    print("")
    
    # Fix 3: Memory efficiency - Load data once globally
    players_df = load_player_data()
    print(f"✓ Loaded {len(players_df)} players")
    print(f"✓ Using top {CONFIG['top_k']} players for simulation")
    
    # Pre-filter top players once for all simulations
    top_players = players_df.nsmallest(CONFIG['top_k'], 'espn_rank')
    print("")
    
    # Generate pick order
    pick_order = get_snake_draft_order(CONFIG['n_teams'], CONFIG['rounds'])
    
    # Results storage
    results = []
    round_positions = defaultdict(lambda: defaultdict(int))
    pattern_values = defaultdict(list)
    contingency_counts = defaultdict(int)
    
    # Run simulations
    print("Running simulations...")
    start_time = time.time()
    
    for sim_idx in range(CONFIG['n_sims']):
        if sim_idx % 100 == 0:
            elapsed = time.time() - start_time
            if sim_idx > 0:
                rate = sim_idx / elapsed
                eta = (CONFIG['n_sims'] - sim_idx) / rate
                print(f"  Progress: {sim_idx}/{CONFIG['n_sims']} simulations ({sim_idx/CONFIG['n_sims']*100:.0f}%) - ETA: {eta:.0f}s")
        
        # Run single simulation
        rng = np.random.default_rng(42 + sim_idx)
        result = simulate_draft_fast(sim_idx, top_players, pick_order, rng, strategy=strategy)
        results.append(result)
        
        # Track contingencies
        for contingency in result.get('contingencies', []):
            contingency_counts[contingency] += 1
        
        # Track round-by-round positions
        for round_num, pos in enumerate(result['position_sequence'], 1):
            round_positions[round_num][pos] += 1
        
        # Track pattern values
        for pattern_key in ['pattern_2', 'pattern_3', 'pattern_4', 'pattern_7']:
            if result[pattern_key]:
                pattern_values[result[pattern_key]].append(result['roster_value'])
    
    elapsed = time.time() - start_time
    print(f"\n✅ Completed {CONFIG['n_sims']} simulations in {elapsed:.1f} seconds")
    print(f"   Average time per simulation: {elapsed/CONFIG['n_sims']*1000:.1f}ms")
    
    # Show contingency statistics
    if contingency_counts:
        print(f"\n⚠️  CONTINGENCY ACTIVATIONS:")
        for contingency, count in contingency_counts.items():
            print(f"   {contingency}: {count} times ({count/CONFIG['n_sims']*100:.1f}%)")
    
    # Analyze results
    return analyze_results(results, round_positions, pattern_values)

def analyze_results(results, round_positions, pattern_values):
    """Analyze simulation results to find optimal strategies"""
    
    print("\n" + "=" * 70)
    print("📊 STRATEGY ANALYSIS RESULTS")
    print("=" * 70)
    
    # Overall statistics
    all_values = [r['roster_value'] for r in results]
    print(f"\n📈 Overall Performance:")
    print(f"   Average roster value: {np.mean(all_values):.1f} ± {np.std(all_values):.1f}")
    print(f"   Best roster: {np.max(all_values):.1f} points")
    print(f"   Worst roster: {np.min(all_values):.1f} points")
    
    # Round-by-round analysis
    print("\n🎯 ROUND-BY-ROUND POSITION FREQUENCIES:")
    print("-" * 40)
    for round_num in range(1, CONFIG['rounds'] + 1):
        total = sum(round_positions[round_num].values())
        if total > 0:
            freqs = {pos: count/total for pos, count in round_positions[round_num].items()}
            sorted_freqs = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
            print(f"Round {round_num}: " + ", ".join([
                f"{pos} ({freq*100:.0f}%)" 
                for pos, freq in sorted_freqs[:4]
            ]))
    
    # Opening strategy analysis (2 rounds)
    print("\n🏆 TOP OPENING STRATEGIES (Rounds 1-2):")
    print("-" * 40)
    pattern_2_stats = analyze_patterns(pattern_values, 2)
    for i, (pattern, stats) in enumerate(pattern_2_stats[:5], 1):
        print(f"{i}. {pattern}: {stats['freq']*100:.1f}% frequency, "
              f"{stats['avg']:.1f} ± {stats['std']:.1f} points "
              f"(+{stats['avg'] - np.mean(all_values):.1f} vs avg)")
    
    # Core strategy analysis (3 rounds)
    print("\n⚡ BEST 3-ROUND STARTS:")
    print("-" * 40)
    pattern_3_stats = analyze_patterns(pattern_values, 3)
    for i, (pattern, stats) in enumerate(pattern_3_stats[:5], 1):
        print(f"{i}. {pattern}: {stats['freq']*100:.1f}% frequency, "
              f"{stats['avg']:.1f} points")
    
    # Core strategy analysis (4 rounds)
    print("\n🔥 OPTIMAL 4-ROUND CORES:")
    print("-" * 40)
    pattern_4_stats = analyze_patterns(pattern_values, 4)
    for i, (pattern, stats) in enumerate(pattern_4_stats[:5], 1):
        print(f"{i}. {pattern}: {stats['freq']*100:.1f}% frequency, "
              f"{stats['avg']:.1f} points")
    
    # Full 7-round strategy
    print("\n💎 BEST COMPLETE 7-ROUND STRATEGIES:")
    print("-" * 40)
    pattern_7_stats = analyze_patterns(pattern_values, 7)
    for i, (pattern, stats) in enumerate(pattern_7_stats[:5], 1):
        if stats['count'] >= 5:  # Only show patterns with enough samples
            print(f"{i}. {pattern}:")
            print(f"   Frequency: {stats['freq']*100:.1f}% ({stats['count']} occurrences)")
            print(f"   Avg Value: {stats['avg']:.1f} points (+{stats['avg'] - np.mean(all_values):.1f} vs avg)")
    
    # Strategy insights
    print("\n💡 KEY STRATEGIC INSIGHTS:")
    print("-" * 40)
    
    # Best vs worst opening
    if len(pattern_2_stats) >= 2:
        best = pattern_2_stats[0]
        worst = pattern_2_stats[-1]
        print(f"• {best[0]} opening beats {worst[0]} by {best[1]['avg'] - worst[1]['avg']:.1f} points")
    
    # QB timing
    qb_rounds = []
    for round_num, positions in round_positions.items():
        if 'QB' in positions and positions['QB'] / sum(positions.values()) > 0.15:
            qb_rounds.append(round_num)
    if qb_rounds:
        print(f"• Optimal QB timing: Rounds {qb_rounds[:3]}")
    
    # RB vs WR strategy
    rb_first = sum(1 for r in results if r['position_sequence'] and r['position_sequence'][0] == 'RB')
    wr_first = sum(1 for r in results if r['position_sequence'] and r['position_sequence'][0] == 'WR')
    print(f"• First pick breakdown: RB {rb_first/len(results)*100:.0f}%, WR {wr_first/len(results)*100:.0f}%")
    
    # Position run detection
    print(f"• Most common 3-round start: {pattern_3_stats[0][0] if pattern_3_stats else 'N/A'}")
    
    return {
        'summary_stats': {
            'mean_value': np.mean(all_values),
            'std_value': np.std(all_values),
            'max_value': np.max(all_values),
            'min_value': np.min(all_values)
        },
        'round_frequencies': dict(round_positions),
        'best_patterns': {
            '2_round': pattern_2_stats[:5] if pattern_2_stats else [],
            '4_round': pattern_4_stats[:5] if pattern_4_stats else [],
            '7_round': pattern_7_stats[:5] if pattern_7_stats else []
        }
    }

def analyze_patterns(pattern_values, pattern_length):
    """Analyze patterns of specific length"""
    pattern_stats = {}
    
    for pattern, values in pattern_values.items():
        if len(pattern.split('-')) == pattern_length and len(values) >= 3:
            pattern_stats[pattern] = {
                'count': len(values),
                'freq': len(values) / CONFIG['n_sims'],
                'avg': np.mean(values),
                'std': np.std(values),
                'max': np.max(values),
                'min': np.min(values)
            }
    
    # Sort by average value
    sorted_patterns = sorted(
        pattern_stats.items(),
        key=lambda x: x[1]['avg'],
        reverse=True
    )
    
    return sorted_patterns

def analyze_strategy_comparison(strategy_results):
    """Analyze results across all strategies"""
    print("\n📊 STRATEGY COMPARISON RESULTS")
    print("=" * 70)
    
    # Sort strategies by average value
    sorted_strategies = sorted(
        strategy_results.items(),
        key=lambda x: x[1]['avg_value'],
        reverse=True
    )
    
    print("🏆 STRATEGY RANKINGS:")
    print("-" * 40)
    best_avg = sorted_strategies[0][1]['avg_value']
    
    for i, (strategy, data) in enumerate(sorted_strategies, 1):
        diff = data['avg_value'] - best_avg
        diff_str = f"+{diff:.1f}" if diff >= 0 else f"{diff:.1f}"
        print(f"{i}. {strategy.upper()}: {data['avg_value']:.1f} ± {data['std_value']:.1f} ({diff_str})")
        
        # Show top contingencies
        if data['contingency_counts']:
            top_contingency = max(data['contingency_counts'].items(), key=lambda x: x[1])
            print(f"   Most common contingency: {top_contingency[0]} ({top_contingency[1]/CONFIG['n_sims']*100:.0f}%)")
    
    print("\n🎯 STRATEGY INSIGHTS:")
    print("-" * 40)
    
    # Best vs worst
    best = sorted_strategies[0]
    worst = sorted_strategies[-1]
    print(f"• {best[0].upper()} beats {worst[0].upper()} by {best[1]['avg_value'] - worst[1]['avg_value']:.1f} points")
    
    # Most consistent strategy
    most_consistent = min(strategy_results.items(), key=lambda x: x[1]['std_value'])
    print(f"• Most consistent: {most_consistent[0].upper()} (σ={most_consistent[1]['std_value']:.1f})")
    
    # Strategy-specific patterns
    for strategy, data in sorted_strategies[:2]:  # Top 2 strategies
        top_pattern = None
        if data['pattern_values']:
            # Find most successful pattern for this strategy
            pattern_avgs = {
                pattern: np.mean(values) 
                for pattern, values in data['pattern_values'].items()
                if len(values) >= 5
            }
            if pattern_avgs:
                top_pattern = max(pattern_avgs.items(), key=lambda x: x[1])
        
        if top_pattern:
            print(f"• {strategy.upper()} works best with {top_pattern[0]} pattern ({top_pattern[1]:.1f} avg)")
    
    return {
        'strategy_rankings': sorted_strategies,
        'best_strategy': sorted_strategies[0],
        'most_consistent': most_consistent
    }

def main():
    """Main entry point with enhanced options"""
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == 'compare':
            # Run strategy comparison
            results = run_strategy_comparison()
        elif mode in DRAFT_STRATEGIES:
            # Run specific strategy
            results = run_monte_carlo_strategy_discovery(strategy=mode)
        else:
            print(f"Unknown mode: {mode}")
            print(f"Available modes: compare, {', '.join(DRAFT_STRATEGIES.keys())}")
            return None
    else:
        # Default: run balanced strategy
        results = run_monte_carlo_strategy_discovery()
    
    # Save results
    print("\n" + "=" * 70)
    print("✨ Enhanced Monte Carlo analysis complete!")
    print("=" * 70)
    print("\nUsage:")
    print("  python monte_carlo_optimized_7round.py           # Run balanced strategy")
    print("  python monte_carlo_optimized_7round.py compare   # Compare all strategies")
    print("  python monte_carlo_optimized_7round.py zero_rb   # Run specific strategy")
    print("  python monte_carlo_optimized_7round.py rb_heavy  # Run specific strategy")
    print("  python monte_carlo_optimized_7round.py hero_rb   # Run specific strategy")
    
    return results

if __name__ == "__main__":
    results = main()