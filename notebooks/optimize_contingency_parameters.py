#!/usr/bin/env python3
"""
Contingency Parameter Optimization Framework
Tests different threshold values to find optimal triggers for draft contingencies
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import time
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import functions from the main Monte Carlo script
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Base configuration (same as main script)
BASE_CONFIG = {
    'n_teams': 14,
    'rounds': 7,
    'my_team_idx': 4,  # Pick #5 
    'n_sims': 100,  # Reduced for parameter testing
    'top_k': 100,
    'espn_weight': 0.8,
    'adp_weight': 0.2,
    'temperature': 5.0,
}

# Parameter ranges to test
PARAMETER_TESTS = {
    'qb_elite_gone': {
        'description': 'How many elite QBs must be gone before panic-drafting QB',
        'test_values': [1, 2, 3, 4, 5],
        'default': 3,
        'metric': 'qb_value_captured'
    },
    'rb_scarcity_percent': {
        'description': 'What % of picks being RBs triggers scarcity mode',
        'test_values': [0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        'default': 0.40,
        'metric': 'rb_value_captured'
    },
    'wr_run_count': {
        'description': 'How many WRs in last X picks triggers WR run',
        'test_values': [2, 3, 4, 5, 6],
        'default': 4,
        'metric': 'wr_value_captured'
    },
    'te_tier_break': {
        'description': 'How many elite TEs gone before reaching for TE',
        'test_values': [1, 2, 3, 4, 5],
        'default': 3,
        'metric': 'te_value_captured'
    },
    'panic_boost': {
        'description': 'Multiplier when contingency triggers',
        'test_values': [1.2, 1.5, 1.8, 2.0, 2.5, 3.0],
        'default': 2.0,
        'metric': 'total_value'
    }
}

def load_player_data():
    """Load and prepare player data with elite tier identification"""
    base_path = '/Users/ben/projects/fantasy-football-draft-spreadsheet-draft-pick-odds'
    
    # Load data files
    espn_df = pd.read_csv(os.path.join(base_path, 'data/espn_projections_20250814.csv'))
    adp_df = pd.read_csv(os.path.join(base_path, 'data/fantasypros_adp_20250815.csv'))
    proj_df = pd.read_csv(os.path.join(base_path, 'data/rankings_top300_20250814.csv'))
    
    # Process and merge
    espn_df['espn_rank'] = espn_df['overall_rank']
    adp_df['adp_rank'] = adp_df['RANK']
    adp_df['player_name'] = adp_df['PLAYER']
    
    # Clean projections player names
    proj_df['player_name'] = proj_df['PLAYER'].str.replace(r'\s+[A-Z]{2,3}$', '', regex=True).str.strip()
    proj_df['proj'] = proj_df['FANTASY_PTS'].fillna(100)
    
    # Merge all data
    merged = espn_df[['player_name', 'position', 'espn_rank']].merge(
        adp_df[['player_name', 'adp_rank']], on='player_name', how='outer'
    ).merge(
        proj_df[['player_name', 'proj', 'POSITION']], on='player_name', how='left'
    )
    
    # Clean and prepare
    merged['pos'] = merged['position'].fillna(merged['POSITION']).fillna('FLEX')
    merged['pos'] = merged['pos'].str.extract(r'([A-Z]+)')[0]
    merged['espn_rank'] = merged['espn_rank'].fillna(300)
    merged['adp_rank'] = merged['adp_rank'].fillna(300)
    merged['proj'] = merged['proj'].fillna(50)
    
    # Calculate combined rank
    merged['combined_rank'] = 0.8 * merged['espn_rank'] + 0.2 * merged['adp_rank']
    
    # Identify elite tiers (top players at each position)
    for pos in ['QB', 'RB', 'WR', 'TE']:
        pos_players = merged[merged['pos'] == pos].nsmallest(10, 'combined_rank')
        merged.loc[pos_players.index, f'{pos.lower()}_tier'] = range(1, len(pos_players) + 1)
    
    # Fill non-elite with high tier numbers
    for pos in ['QB', 'RB', 'WR', 'TE']:
        merged[f'{pos.lower()}_tier'] = merged[f'{pos.lower()}_tier'].fillna(99)
    
    return merged

def simulate_with_contingency(players_df, params, n_sims=100):
    """Run simulation with specific contingency parameters"""
    results = []
    
    for sim in range(n_sims):
        roster_value = run_single_draft_simulation(players_df, params, seed=sim)
        results.append(roster_value)
    
    return {
        'mean_value': np.mean(results),
        'std_value': np.std(results),
        'max_value': np.max(results),
        'min_value': np.min(results),
        'results': results
    }

def run_single_draft_simulation(players_df, params, seed=42):
    """Simulate a single draft with given contingency parameters"""
    np.random.seed(seed)
    
    # Track draft state
    available = set(players_df.index)
    my_roster = []
    all_drafted = []
    position_counts = defaultdict(lambda: defaultdict(int))  # team -> position -> count
    
    # Generate snake draft order
    pick_order = []
    for round_num in range(BASE_CONFIG['rounds']):
        if round_num % 2 == 0:
            pick_order.extend(range(BASE_CONFIG['n_teams']))
        else:
            pick_order.extend(reversed(range(BASE_CONFIG['n_teams'])))
    
    # Simulate each pick
    for pick_num, team_idx in enumerate(pick_order):
        if not available:
            break
            
        current_round = pick_num // BASE_CONFIG['n_teams'] + 1
        
        if team_idx == BASE_CONFIG['my_team_idx']:
            # MY PICK - Apply contingency logic
            best_player = select_with_contingencies(
                players_df, available, my_roster, all_drafted, 
                position_counts, params, current_round
            )
            
            if best_player:
                my_roster.append(best_player)
                all_drafted.append(best_player)
                position_counts[team_idx][players_df.loc[best_player, 'pos']] += 1
                available.remove(best_player)
        else:
            # OPPONENT PICK - Standard probability-based
            if available:
                avail_list = list(available)
                probs = players_df.loc[avail_list, 'combined_rank'].values
                probs = np.exp(-probs / 5.0)  # Softmax
                probs = probs / probs.sum()
                
                chosen = np.random.choice(avail_list, p=probs)
                all_drafted.append(chosen)
                position_counts[team_idx][players_df.loc[chosen, 'pos']] += 1
                available.remove(chosen)
    
    # Calculate roster value
    return calculate_roster_value(players_df, my_roster)

def select_with_contingencies(players_df, available, my_roster, all_drafted, 
                              position_counts, params, current_round):
    """Select best player considering contingency triggers"""
    
    # Count my positions
    my_pos_counts = defaultdict(int)
    for player_idx in my_roster:
        my_pos_counts[players_df.loc[player_idx, 'pos']] += 1
    
    # Check contingency triggers
    contingencies = detect_contingencies(
        players_df, all_drafted, position_counts, params, current_round
    )
    
    # Score each available player
    best_score = -np.inf
    best_player = None
    
    for player_idx in available:
        player = players_df.loc[player_idx]
        pos = player['pos']
        
        # Base score (projection / rank)
        score = player['proj'] / (player['combined_rank'] + 10)
        
        # Apply contingency boosts
        if 'qb_panic' in contingencies and pos == 'QB':
            score *= params.get('panic_boost', 2.0)
        if 'rb_scarcity' in contingencies and pos == 'RB':
            score *= params.get('panic_boost', 2.0)
        if 'wr_run' in contingencies and pos == 'WR':
            score *= params.get('panic_boost', 2.0)
        if 'te_tier_break' in contingencies and pos == 'TE':
            score *= params.get('panic_boost', 2.0)
        
        # Position limits
        if pos == 'QB' and my_pos_counts['QB'] >= 1:
            score *= 0.1
        if pos == 'TE' and my_pos_counts['TE'] >= 1:
            score *= 0.3
        if pos == 'K' and current_round < 7:
            score *= 0.01
        if pos == 'DST' and current_round < 7:
            score *= 0.01
        
        if score > best_score:
            best_score = score
            best_player = player_idx
    
    return best_player

def detect_contingencies(players_df, all_drafted, position_counts, params, current_round):
    """Detect which contingencies have triggered"""
    contingencies = []
    
    # 1. QB Panic - Elite QBs gone
    drafted_qbs = [p for p in all_drafted if players_df.loc[p, 'pos'] == 'QB']
    elite_qbs_gone = sum(1 for p in drafted_qbs if players_df.loc[p, 'qb_tier'] <= 5)
    if elite_qbs_gone >= params.get('qb_elite_gone', 3):
        contingencies.append('qb_panic')
    
    # 2. RB Scarcity - Too many RBs drafted
    total_picks = len(all_drafted)
    if total_picks > 0:
        rb_picks = sum(1 for p in all_drafted if players_df.loc[p, 'pos'] == 'RB')
        rb_percent = rb_picks / total_picks
        if rb_percent >= params.get('rb_scarcity_percent', 0.40):
            contingencies.append('rb_scarcity')
    
    # 3. WR Run - Recent picks heavily WR
    if len(all_drafted) >= 8:
        recent_8 = all_drafted[-8:]
        wr_count = sum(1 for p in recent_8 if players_df.loc[p, 'pos'] == 'WR')
        if wr_count >= params.get('wr_run_count', 4):
            contingencies.append('wr_run')
    
    # 4. TE Tier Break - Elite TEs disappearing
    drafted_tes = [p for p in all_drafted if players_df.loc[p, 'pos'] == 'TE']
    elite_tes_gone = sum(1 for p in drafted_tes if players_df.loc[p, 'te_tier'] <= 5)
    if elite_tes_gone >= params.get('te_tier_break', 3):
        contingencies.append('te_tier_break')
    
    return contingencies

def calculate_roster_value(players_df, roster_indices):
    """Calculate total value of drafted roster"""
    if not roster_indices:
        return 0
    
    # Group by position
    position_values = defaultdict(list)
    for idx in roster_indices:
        pos = players_df.loc[idx, 'pos']
        proj = players_df.loc[idx, 'proj']
        position_values[pos].append(proj)
    
    # Sort each position by value
    for pos in position_values:
        position_values[pos].sort(reverse=True)
    
    # Calculate starter value
    total_value = 0
    
    # QB: 1 starter
    if 'QB' in position_values and len(position_values['QB']) > 0:
        total_value += position_values['QB'][0]
    
    # RB: 2 starters
    if 'RB' in position_values:
        total_value += sum(position_values['RB'][:2])
    
    # WR: 2 starters
    if 'WR' in position_values:
        total_value += sum(position_values['WR'][:2])
    
    # TE: 1 starter
    if 'TE' in position_values and len(position_values['TE']) > 0:
        total_value += position_values['TE'][0]
    
    # FLEX: Best remaining RB/WR/TE
    flex_candidates = []
    if 'RB' in position_values and len(position_values['RB']) > 2:
        flex_candidates.extend(position_values['RB'][2:])
    if 'WR' in position_values and len(position_values['WR']) > 2:
        flex_candidates.extend(position_values['WR'][2:])
    if 'TE' in position_values and len(position_values['TE']) > 1:
        flex_candidates.extend(position_values['TE'][1:])
    
    if flex_candidates:
        total_value += max(flex_candidates)
    
    return total_value

def optimize_parameter(param_name, param_config, players_df, baseline_results):
    """Test different values for a single parameter"""
    print(f"\n🔬 Testing: {param_config['description']}")
    print("-" * 60)
    
    results = {}
    best_value = None
    best_score = -np.inf
    
    for test_value in param_config['test_values']:
        # Create params with this test value
        params = {param_name: test_value, 'panic_boost': 2.0}
        
        # Run simulations
        sim_results = simulate_with_contingency(players_df, params, n_sims=50)
        results[test_value] = sim_results
        
        # Check if best
        if sim_results['mean_value'] > best_score:
            best_score = sim_results['mean_value']
            best_value = test_value
        
        # Compare to baseline
        improvement = sim_results['mean_value'] - baseline_results['mean_value']
        print(f"  {param_name}={test_value}: {sim_results['mean_value']:.1f} "
              f"(+{improvement:.1f} vs baseline)")
    
    print(f"\n  🏆 Optimal: {param_name}={best_value} "
          f"(+{best_score - baseline_results['mean_value']:.1f} points)")
    
    return best_value, results

def run_optimization():
    """Main optimization routine"""
    print("=" * 70)
    print("🧬 CONTINGENCY PARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"Testing parameters for Pick #{BASE_CONFIG['my_team_idx']+1}")
    print(f"Simulations per parameter: 50")
    print("")
    
    # Load player data
    print("Loading player data...")
    players_df = load_player_data()
    print(f"✓ Loaded {len(players_df)} players")
    
    # Get baseline (no contingencies)
    print("\n📊 Establishing baseline (no contingencies)...")
    baseline_params = {}
    baseline_results = simulate_with_contingency(players_df, baseline_params, n_sims=100)
    print(f"Baseline value: {baseline_results['mean_value']:.1f} ± {baseline_results['std_value']:.1f}")
    
    # Optimize each parameter
    optimal_params = {}
    all_results = {'baseline': baseline_results}
    
    for param_name, param_config in PARAMETER_TESTS.items():
        best_value, param_results = optimize_parameter(
            param_name, param_config, players_df, baseline_results
        )
        optimal_params[param_name] = best_value
        all_results[param_name] = param_results
    
    # Test combined optimal parameters
    print("\n" + "=" * 70)
    print("🎯 TESTING COMBINED OPTIMAL PARAMETERS")
    print("-" * 60)
    
    combined_results = simulate_with_contingency(players_df, optimal_params, n_sims=100)
    improvement = combined_results['mean_value'] - baseline_results['mean_value']
    
    print(f"Combined optimal value: {combined_results['mean_value']:.1f} ± {combined_results['std_value']:.1f}")
    print(f"Improvement vs baseline: +{improvement:.1f} points")
    print(f"Improvement %: {improvement/baseline_results['mean_value']*100:.1f}%")
    
    # Display optimal configuration
    print("\n📋 OPTIMAL CONFIGURATION:")
    print("-" * 60)
    for param_name, value in optimal_params.items():
        desc = PARAMETER_TESTS[param_name]['description']
        print(f"• {param_name}: {value}")
        print(f"  ({desc})")
    
    # Save results
    output = {
        'optimal_params': optimal_params,
        'baseline_value': baseline_results['mean_value'],
        'optimized_value': combined_results['mean_value'],
        'improvement': improvement,
        'parameter_tests': {
            param: {
                'best_value': optimal_params[param],
                'description': PARAMETER_TESTS[param]['description']
            }
            for param in optimal_params
        }
    }
    
    with open('optimal_contingency_params.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n✅ Results saved to optimal_contingency_params.json")
    
    return optimal_params, all_results

if __name__ == "__main__":
    optimal_params, results = run_optimization()
    
    print("\n" + "=" * 70)
    print("💡 KEY FINDINGS:")
    print("-" * 60)
    print("These parameters are now empirically optimized based on")
    print("maximizing your roster's total fantasy points.")
    print("")
    print("Use these values in your main Monte Carlo script for")
    print("better draft recommendations!")
    print("=" * 70)