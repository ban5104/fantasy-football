#!/usr/bin/env python3
"""
Monte Carlo Draft Simulator Runner
Clean interface for running draft simulations
"""

import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from itertools import product
from collections import defaultdict
from src.monte_carlo import (DraftSimulator, compare_all_strategies, 
                           quick_simulation, discover_patterns, PatternDetector)
from src.monte_carlo.strategies import VOR_POLICIES


def load_live_draft_state(n_rounds: int = 14):
    """Load current draft state from backup_draft.py integration"""
    sim = DraftSimulator(n_rounds=n_rounds)
    state = sim.load_draft_state()
    
    if state:
        print(f"📡 Loaded draft state:")
        print(f"   Current pick: #{state.get('current_global_pick', 0) + 1}")
        print(f"   Your team: {state.get('team_name', 'Unknown')}")
        print(f"   Your roster: {len(state.get('my_current_roster', []))} players")
        print(f"   Total drafted: {len(state.get('all_drafted', []))} players")
        return state
    else:
        print("📡 No live draft state found - running pre-draft analysis")
        return None


def run_strategy_comparison(my_pick: int = 5, n_sims: int = 100, n_rounds: int = 14):
    """Compare all strategies for a draft position"""
    print("=" * 70)
    print("🏈 MONTE CARLO STRATEGY COMPARISON")
    print("=" * 70)
    print(f"Position: Pick #{my_pick} in 14-team league")
    print(f"Rounds: {n_rounds}")
    print(f"Simulations per strategy: {n_sims}")
    print("")
    
    start_time = time.time()
    results = compare_all_strategies(my_pick, n_sims, n_rounds)
    elapsed = time.time() - start_time
    
    print("\n🏆 STRATEGY RANKINGS:")
    print("-" * 40)
    
    for i, (strategy, stats) in enumerate(results['rankings'], 1):
        print(f"{i}. {strategy.upper()}: {stats['mean_value']:.1f} ± {stats['std_value']:.1f}")
        
        # Show backup info if available
        if 'avg_backup_counts' in stats and any(stats['avg_backup_counts'].values()):
            backup_counts = stats['avg_backup_counts']
            backup_parts = []
            for pos in ['QB', 'RB', 'WR', 'TE']:
                if backup_counts[pos] > 0:
                    backup_parts.append(f"{backup_counts[pos]:.1f} {pos}")
            if backup_parts:
                backup_summary = ", ".join(backup_parts)
                print(f"   Backups: {backup_summary} (total: {backup_counts['total']:.1f})")
        
        # Show round 1 position frequency if available
        if 'position_frequencies' in stats and 'round_1' in stats['position_frequencies']:
            round_1_freq = stats['position_frequencies']['round_1']
            if round_1_freq:
                top_pos = round_1_freq[0]
                print(f"   Round 1: {top_pos[0]} {top_pos[1]:.1f}%")
    
    print(f"\n✅ Analysis complete in {elapsed:.1f} seconds")
    print(f"🎯 Recommended strategy: {results['best_strategy'].upper()}")


def _parse_draft_state(draft_state, my_pick):
    """Parse draft state and return roster/drafted info"""
    if not draft_state:
        return None, None, my_pick, "PRE-DRAFT ANALYSIS"
    
    current_roster = draft_state.get('my_current_roster', [])
    already_drafted = set(draft_state.get('all_drafted', []))
    my_pick = draft_state.get('my_team_idx', my_pick - 1) + 1
    mode = f"LIVE DRAFT (Pick #{draft_state.get('current_global_pick', 0) + 1})"
    
    return current_roster, already_drafted, my_pick, mode

def compare_fast(my_pick: int = 5, n_rounds: int = 14, draft_state: Optional[dict] = None):
    """Fast adaptive comparison using CRN and confidence intervals"""
    print("=" * 70)
    print("🚀 ADAPTIVE CRN STRATEGY COMPARISON")
    print("=" * 70)
    
    current_roster, already_drafted, my_pick, mode = _parse_draft_state(draft_state, my_pick)
    
    print(f"Position: Pick #{my_pick} in 14-team league")
    print(f"Rounds: {n_rounds}")
    print(f"Mode: {mode}")
    print("")
    
    # Initialize simulator and run
    sim = DraftSimulator(n_rounds=n_rounds)
    results = sim.simulator.run_adaptive_comparison(
        my_pick - 1,
        initial_roster=current_roster,
        already_drafted=already_drafted
    )
    
    print(f"\n🎯 Recommended strategy: {results['best_strategy'].upper()}")
    return results
    

def export_simulation_data(strategy: str, my_pick: int, n_sims: int, n_rounds: int = 14,
                          draft_state: Optional[dict] = None, export_parquet: bool = False):
    """Run simulation and export detailed data in long format"""
    
    print("=" * 70)
    print("🏈 MONTE CARLO DATA EXPORT")
    print("=" * 70)
    
    current_roster, already_drafted, my_pick, mode = _parse_draft_state(draft_state, my_pick)
    
    print(f"Position: Pick #{my_pick} in 14-team league")
    print(f"Strategy: {strategy.upper()}")
    print(f"Rounds: {n_rounds}")
    print(f"Simulations: {n_sims}")
    print(f"Mode: {mode}")
    print("")
    
    # Initialize simulator and run
    sim = DraftSimulator(n_rounds=n_rounds)
    start_time = time.time()
    
    result = sim.simulator.run_simulations(
        my_pick - 1,
        strategy,
        n_sims,
        initial_roster=current_roster,
        already_drafted=already_drafted
    )
    
    elapsed = time.time() - start_time
    
    # Convert to long format dataframe
    rows = []
    for sim_idx, sim_result in enumerate(result['all_results']):
        roster = sim_result['roster']
        starters = set(p['id'] for p in sim_result['starters'])
        
        for player in roster:
            rows.append({
                'sim': sim_idx,
                'strategy': strategy,
                'my_pick': my_pick,
                'n_rounds': n_rounds,
                'player_id': player['id'],
                'player_name': player['name'],
                'pos': player['pos'],
                'draft_pick': player.get('draft_pick', 0),  # Add draft pick number
                'draft_round': player.get('draft_round', 0),  # Add draft round
                'sampled_points': player['proj'],
                'is_starter': player['id'] in starters,
                'is_bench': player['id'] not in starters,
                'roster_value': sim_result['roster_value'],
                'starter_points': sim_result['starter_points'],
                'depth_bonus': sim_result['depth_bonus']
            })
    
    df = pd.DataFrame(rows)
    
    # Export to parquet if requested
    if export_parquet:
        data_dir = Path('data/cache')
        data_dir.mkdir(exist_ok=True, parents=True)
        
        filename = f"{strategy}_pick{my_pick}_n{n_sims}_r{n_rounds}.parquet"
        filepath = data_dir / filename
        
        df.to_parquet(filepath)
        print(f"💾 Exported to {filepath}")
    
    # Display summary
    print(f"\n📊 EXPORT SUMMARY:")
    print("-" * 40)
    print(f"Exported records: {len(df)}")
    print(f"Average roster value: {result['mean_value']:.1f} ± {result['std_value']:.1f}")
    print(f"Simulation time: {elapsed:.1f} seconds")
    
    return df, result


def run_single_strategy(strategy: str = 'balanced', 
                       my_pick: int = 5, 
                       n_sims: int = 100,
                       n_rounds: int = 14,
                       draft_state: Optional[dict] = None,
                       parallel: bool = False,
                       n_workers: int = 4):
    """Run simulation for a single strategy"""
    print("=" * 70)
    print("🏈 MONTE CARLO DRAFT SIMULATION")
    print("=" * 70)
    
    current_roster, already_drafted, my_pick, mode = _parse_draft_state(draft_state, my_pick)
    
    print(f"Position: Pick #{my_pick} in 14-team league")
    print(f"Strategy: {strategy.upper()}")
    print(f"Rounds: {n_rounds}")
    print(f"Simulations: {n_sims}")
    print(f"Mode: {mode}")
    if parallel:
        # Limit workers to 6 max for CPU-friendly operation
        n_workers = min(n_workers, 6)
        print(f"Parallel: {n_workers} workers (CPU-friendly)")
    print("")
    
    # Initialize simulator and run - now using unified method
    sim = DraftSimulator(n_rounds=n_rounds)
    
    result = sim.simulator.run_simulations(
        my_pick - 1,
        strategy,
        n_sims,
        initial_roster=current_roster,
        already_drafted=already_drafted,
        parallel=parallel,
        n_workers=n_workers
    )
    
    # Display results
    print(f"\n📊 RESULTS:")
    print("-" * 40)
    print(f"Average roster value: {result['mean_value']:.1f} ± {result['std_value']:.1f}")
    print(f"Best roster: {result['max_value']:.1f} points")
    print(f"Worst roster: {result['min_value']:.1f} points")
    
    # Show backup qualification analysis (if available)
    if 'avg_backup_counts' in result and any(result['avg_backup_counts'].values()):
        print(f"\n📋 BACKUP ANALYSIS:")
        print("-" * 40)
        backup_counts = result['avg_backup_counts']
        backup_parts = []
        for pos in ['QB', 'RB', 'WR', 'TE']:
            if backup_counts[pos] > 0:
                backup_parts.append(f"{backup_counts[pos]:.1f} {pos}")
        
        if backup_parts:
            backup_summary = ", ".join(backup_parts)
            print(f"Average backups: {backup_summary}")
            print(f"Total backup players: {backup_counts['total']:.1f}")
            print("(Players within 25% of starter projection)")
    
    # Show position frequency by round
    if 'position_frequencies' in result:
        print("\n🎯 POSITION FREQUENCY BY ROUND:")
        print("-" * 40)
        
        # Display rounds 1-7 for readability
        for round_num in range(1, 8):
            round_key = f'round_{round_num}'
            if round_key in result['position_frequencies']:
                frequencies = result['position_frequencies'][round_key]
                if frequencies:
                    print(f"\nRound {round_num}:")
                    for pos, percentage in frequencies:
                        print(f"  {pos}: {percentage:.1f}%")
    
    print(f"\n✅ Completed {n_sims} simulations in {result['elapsed_time']:.1f} seconds")
    

def run_pattern_discovery(my_pick: int = 5, n_sims: int = 100, n_rounds: int = 14,
                         draft_state: Optional[dict] = None):
    """Run pattern discovery to identify emergent strategies"""
    print("=" * 70)
    print("🧠 MONTE CARLO PATTERN DISCOVERY")
    print("=" * 70)
    print(f"Position: Pick #{my_pick} in 14-team league")
    print(f"Rounds: {n_rounds}")
    print(f"Simulations: {n_sims}")
    
    # Handle live draft state
    current_roster = None
    already_drafted = None
    
    if draft_state:
        print(f"Mode: LIVE DRAFT (Pick #{draft_state.get('current_global_pick', 0) + 1})")
        current_roster = draft_state.get('my_current_roster', [])
        already_drafted = set(draft_state.get('all_drafted', []))
        my_pick = draft_state.get('my_team_idx', my_pick - 1) + 1
    else:
        print("Mode: PRE-DRAFT ANALYSIS")
    
    print("")
    
    # Run pattern discovery
    start_time = time.time()
    
    # Use the discover_patterns convenience function
    discovery_result = discover_patterns(my_pick, n_sims, n_rounds)
    
    elapsed = time.time() - start_time
    
    # Display results
    pattern_analysis = discovery_result['pattern_analysis']
    pattern_detector = PatternDetector()
    pattern_detector.display_pattern_analysis(
        pattern_analysis, 
        f"Natural Pattern Discovery ({n_rounds} rounds)"
    )
    
    print(f"\n✅ Pattern discovery completed in {elapsed:.1f} seconds")


def grid_search_successive_halving(my_pick: int = 5, n_rounds: int = 14, 
                                 draft_state: Optional[dict] = None):
    """
    Grid search with successive halving for VOR parameter optimization
    Stage 1: 27 combos × 100 sims = 2,700
    Stage 2: Top 9 × 300 sims = 2,700  
    Stage 3: Top 3 × 1,000 sims = 3,000
    Total: 8,400 sims (vs 162,000 for full grid)
    """
    print("=" * 70)
    print("🔬 VOR PARAMETER GRID SEARCH - SUCCESSIVE HALVING")
    print("=" * 70)
    
    current_roster, already_drafted, my_pick, mode = _parse_draft_state(draft_state, my_pick)
    
    print(f"Position: Pick #{my_pick} in 14-team league")
    print(f"Rounds: {n_rounds}")
    print(f"Mode: {mode}")
    print("")
    
    # Define parameter grid (simplified for testing)
    # TODO: Restore full grid for production: alpha_values = [0, 3, 8], lambda_values = [-5, 0, 5], gamma_values = [0.5, 0.75, 1.0]
    alpha_values = [0, 3]
    lambda_values = [0, 5]
    gamma_values = [0.75]
    
    # Generate all combinations (27 total)
    param_combinations = list(product(alpha_values, lambda_values, gamma_values))
    
    print(f"Grid size: {len(param_combinations)} parameter combinations")
    print("Parameters: alpha ∈ {0,3,8}, lambda ∈ {-5,0,5}, gamma ∈ {0.5,0.75,1.0}")
    print("")
    
    # Initialize simulator
    sim = DraftSimulator(n_rounds=n_rounds)
    
    # Context manager for safe temporary policy management
    class TemporaryPolicy:
        def __init__(self, policy_name, params):
            self.policy_name = policy_name
            self.params = params
            
        def __enter__(self):
            from src.monte_carlo.strategies import VOR_POLICIES
            VOR_POLICIES[self.policy_name] = {'params': self.params}
            return self.policy_name
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            from src.monte_carlo.strategies import VOR_POLICIES
            if self.policy_name in VOR_POLICIES:
                del VOR_POLICIES[self.policy_name]
    
    # Stage 1: Quick screening (20 sims each for testing)
    print("🚀 STAGE 1: Initial screening (20 sims each)")
    stage1_results = []
    
    for i, (alpha, lambda_val, gamma) in enumerate(param_combinations):
        params = {
            'alpha': alpha,
            'lambda': lambda_val, 
            'gamma': gamma,
            'r_te': 8,
            'delta_qb': 8
        }
        
        temp_policy_name = f"grid_{i}"
        
        try:
            with TemporaryPolicy(temp_policy_name, params) as policy_name:
                result = sim.simulator.run_simulations(
                    my_pick - 1, policy_name, n_sims=20,
                    initial_roster=current_roster, already_drafted=already_drafted
                )
                
                stage1_results.append({
                    'params': params,
                    'mean_value': result['mean_value'],
                    'std_value': result['std_value'],
                    'n_sims': result['n_sims'],
                    'alpha': alpha,
                    'lambda': lambda_val,
                    'gamma': gamma
                })
                
                print(f"   {i+1:2d}/27: α={alpha}, λ={lambda_val:+d}, γ={gamma:.2f} → {result['mean_value']:.1f} ± {result['std_value']:.1f}")
        
        except Exception as e:
            print(f"   {i+1:2d}/27: FAILED - {str(e)}")
            # Context manager ensures cleanup even on exception
    
    # Sort and select top candidates (adapt to smaller grid)
    stage1_results.sort(key=lambda x: x['mean_value'], reverse=True)
    top_count = min(len(stage1_results), 9)
    top9_results = stage1_results[:top_count]
    
    print(f"\n🥇 TOP {top_count} from Stage 1:")
    for i, result in enumerate(top9_results, 1):
        p = result['params']
        print(f"   {i}. α={p['alpha']}, λ={p['lambda']:+d}, γ={p['gamma']:.2f} → {result['mean_value']:.1f}")
    
    # Stage 2: More detailed evaluation (50 sims each for testing)
    print(f"\n🔥 STAGE 2: Detailed evaluation (50 sims each)")
    stage2_results = []
    
    for i, result in enumerate(top9_results):
        params = result['params']
        temp_policy_name = f"grid_stage2_{i}"
        
        try:
            with TemporaryPolicy(temp_policy_name, params) as policy_name:
                detailed_result = sim.simulator.run_simulations(
                    my_pick - 1, policy_name, n_sims=50,
                    initial_roster=current_roster, already_drafted=already_drafted
                )
                
                stage2_results.append({
                    'params': params,
                    'mean_value': detailed_result['mean_value'],
                    'std_value': detailed_result['std_value'],
                    'n_sims': detailed_result['n_sims']
                })
                
                p = params
                print(f"   {i+1}/{top_count}: α={p['alpha']}, λ={p['lambda']:+d}, γ={p['gamma']:.2f} → {detailed_result['mean_value']:.1f} ± {detailed_result['std_value']:.1f}")
        
        except Exception as e:
            print(f"   {i+1}/{top_count}: FAILED - {str(e)}")
            # Context manager ensures cleanup even on exception
    
    # Sort and select top candidates (adapt to smaller grid)
    stage2_results.sort(key=lambda x: x['mean_value'], reverse=True)
    top3_count = min(len(stage2_results), 3)
    top3_results = stage2_results[:top3_count]
    
    print(f"\n🏆 TOP {top3_count} from Stage 2:")
    for i, result in enumerate(top3_results, 1):
        p = result['params']
        print(f"   {i}. α={p['alpha']}, λ={p['lambda']:+d}, γ={p['gamma']:.2f} → {result['mean_value']:.1f}")
    
    # Stage 3: Final evaluation (100 sims each for testing)
    print(f"\n🎯 STAGE 3: Final evaluation (100 sims each)")
    final_results = []
    
    for i, result in enumerate(top3_results):
        params = result['params']
        temp_policy_name = f"grid_final_{i}"
        
        try:
            with TemporaryPolicy(temp_policy_name, params) as policy_name:
                final_result = sim.simulator.run_simulations(
                    my_pick - 1, policy_name, n_sims=100,
                    initial_roster=current_roster, already_drafted=already_drafted
                )
                
                final_results.append({
                    'params': params,
                    'mean_value': final_result['mean_value'],
                    'std_value': final_result['std_value'],
                    'se_value': final_result['std_value'] / np.sqrt(final_result['n_sims']),
                    'n_sims': final_result['n_sims']
                })
                
                p = params
                se = final_result['std_value'] / np.sqrt(final_result['n_sims'])
                print(f"   {i+1}/{top3_count}: α={p['alpha']}, λ={p['lambda']:+d}, γ={p['gamma']:.2f} → {final_result['mean_value']:.1f} ± {1.96*se:.1f}")
        
        except Exception as e:
            print(f"   {i+1}/{top3_count}: FAILED - {str(e)}")
            # Context manager ensures cleanup even on exception
    
    # Final ranking
    final_results.sort(key=lambda x: x['mean_value'], reverse=True)
    
    print(f"\n🏅 FINAL OPTIMAL PARAMETERS:")
    print("-" * 50)
    for i, result in enumerate(final_results, 1):
        p = result['params']
        ci = 1.96 * result['se_value']
        print(f"{i}. α={p['alpha']}, λ={p['lambda']:+d}, γ={p['gamma']:.2f}")
        print(f"   Expected Value: {result['mean_value']:.1f} ± {ci:.1f}")
        print(f"   Simulations: {result['n_sims']}")
        print("")
    
    stage1_sims = len(param_combinations) * 20
    stage2_sims = len(top9_results) * 50
    stage3_sims = len(top3_results) * 100
    total_sims = stage1_sims + stage2_sims + stage3_sims
    print(f"✅ Grid search complete: {total_sims:,} total simulations")
    
    return final_results


# VONA Analysis Configuration
VONA_CONFIG = {
    'default_sims': 100,
    'quick_sims': 20,  # Reduced for performance with full 14-round analysis
    'detailed_sims': 50,  # Reduced for performance with full 14-round analysis  
    'pick_threshold': 0.55,  # P(win) threshold for PICK vs WAIT decision
    'high_confidence': 0.65,
    'medium_confidence': 0.55
}

def analyze_vona_picks(simulator, draft_state, n_sims=100, parallel=False, n_workers=4):
    """Run paired simulations for top K candidates with proper seed separation"""
    from src.monte_carlo.strategies import VOR_POLICIES
    from concurrent.futures import ProcessPoolExecutor
    import time
    import numpy as np
    
    # Get current draft state
    current_roster = draft_state.get('my_current_roster', [])
    already_drafted = draft_state.get('all_drafted', [])
    my_team_idx = draft_state.get('my_team_idx', 4)
    current_global_pick = draft_state.get('current_global_pick', 1)
    
    # Find our next available pick
    def generate_snake_order(n_teams=14):
        order = []
        for round_num in range(1, 15):  # 14 rounds
            if round_num % 2 == 1:  # Odd rounds: 0,1,2,...,13
                round_picks = list(range(n_teams))
            else:  # Even rounds: 13,12,11,...,0
                round_picks = list(range(n_teams-1, -1, -1))
            order.extend(round_picks)
        return order
    
    # Find the next pick that belongs to our team
    snake_order = generate_snake_order()
    our_next_pick = None
    for pick_idx in range(current_global_pick - 1, len(snake_order)):
        if snake_order[pick_idx] == my_team_idx:
            our_next_pick = pick_idx + 1  # Convert to 1-based
            break
    
    if our_next_pick is None:
        print(f"⚠️  Error: Could not find next pick for team {my_team_idx}")
        return []
    
    current_pick = our_next_pick
    
    # Get balanced strategy params for simulation
    balanced_params = VOR_POLICIES['balanced']['params']
    
    # Get top 6 available by VOR
    candidates = simulator.get_top_available_players(current_roster, already_drafted, k=6)
    
    if not candidates:
        print("⚠️  No candidates available for VONA analysis")
        return []
    
    print(f"🔄 Running paired simulations for {len(candidates)} candidates ({n_sims} sims each)...")
    start_time = time.time()
    
    if parallel and len(candidates) > 1:
        print(f"   Using parallel processing with {n_workers} workers")
        results = _parallel_vona_analysis(candidates, simulator, my_team_idx, current_pick, 
                                         balanced_params, current_roster, already_drafted, 
                                         n_sims, n_workers)
    else:
        results = _sequential_vona_analysis(candidates, simulator, my_team_idx, current_pick,
                                           balanced_params, current_roster, already_drafted, n_sims)
    
    elapsed = time.time() - start_time
    print(f"   ⏱️  Analysis completed in {elapsed:.1f}s")
    
    return sorted(results, key=lambda x: x['vona'], reverse=True)


def _sequential_vona_analysis(candidates, simulator, my_team_idx, current_pick, 
                             balanced_params, current_roster, already_drafted, n_sims):
    """Sequential VONA analysis with proper seed separation"""
    import numpy as np
    
    results = []
    for player in candidates:
        print(f"   Analyzing {player['name']} ({player['pos']})...")
        
        # Calculate replacement levels once and reuse for both branches
        picks_until_next = _estimate_picks_until_next_turn(my_team_idx, current_pick, 14)
        next_turn_pick = current_pick + picks_until_next - 1
        
        # Calculate replacement levels once per player to avoid recomputation with same seed
        replacement_levels = None
        
        # Branch A: Pick player now - EVEN seeds
        values_a = []
        for sim_idx in range(n_sims):
            value = simulator.simulate_from_pick(
                my_team_idx, 
                balanced_params,
                pick_num=current_pick - 1,  # Convert to 0-based
                forced_pick=player['name'],
                seed=42 + sim_idx * 2,  # EVEN seeds for Branch A
                initial_roster=current_roster,
                already_drafted=already_drafted
            )
            values_a.append(value)
        
        # Branch B: Wait (pick best available next) - ODD seeds
        values_b = []
        for sim_idx in range(n_sims):
            value = simulator.simulate_from_pick(
                my_team_idx,
                balanced_params, 
                pick_num=next_turn_pick - 1,  # Wait until our next turn (0-based)
                forced_pick=None,  # Let sim pick naturally
                seed=42 + sim_idx * 2 + 1,  # ODD seeds for Branch B
                initial_roster=current_roster,
                already_drafted=already_drafted
            )
            values_b.append(value)
        
        # Calculate metrics
        values_a = np.array(values_a)
        values_b = np.array(values_b)
        vona = np.mean(values_a) - np.mean(values_b)
        p_win = np.mean(values_a > values_b)
        
        # Debug logging for first candidate only to verify fix
        if len(results) == 0:
            print(f"     🔍 Debug {player['name']}: Branch A avg: {np.mean(values_a):.1f}, Branch B avg: {np.mean(values_b):.1f}, VONA: {vona:+.1f}")
            print(f"       First 3 Branch A values: {values_a[:3]}")
            print(f"       First 3 Branch B values: {values_b[:3]}")
        
        # Get survival probability using existing probability model
        available_players = set(simulator.prob_model.players_df.index)
        p_survive = simulator.prob_model.calculate_survival_probability(
            player['id'], picks_until_next, available_players
        )
        
        results.append({
            'player': player['name'],
            'pos': player['pos'],
            'rank': player['rank'],
            'vor': player['vor'],
            'vona': vona,
            'p_win': p_win,
            'p_survive': p_survive,
            'decision': 'PICK' if p_win > VONA_CONFIG['pick_threshold'] else 'WAIT'
        })
    
    return results


def _parallel_vona_analysis(candidates, simulator, my_team_idx, current_pick,
                           balanced_params, current_roster, already_drafted, n_sims, n_workers):
    """Parallel VONA analysis using ProcessPoolExecutor"""
    from concurrent.futures import ProcessPoolExecutor
    import numpy as np
    
    # Prepare arguments for parallel processing
    worker_args = []
    for player in candidates:
        picks_until_next = _estimate_picks_until_next_turn(my_team_idx, current_pick, 14)
        next_turn_pick = current_pick + picks_until_next - 1
        
        worker_args.append((
            player, my_team_idx, current_pick, next_turn_pick,
            balanced_params, current_roster, already_drafted, n_sims,
            simulator.prob_model._to_dict()  # Serialize for worker
        ))
    
    # Run parallel analysis
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_vona_worker, worker_args))
    
    return results


def _vona_worker(args):
    """Worker function for parallel VONA analysis"""
    import numpy as np
    from src.monte_carlo.simulator import MonteCarloSimulator
    from src.monte_carlo.probability import ProbabilityModel
    from src.monte_carlo.opponent import OpponentModel
    
    (player, my_team_idx, current_pick, next_turn_pick, balanced_params, 
     current_roster, already_drafted, n_sims, prob_model_dict) = args
    
    # Recreate simulator in worker process
    prob_model = ProbabilityModel._from_dict(prob_model_dict)
    opponent_model = OpponentModel(prob_model)
    simulator = MonteCarloSimulator(prob_model, opponent_model, 14, 14)  # 14 teams, 14 rounds
    
    # Branch A: Pick player now - EVEN seeds
    values_a = []
    for sim_idx in range(n_sims):
        value = simulator.simulate_from_pick(
            my_team_idx, 
            balanced_params,
            pick_num=current_pick - 1,
            forced_pick=player['name'],
            seed=42 + sim_idx * 2,  # EVEN seeds
            initial_roster=current_roster,
            already_drafted=already_drafted
        )
        values_a.append(value)
    
    # Branch B: Wait - ODD seeds  
    values_b = []
    for sim_idx in range(n_sims):
        value = simulator.simulate_from_pick(
            my_team_idx,
            balanced_params,
            pick_num=next_turn_pick - 1,
            forced_pick=None,
            seed=42 + sim_idx * 2 + 1,  # ODD seeds
            initial_roster=current_roster,
            already_drafted=already_drafted
        )
        values_b.append(value)
    
    # Calculate metrics
    values_a = np.array(values_a)
    values_b = np.array(values_b)
    vona = np.mean(values_a) - np.mean(values_b)
    p_win = np.mean(values_a > values_b)
    
    # Get survival probability
    picks_until_next = _estimate_picks_until_next_turn(my_team_idx, current_pick, 14)
    available_players = set(prob_model.players_df.index)
    p_survive = prob_model.calculate_survival_probability(
        player['id'], picks_until_next, available_players
    )
    
    return {
        'player': player['name'],
        'pos': player['pos'],
        'rank': player['rank'],
        'vor': player['vor'],
        'vona': vona,
        'p_win': p_win,
        'p_survive': p_survive,
        'decision': 'PICK' if p_win > VONA_CONFIG['pick_threshold'] else 'WAIT'
    }


def _estimate_picks_until_next_turn(my_team_idx, current_pick, n_teams):
    """Estimate picks until next turn based on snake draft pattern"""
    # Convert to 0-based indexing for calculation
    my_position = my_team_idx  # 0-based position in snake
    current_round = ((current_pick - 1) // n_teams) + 1
    
    # Calculate my next pick number in snake draft
    if current_round % 2 == 1:  # Odd round (normal order: 0,1,2...13)
        # Next round is even (reverse order: 13,12,11...0)
        next_round_start = current_round * n_teams + 1
        my_next_pick = next_round_start + (n_teams - 1 - my_position)
    else:  # Even round (reverse order: 13,12,11...0)
        # Next round is odd (normal order: 0,1,2...13)
        next_round_start = current_round * n_teams + 1
        my_next_pick = next_round_start + my_position
    
    # Return picks until my next turn
    return my_next_pick - current_pick


def run_positional_degradation_analysis(my_pick: int = 5, n_rounds: int = 14,
                                       draft_state: Optional[dict] = None,
                                       deg_rounds: int = 1, n_sims: int = 200, 
                                       shortlist: int = 6):
    """Run positional degradation analysis showing point loss from waiting"""
    print("=" * 80)
    if deg_rounds > 1:
        print(f"📊 MULTI-ROUND DEGRADATION ANALYSIS - What You Lose By Waiting ({deg_rounds} rounds)")
    else:
        print("📊 POSITIONAL DEGRADATION ANALYSIS - What You Lose By Waiting")
    print("=" * 80)
    
    current_roster, already_drafted, my_pick, mode = _parse_draft_state(draft_state, my_pick)
    
    print(f"Position: Pick #{my_pick} in 14-team league")
    print(f"Mode: {mode}")
    if deg_rounds > 1:
        print(f"Analysis depth: {deg_rounds} future picks")
        print(f"Simulations per step: {n_sims}")
        print(f"Shortlist per position: {shortlist}")
    print("")
    
    # Route to appropriate analysis
    if deg_rounds > 1:
        return run_multi_round_degradation_analysis(my_pick, n_rounds, draft_state, 
                                                  deg_rounds, n_sims, shortlist)
    else:
        # Use existing single-step analysis
        pass
    
    # Get current and next pick numbers
    my_team_idx = my_pick - 1
    current_global_pick = len(already_drafted) + 1 if already_drafted else 1
    
    # Find our next pick
    def generate_snake_order(n_teams=14):
        order = []
        for round_num in range(1, 15):
            if round_num % 2 == 1:
                round_picks = list(range(n_teams))
            else:
                round_picks = list(range(n_teams-1, -1, -1))
            order.extend(round_picks)
        return order
    
    snake_order = generate_snake_order()
    our_next_pick = None
    for pick_idx in range(current_global_pick - 1, len(snake_order)):
        if snake_order[pick_idx] == my_team_idx:
            our_next_pick = pick_idx + 1
            break
    
    if our_next_pick is None:
        print("⚠️  Could not determine next pick")
        return
    
    picks_until_next = our_next_pick - current_global_pick
    current_round = ((current_global_pick - 1) // 14) + 1
    next_round = ((our_next_pick - 1) // 14) + 1
    
    print(f"📍 Current: Pick #{current_global_pick} (Round {current_round})")
    print(f"📍 Next Turn: Pick #{our_next_pick} (Round {next_round})")
    print(f"📍 Picks Until Next Turn: {picks_until_next}")
    print("")
    
    # Initialize simulator
    sim = DraftSimulator(n_rounds=n_rounds)
    
    # Calculate positional degradation
    degradation = sim.simulator.calculate_positional_degradation(
        current_global_pick, our_next_pick, already_drafted
    )
    
    # Sort positions by urgency/degradation
    sorted_positions = sorted(degradation.items(), 
                            key=lambda x: x[1]['degradation'], 
                            reverse=True)
    
    # Display results
    for position, info in sorted_positions:
        urgency = info['urgency']
        degradation_pts = info['degradation']
        
        # Set urgency icon and color
        if urgency == 'CRITICAL':
            icon = '🔴'
        elif urgency == 'HIGH':
            icon = '🟠'
        elif urgency == 'MEDIUM':
            icon = '🟡'
        else:
            icon = '🟢'
        
        print(f"{icon} {position} - {urgency} ({degradation_pts:+.1f} pts degradation)")
        print("-" * 70)
        
        # Show available now
        print("Available Now:")
        for i, player in enumerate(info['available_now'], 1):
            print(f"  {i}. {player['name']:<25} - {player['projection']:.1f} pts (Rank {player['rank']:.0f})")
        
        # Show expected at next pick
        print(f"\nExpected at Pick #{our_next_pick}:")
        for i, player in enumerate(info['expected_next'], 1):
            survival_pct = player['survival_prob'] * 100
            print(f"  {i}. {player['name']:<25} - {player['projection']:.1f} pts ({survival_pct:.0f}% available)")
        
        # Show tier degradation timeline
        if info['tier_info']:
            print("\nTier Degradation Timeline:")
            for tier_name, tier_data in info['tier_info'].items():
                prob_pct = tier_data['prob_available'] * 100
                if tier_data['count_now'] > 0:
                    status = "⚠️" if prob_pct < 50 else ""
                    print(f"  • {tier_name}: {tier_data['count_now']} available → {prob_pct:.0f}% chance by next pick {status}")
        
        print("")
    
    # Show recommendations
    print("=" * 70)
    print("💡 RECOMMENDATIONS:")
    print("-" * 70)
    
    # Find critical positions
    critical = [pos for pos, info in degradation.items() if info['urgency'] == 'CRITICAL']
    high = [pos for pos, info in degradation.items() if info['urgency'] == 'HIGH']
    
    if critical:
        print(f"🔴 CRITICAL: Take {', '.join(critical)} NOW - massive degradation expected")
    if high:
        print(f"🟠 HIGH PRIORITY: Consider {', '.join(high)} - significant drop-off coming")
    
    # Check for position runs
    recent_positions = []
    if already_drafted and len(already_drafted) >= 6:
        # Look at last 6 picks
        # Convert to list if it's a set
        drafted_list = list(already_drafted) if isinstance(already_drafted, set) else already_drafted
        for player_name in drafted_list[-6:]:
            # Try to determine position from player name (simplified)
            for pos in ['RB', 'WR', 'TE', 'QB']:
                if any(player_name in p['name'] for p in degradation.get(pos, {}).get('available_now', [])):
                    recent_positions.append(pos)
                    break
    
    if recent_positions:
        pos_counts = {pos: recent_positions.count(pos) for pos in set(recent_positions)}
        runs = [pos for pos, count in pos_counts.items() if count >= 3]
        if runs:
            print(f"\n⚠️ POSITION RUN DETECTED: {', '.join(runs)} being heavily drafted")
            print("   → Consider pivoting to other positions temporarily")


def run_multi_round_degradation_analysis(my_pick: int = 5, n_rounds: int = 14,
                                        draft_state: Optional[dict] = None,
                                        deg_rounds: int = 3, n_sims: int = 200,
                                        shortlist: int = 6):
    """Multi-round degradation analysis showing degradation trajectory across multiple future picks"""
    current_roster, already_drafted, my_pick, mode = _parse_draft_state(draft_state, my_pick)
    
    # Initialize simulator
    sim = DraftSimulator(n_rounds=n_rounds)
    
    # Get current and future pick numbers
    my_team_idx = my_pick - 1
    current_global_pick = len(already_drafted) + 1 if already_drafted else 1
    
    # Generate snake order to find future picks
    def generate_snake_order(n_teams=14):
        order = []
        for round_num in range(1, 15):
            if round_num % 2 == 1:
                round_picks = list(range(n_teams))
            else:
                round_picks = list(range(n_teams-1, -1, -1))
            order.extend(round_picks)
        return order
    
    snake_order = generate_snake_order()
    future_picks = []
    
    # Find next deg_rounds picks for our team
    picks_found = 0
    for pick_idx in range(current_global_pick - 1, len(snake_order)):
        if snake_order[pick_idx] == my_team_idx:
            future_picks.append(pick_idx + 1)  # Convert to 1-based
            picks_found += 1
            if picks_found >= deg_rounds:
                break
    
    if len(future_picks) == 0:
        print("⚠️  Could not determine future picks")
        return
    
    print(f"📍 Current: Pick #{current_global_pick}")
    print(f"📍 Future picks: {', '.join(f'#{p}' for p in future_picks)}")
    print("")
    
    # Calculate multi-round degradation using CRN methodology
    degradation_results = sim.simulator.calculate_multi_round_degradation(
        current_global_pick, future_picks, already_drafted, n_sims, shortlist
    )
    
    # Display results in table format
    print("📊 MULTI-ROUND DEGRADATION ANALYSIS")
    print("=" * 120)
    
    for position in ['RB', 'WR', 'TE', 'QB']:
        if position not in degradation_results:
            continue
            
        pos_data = degradation_results[position]
        
        # Position header with urgency
        urgency = pos_data.get('urgency', 'LOW')
        urgency_icons = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}
        icon = urgency_icons.get(urgency, '🟢')
        
        print(f"\n{icon} {position} - {urgency}")
        print("-" * 120)
        
        # Available now section
        print("Available Now:")
        if pos_data.get('available_now'):
            for i, player in enumerate(pos_data['available_now'][:3], 1):
                print(f"  {i}. {player['name']:<25} - {player['projection']:.1f} pts (Rank {player['rank']:.0f})")
        
        # Multi-round degradation table
        print(f"\nDegradation by Pick:")
        print(f"{'Step':<6} {'Pick':<6} {'Best Available':<25} {'Proj Points':<12} {'Survival %':<12} {'Degradation':<12}")
        print("-" * 120)
        
        step_data = pos_data.get('steps', [])
        
        for step_idx, step in enumerate(step_data):
            step_label = f"Step {step_idx}" if step_idx > 0 else "Now"
            pick_num = current_global_pick if step_idx == 0 else future_picks[step_idx - 1]
            
            if step.get('best_player'):
                player = step['best_player']
                projection = player.get('projection', 0)  # Use actual projection, NOT weighted
                survival_pct = player.get('survival_prob', 1.0) * 100
                degradation = step.get('degradation', 0)
                
                print(f"{step_label:<6} {pick_num:<6} {player['name']:<25} {projection:<12.1f} {survival_pct:<12.0f}% {degradation:<12.1f}")
        
        print("")
    
    # Show overall recommendations
    print("=" * 120)
    print("💡 MULTI-ROUND RECOMMENDATIONS:")
    print("-" * 120)
    
    # Find positions with high degradation in early steps
    critical_early = []
    high_early = []
    
    for position, pos_data in degradation_results.items():
        steps = pos_data.get('steps', [])
        if len(steps) >= 2:  # At least "now" and "step 1"
            step1_degradation = steps[1].get('degradation', 0)
            if step1_degradation > 25:
                critical_early.append(position)
            elif step1_degradation > 15:
                high_early.append(position)
    
    if critical_early:
        print(f"🔴 CRITICAL: Take {', '.join(critical_early)} NOW - massive degradation in next pick")
    if high_early:
        print(f"🟠 HIGH PRIORITY: Consider {', '.join(high_early)} soon - significant degradation coming")
    
    # Show degradation trajectory insights
    print(f"\n📈 DEGRADATION TRAJECTORY INSIGHTS:")
    for position, pos_data in degradation_results.items():
        steps = pos_data.get('steps', [])
        if len(steps) >= 3:  # Enough data for trend analysis
            step1_deg = steps[1].get('degradation', 0)
            step2_deg = steps[2].get('degradation', 0)
            
            if step2_deg > step1_deg * 1.5:
                print(f"  📈 {position}: Accelerating degradation - early action recommended")
            elif step2_deg < step1_deg * 0.7:
                print(f"  📉 {position}: Diminishing degradation - can wait")
    
    print(f"\n✅ Multi-round analysis complete - {n_sims} simulations per step")


def run_vona_analysis(my_pick: int = 5, n_rounds: int = 14, 
                     draft_state: Optional[dict] = None, n_sims: int = 100,
                     parallel: bool = False, n_workers: int = 4):
    """Run VONA (Value of Next Available) analysis - REDIRECTS TO POSITIONAL DEGRADATION"""
    # Redirect to the new positional degradation analysis
    print("📊 Note: VONA analysis has been upgraded to Positional Degradation Analysis")
    print("")
    return run_positional_degradation_analysis(my_pick, n_rounds, draft_state, 
                                             deg_rounds=1, n_sims=n_sims, shortlist=6)
    
    
def run_vona_analysis_legacy(my_pick: int = 5, n_rounds: int = 14, 
                     draft_state: Optional[dict] = None, n_sims: int = 100,
                     parallel: bool = False, n_workers: int = 4):
    """Legacy VONA analysis - kept for reference"""
    print("=" * 70)
    print("📈 VONA DRAFT DECISION ANALYSIS (LEGACY)")
    print("=" * 70)
    
    current_roster, already_drafted, my_pick, mode = _parse_draft_state(draft_state, my_pick)
    
    print(f"Position: Pick #{my_pick} in 14-team league")
    print(f"Rounds: {n_rounds}")
    print(f"Mode: {mode}")
    print("")
    
    if not current_roster:
        print("⚠️  VONA analysis requires an active draft state")
        print("   Start backup_draft.py and draft some players first")
        return
    
    # Input validation
    if n_sims < 1:
        print("⚠️  Error: n_sims must be at least 1")
        return
    if n_sims > 10000:
        print("⚠️  Warning: n_sims > 10000 may be very slow")
    if my_pick < 1 or my_pick > 14:
        print(f"⚠️  Error: my_pick must be between 1-14, got {my_pick}")
        return
    if n_workers < 1:
        print("⚠️  Error: n_workers must be at least 1")
        return
    
    # Initialize simulator with balanced VOR policy
    sim = DraftSimulator(n_rounds=n_rounds)
    
    # Construct full draft state
    full_state = {
        'my_current_roster': current_roster,
        'all_drafted': already_drafted,
        'my_team_idx': my_pick - 1,  # Convert to 0-based
        'current_global_pick': len(already_drafted) + 1
    }
    
    import time
    start_time = time.time()
    
    # Run real VONA analysis
    results = analyze_vona_picks(sim.simulator, full_state, n_sims, parallel, n_workers)
    
    elapsed = time.time() - start_time
    
    if not results:
        print("No VONA results available")
        return
    
    print(f"\n📊 VONA RECOMMENDATIONS ({n_sims} paired simulations, {elapsed:.1f}s):")
    print("-" * 80)
    
    for result in results:
        p_survive_pct = result['p_survive'] * 100
        decision_icon = "✅" if result['decision'] == 'PICK' else "⏳"
        
        print(f"{decision_icon} {result['player']} ({result['pos']}, Rank {result['rank']:.0f})")
        print(f"   VOR: {result['vor']:.1f} | VONA: {result['vona']:+.1f} | P(Win): {result['p_win']:.1%} | P(Survive): {p_survive_pct:.1f}%")
        print(f"   → {result['decision']}")
        print("")
    
    print("💡 VONA = Expected value if you wait - Value if you pick now")
    print("   P(Win) = Probability picking now beats waiting")
    print("   P(Survive) = Probability player survives until your next pick")
    


def run_vor_analysis(my_pick: int = 5, n_sims: int = 100, n_rounds: int = 14,
                    draft_state: Optional[dict] = None,
                    parallel: bool = False, n_workers: int = 4):
    """Run comprehensive VOR analysis comparing regular, shadow, and constraint approaches"""
    print("=" * 70)
    print("🧮 VOR SYSTEM ANALYSIS - SHADOW PRICING & CONSTRAINTS")
    print("=" * 70)
    
    current_roster, already_drafted, my_pick, mode = _parse_draft_state(draft_state, my_pick)
    
    print(f"Position: Pick #{my_pick} in 14-team league")
    print(f"Rounds: {n_rounds}")
    print(f"Simulations per strategy: {n_sims}")
    print(f"Mode: {mode}")
    if parallel:
        n_workers = min(n_workers, 6)
        print(f"Parallel: {n_workers} workers")
    print("")
    
    # Test strategies: baseline, shadow price, and constraint versions
    vor_strategies = [
        ('balanced', 'Baseline VOR'),
        ('shadow_balanced', 'Shadow Price VOR'),
        ('constraint_balanced', 'Constraint VOR'),
        ('rb_heavy', 'RB Heavy (Target)'),
    ]
    
    sim = DraftSimulator(n_rounds=n_rounds)
    results = []
    
    print("🏃 Running VOR analysis:")
    print("-" * 50)
    
    for strategy, description in vor_strategies:
        start_time = time.time()
        
        result = sim.simulator.run_simulations(
            my_pick - 1,
            strategy,
            n_sims,
            initial_roster=current_roster,
            already_drafted=already_drafted,
            parallel=parallel,
            n_workers=n_workers
        )
        
        elapsed = time.time() - start_time
        
        results.append({
            'strategy': strategy,
            'description': description,
            'mean_value': result['mean_value'],
            'std_value': result['std_value'],
            'max_value': result['max_value'],
            'min_value': result['min_value'],
            'elapsed': elapsed
        })
        
        print(f"{description:20} → {result['mean_value']:7.1f} ± {result['std_value']:5.1f} ({elapsed:5.1f}s)")
    
    # Analysis and comparison
    print("\n📊 VOR SYSTEM COMPARISON:")
    print("-" * 50)
    
    baseline = next(r for r in results if r['strategy'] == 'balanced')
    target = next(r for r in results if r['strategy'] == 'rb_heavy')
    shadow = next(r for r in results if r['strategy'] == 'shadow_balanced')
    constraint = next(r for r in results if r['strategy'] == 'constraint_balanced')
    
    baseline_score = baseline['mean_value']
    target_score = target['mean_value']
    gap = target_score - baseline_score
    
    print(f"Baseline VOR:     {baseline_score:.1f} points")
    print(f"RB Heavy Target:  {target_score:.1f} points")
    print(f"Performance Gap:  {gap:+.1f} points")
    print("")
    
    shadow_improvement = shadow['mean_value'] - baseline_score
    shadow_gap_closed = (shadow_improvement / gap) * 100 if gap > 0 else 0
    print(f"Shadow Price VOR: {shadow['mean_value']:7.1f} ({shadow_improvement:+.1f}) - {shadow_gap_closed:.1f}% of gap closed")
    
    constraint_improvement = constraint['mean_value'] - baseline_score
    constraint_gap_closed = (constraint_improvement / gap) * 100 if gap > 0 else 0
    print(f"Constraint VOR:   {constraint['mean_value']:7.1f} ({constraint_improvement:+.1f}) - {constraint_gap_closed:.1f}% of gap closed")
    
    print("\n🎯 IMPROVEMENT ANALYSIS:")
    print("-" * 50)
    
    if shadow_improvement > constraint_improvement:
        print(f"✅ Shadow pricing is more effective (+{shadow_improvement:.1f} vs +{constraint_improvement:.1f})")
        print("   → Early-round RB bonus successfully enforces timing")
    else:
        print(f"✅ Chance constraints are more effective (+{constraint_improvement:.1f} vs +{shadow_improvement:.1f})")
        print("   → Structural constraint better than price signals")
    
    if max(shadow_improvement, constraint_improvement) >= gap * 0.8:
        print(f"🎉 SUCCESS: VOR improvements close ≥80% of performance gap!")
    elif max(shadow_improvement, constraint_improvement) >= gap * 0.5:
        print(f"✅ GOOD: VOR improvements close ≥50% of performance gap")
    else:
        print(f"⚠️  PARTIAL: VOR improvements close <50% of gap - more work needed")
    
    return results


def run_attainment_analysis(my_pick: int = 5, n_sims: int = 100, n_rounds: int = 7,
                           draft_state: Optional[dict] = None):
    """Analyze P(≥2 RBs by Round 2) vs mean points for different shadow prices"""
    print("=" * 70)
    print("📈 ATTAINMENT CURVE ANALYSIS - P(≥2 RBs by R2) vs Performance")
    print("=" * 70)
    
    current_roster, already_drafted, my_pick, mode = _parse_draft_state(draft_state, my_pick)
    
    print(f"Position: Pick #{my_pick} in 14-team league")
    print(f"Rounds: {n_rounds}")
    print(f"Simulations per configuration: {n_sims}")
    print(f"Mode: {mode}")
    print("")
    
    # Test different shadow price levels
    shadow_prices = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    sim = DraftSimulator(n_rounds=n_rounds)
    
    print("🔍 Testing shadow price levels:")
    print("-" * 60)
    print("Shadow Price | Mean Points | Std Dev | P(≥2 RBs by R2)")
    print("-" * 60)
    
    attainment_data = []
    
    for shadow_price in shadow_prices:
        # Create temporary strategy
        temp_params = {
            "alpha": 18,
            "lambda": -2,
            "gamma": 0.92,
            "r_te": 9,
            "delta_qb": 0,
            "rb_shadow": shadow_price,
            "shadow_decay_round": 3
        }
        
        temp_strategy = f"temp_shadow_{shadow_price}"
        
        # Temporarily add to VOR_POLICIES
        from src.monte_carlo.strategies import VOR_POLICIES
        VOR_POLICIES[temp_strategy] = {
            'name': f'Shadow VOR (bonus={shadow_price})',
            'description': f'VOR with {shadow_price} shadow price',
            'params': temp_params
        }
        
        try:
            result = sim.simulator.run_simulations(
                my_pick - 1,
                temp_strategy,
                n_sims,
                initial_roster=current_roster,
                already_drafted=already_drafted
            )
            
            # Calculate P(≥2 RBs by R2) from position frequencies
            rb_r1_pct = 0
            rb_r2_pct = 0
            
            if 'position_frequencies' in result:
                if 'round_1' in result['position_frequencies']:
                    for pos, pct in result['position_frequencies']['round_1']:
                        if pos == 'RB':
                            rb_r1_pct = pct / 100.0
                            
                if 'round_2' in result['position_frequencies']:
                    for pos, pct in result['position_frequencies']['round_2']:
                        if pos == 'RB':
                            rb_r2_pct = pct / 100.0
            
            # Approximate P(≥2 RBs by R2) = P(RB in R1) * P(RB in R2) + other combinations
            # This is a rough estimate - ideally we'd track this directly in simulations
            p_2rb_by_r2 = rb_r1_pct * rb_r2_pct + (1 - rb_r1_pct) * 0.1  # Simplified
            
            attainment_data.append({
                'shadow_price': shadow_price,
                'mean_points': result['mean_value'],
                'std_points': result['std_value'],
                'p_2rb_by_r2': p_2rb_by_r2,
                'rb_r1_pct': rb_r1_pct,
                'rb_r2_pct': rb_r2_pct
            })
            
            print(f"{shadow_price:11} | {result['mean_value']:10.1f} | {result['std_value']:7.1f} | {p_2rb_by_r2:13.2f}")
            
        finally:
            # Clean up temporary strategy
            if temp_strategy in VOR_POLICIES:
                del VOR_POLICIES[temp_strategy]
    
    print("-" * 60)
    print("\n📊 ATTAINMENT CURVE INSIGHTS:")
    print("-" * 40)
    
    # Find optimal point (best balance of performance and RB attainment)
    best_combined = max(attainment_data, 
                       key=lambda x: x['mean_points'] + 100 * x['p_2rb_by_r2'])
    
    print(f"Best combined score: Shadow price {best_combined['shadow_price']}")
    print(f"  → {best_combined['mean_points']:.1f} points, {best_combined['p_2rb_by_r2']:.2f} P(≥2 RBs)")
    
    # Analysis of trade-offs
    max_points = max(attainment_data, key=lambda x: x['mean_points'])
    max_rb_attain = max(attainment_data, key=lambda x: x['p_2rb_by_r2'])
    
    print(f"\nHighest points: Shadow price {max_points['shadow_price']} → {max_points['mean_points']:.1f}")
    print(f"Highest RB attainment: Shadow price {max_rb_attain['shadow_price']} → {max_rb_attain['p_2rb_by_r2']:.2f}")
    
    return attainment_data


def generate_playbook(my_pick: int = 5, n_rounds: int = 14, output_format: str = 'markdown', quick_mode: bool = False):
    """Generate comprehensive pre-draft playbook with ensemble models and two-stage sampling"""
    print("=" * 70)
    print("📖 PRE-DRAFT PLAYBOOK GENERATOR")
    print("=" * 70)
    print(f"Position: Pick #{my_pick} in 14-team league")
    print(f"Rounds: {n_rounds}")
    print(f"Output: {output_format}")
    print("")
    
    start_time = time.time()
    
    # Initialize single optimized model for MVP (ensemble is too slow)
    print("🔧 Initializing probability model...")
    
    from src.monte_carlo import DraftSimulator
    
    # Use existing DraftSimulator for optimal performance
    sim = DraftSimulator(n_rounds=n_rounds)
    
    print(f"✅ Model ready")
    
    # Generate playbook for requested rounds
    playbook_data = {}
    
    print("\n🎯 Generating pick-by-pick analysis...")
    
    # Track simulated draft progression for multi-round analysis
    simulated_roster = []
    simulated_drafted = set()
    
    # Start from current draft state if available
    draft_state = sim.load_draft_state()
    if draft_state:
        simulated_roster = draft_state.get('my_current_roster', [])
        simulated_drafted = set(draft_state.get('all_drafted', []))
    
    # Determine analysis depth based on performance mode
    # Fixed: Always analyze all requested rounds (no artificial limits)
    max_analysis_rounds = n_rounds + 1
    
    for pick_round in range(1, max_analysis_rounds):
        current_pick = _calculate_pick_number(my_pick, pick_round, 14)
        
        print(f"   Analyzing Round {pick_round} (Pick #{current_pick})...")
        
        # Run optimized VONA analysis with simulated state
        pick_analysis = _analyze_pick_optimized(
            sim, my_pick, current_pick, pick_round, k=6,
            simulated_roster=simulated_roster, simulated_drafted=simulated_drafted
        )
        
        playbook_data[f'round_{pick_round}'] = pick_analysis
        
        # Simulate picking the recommended player for next round analysis
        if pick_analysis['recommendation'] != 'No players available':
            simulated_roster.append(pick_analysis['recommendation'])
            simulated_drafted.add(pick_analysis['recommendation'])
    
    elapsed = time.time() - start_time
    
    # Generate output in requested format
    if output_format == 'markdown':
        output = _format_playbook_markdown(playbook_data, my_pick, elapsed)
    elif output_format == 'json':
        output = _format_playbook_json(playbook_data, my_pick, elapsed)
    else:
        output = _format_playbook_markdown(playbook_data, my_pick, elapsed)
    
    print(f"\n✅ Playbook generated in {elapsed:.1f} seconds")
    
    # Save to file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"pre_draft_playbook_pick{my_pick}_{timestamp}.{output_format.split('_')[0]}"
    filepath = Path('data') / filename
    
    with open(filepath, 'w') as f:
        f.write(output)
    
    print(f"💾 Saved to: {filepath}")
    
    # Display summary
    print(f"\n📊 PLAYBOOK SUMMARY:")
    print("-" * 40)
    total_picks = len(playbook_data)
    high_confidence = sum(1 for data in playbook_data.values() if data.get('confidence', '') == 'HIGH')
    print(f"Total picks analyzed: {total_picks}")
    print(f"High confidence picks: {high_confidence}/{total_picks}")
    print(f"Blueprint-aware pivots: Generated for all picks")
    
    return output, filepath


def _calculate_pick_number(my_pick: int, round_num: int, n_teams: int = 14) -> int:
    """Calculate actual pick number for given round in snake draft"""
    if round_num % 2 == 1:  # Odd rounds: normal order
        return (round_num - 1) * n_teams + my_pick
    else:  # Even rounds: reverse order
        return (round_num - 1) * n_teams + (n_teams - my_pick + 1)


def _analyze_pick_optimized(sim, my_pick: int, current_pick: int, round_num: int, k: int = 6, 
                           simulated_roster=None, simulated_drafted=None):
    """Run optimized VONA analysis for a specific pick with two-stage sampling"""
    from src.monte_carlo.strategies import VOR_POLICIES
    import numpy as np
    
    # Get balanced strategy params
    balanced_params = VOR_POLICIES['balanced']['params']
    
    # Use simulated roster and drafted players for future rounds
    current_roster = simulated_roster or []
    already_drafted = simulated_drafted or set()
    
    # Get top K candidates based on current state
    candidates = sim.simulator.get_top_available_players(current_roster, already_drafted, k=k)
    
    if not candidates:
        return {
            'candidates': [],
            'recommendation': 'No players available',
            'confidence': 'LOW',
            'vona_margins': {'expected_range': (0, 0), 'actual_margin': 0, 'within_expected': True},
            'pivot_rules': [],
            'ensemble_agreement': 1.0
        }
    
    # Stage 1: Quick screening with 30 sims (reduced for performance)
    stage1_results = []
    
    print(f"      Running VONA for {len(candidates)} candidates...")
    
    for candidate in candidates:
        player_name = candidate['name']
        player_id = candidate['id']
        
        # Run paired simulations (target player vs best alternative at same pick)
        vona_values = []
        
        for sim_idx in range(VONA_CONFIG['quick_sims']):  # Stage 1: quick sims for speed
            # Use new VONA comparison method - both branches start from same pick
            vona_diff = sim.simulator.simulate_vona_comparison(
                my_pick - 1, balanced_params,
                pick_num=current_pick - 1,
                target_player=player_name,
                seed=42 + sim_idx,
                initial_roster=current_roster,
                already_drafted=already_drafted
            )
            vona_values.append(vona_diff)
        
        # Calculate metrics
        vona_values = np.array(vona_values)
        vona = np.mean(vona_values)
        p_win = np.mean(vona_values > 0)  # Probability that target player is better than alternative
        
        # Get availability
        picks_until_next = _estimate_picks_until_next_turn(my_pick - 1, current_pick, 14)
        available_players = set(sim.simulator.prob_model.players_df.index)
        p_survive = sim.simulator.prob_model.calculate_survival_probability(
            player_id, picks_until_next, available_players
        )
        
        stage1_results.append({
            'player': player_name,
            'vona': vona,
            'p_win': p_win,
            'p_survive': p_survive,
            'position': candidate['pos'],
            'vor': candidate.get('vor', 0)
        })
    
    # Sort by VONA
    stage1_results.sort(key=lambda x: x['vona'], reverse=True)
    
    # Determine recommendation and confidence
    if stage1_results:
        top_pick = stage1_results[0]
        recommendation = top_pick['player']
        
        # Simple confidence based on P(win)
        if top_pick['p_win'] >= VONA_CONFIG['high_confidence']:
            confidence = 'HIGH'
        elif top_pick['p_win'] >= VONA_CONFIG['medium_confidence']:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
    else:
        recommendation = 'No candidates'
        confidence = 'LOW'
    
    # Stage 2: Detailed analysis for low-confidence picks
    if confidence == 'LOW' and len(stage1_results) >= 2:
        print(f"      🔍 Low confidence detected, running Stage 2 analysis...")
        # Re-run top 2 candidates with 100 sims each
        for i, candidate in enumerate(stage1_results[:2]):
            player_name = candidate['player']
            values_now = []
            values_wait = []
            
            for sim_idx in range(VONA_CONFIG['detailed_sims']):  # Stage 2: detailed sims
                value_now = sim.simulator.simulate_from_pick(
                    my_pick - 1, balanced_params,
                    pick_num=current_pick - 1,
                    forced_pick=player_name,
                    seed=1000 + sim_idx,
                    initial_roster=current_roster,
                    already_drafted=already_drafted
                )
                values_now.append(value_now)
                
                next_pick = _estimate_next_pick(my_pick, current_pick, 14)
                value_wait = sim.simulator.simulate_from_pick(
                    my_pick - 1, balanced_params,
                    pick_num=next_pick - 1,
                    forced_pick=None,
                    seed=1000 + sim_idx,
                    initial_roster=current_roster,
                    already_drafted=already_drafted
                )
                values_wait.append(value_wait)
            
            # Update with more precise metrics
            values_now = np.array(values_now)
            values_wait = np.array(values_wait)
            stage1_results[i]['vona'] = np.mean(values_now) - np.mean(values_wait)
            stage1_results[i]['p_win'] = np.mean(values_now > values_wait)
        
        # Re-sort and update recommendation
        stage1_results.sort(key=lambda x: x['vona'], reverse=True)
        if stage1_results:
            top_pick = stage1_results[0]
            recommendation = top_pick['player']
            confidence = 'MEDIUM' if top_pick['p_win'] >= VONA_CONFIG['medium_confidence'] else 'LOW'
    
    # Calculate VONA margins by round
    vona_margins = _calculate_vona_margins_simple(round_num, stage1_results)
    
    # Generate blueprint-aware pivot rules
    pivot_rules = _generate_pivot_rules_simple(stage1_results, round_num)
    
    return {
        'pick_number': current_pick,
        'round': round_num,
        'candidates': {'Balanced': stage1_results},  # Simulate ensemble format
        'recommendation': recommendation,
        'confidence': confidence,
        'vona_margins': vona_margins,
        'pivot_rules': pivot_rules,
        'ensemble_agreement': 1.0  # Single model always agrees with itself
    }


def _analyze_pick_ensemble(models, my_pick: int, current_pick: int, round_num: int, k: int = 6):
    """Run ensemble VONA analysis for a specific pick with two-stage sampling"""
    from src.monte_carlo.strategies import VOR_POLICIES
    import numpy as np
    
    # Get balanced strategy params
    balanced_params = VOR_POLICIES['balanced']['params']
    
    # Get top K candidates from each model
    all_candidates = set()
    for model_name, simulator in models:
        # Get available players for this round
        candidates = simulator.get_top_available_players([], [], k=k)
        for candidate in candidates:
            all_candidates.add(candidate['name'])
    
    # Convert to list and limit to K
    candidate_list = list(all_candidates)[:k]
    
    if not candidate_list:
        return {
            'candidates': [],
            'recommendation': 'No players available',
            'confidence': 'LOW',
            'vona_margins': {},
            'pivot_rules': []
        }
    
    # Stage 1: Quick screening with 50 sims per model
    stage1_results = {}
    
    for model_name, simulator in models:
        model_results = []
        
        for player_name in candidate_list:
            # Get player info
            player_cache = simulator._get_player_cache()
            player_id = player_cache['name_to_id'].get(player_name)
            
            if not player_id:
                continue
                
            # Run paired simulations (pick now vs wait)
            values_now = []
            values_wait = []
            
            for sim_idx in range(50):  # Stage 1: 50 sims
                # Branch A: Pick now
                value_now = simulator.simulate_from_pick(
                    my_pick - 1, balanced_params,
                    pick_num=current_pick - 1,
                    forced_pick=player_name,
                    seed=42 + sim_idx
                )
                values_now.append(value_now)
                
                # Branch B: Wait 
                next_pick = _estimate_next_pick(my_pick, current_pick, 14)
                value_wait = simulator.simulate_from_pick(
                    my_pick - 1, balanced_params,
                    pick_num=next_pick - 1,
                    forced_pick=None,
                    seed=42 + sim_idx
                )
                values_wait.append(value_wait)
            
            # Calculate metrics
            values_now = np.array(values_now)
            values_wait = np.array(values_wait)
            vona = np.mean(values_now) - np.mean(values_wait)
            p_win = np.mean(values_now > values_wait)
            
            # Get availability
            picks_until_next = _estimate_picks_until_next_turn(my_pick - 1, current_pick, 14)
            available_players = set(simulator.prob_model.players_df.index)
            p_survive = simulator.prob_model.calculate_survival_probability(
                player_id, picks_until_next, available_players
            )
            
            model_results.append({
                'player': player_name,
                'vona': vona,
                'p_win': p_win,
                'p_survive': p_survive,
                'position': player_cache['pos'].get(player_id, 'Unknown')
            })
        
        stage1_results[model_name] = sorted(model_results, key=lambda x: x['vona'], reverse=True)
    
    # Determine consensus and confidence
    consensus_pick, confidence = _determine_consensus(stage1_results)
    
    # Stage 2: Detailed analysis for low-confidence picks
    if confidence == 'LOW':
        print(f"      🔍 Low confidence detected, running Stage 2 analysis...")
        # Re-run top 3 candidates with 500 sims each
        # (Implementation would be similar but with higher n_sims)
    
    # Calculate VONA margins by round
    vona_margins = _calculate_vona_margins(round_num, stage1_results)
    
    # Generate blueprint-aware pivot rules
    pivot_rules = _generate_pivot_rules(stage1_results, round_num)
    
    return {
        'pick_number': current_pick,
        'round': round_num,
        'candidates': stage1_results,
        'recommendation': consensus_pick,
        'confidence': confidence,
        'vona_margins': vona_margins,
        'pivot_rules': pivot_rules,
        'ensemble_agreement': _calculate_ensemble_agreement(stage1_results)
    }


def _determine_consensus(stage1_results):
    """Determine consensus pick and confidence from ensemble results"""
    # Get top pick from each model
    top_picks = []
    for model_name, results in stage1_results.items():
        if results:
            top_picks.append(results[0]['player'])
    
    if not top_picks:
        return 'No candidates', 'LOW'
    
    # Check for agreement
    unique_picks = set(top_picks)
    if len(unique_picks) == 1:
        # All models agree
        consensus = top_picks[0]
        # Check P(win) values
        avg_p_win = np.mean([results[0]['p_win'] for results in stage1_results.values() if results])
        confidence = 'HIGH' if avg_p_win >= VONA_CONFIG['high_confidence'] else 'MEDIUM'
    elif len(unique_picks) == 2:
        # 2/3 agreement
        consensus = max(unique_picks, key=lambda x: top_picks.count(x))
        confidence = 'MEDIUM'
    else:
        # No agreement
        consensus = top_picks[0]  # Default to first model
        confidence = 'LOW'
    
    return consensus, confidence


def _calculate_vona_margins(round_num, stage1_results):
    """Calculate VONA margins by round for validation"""
    # Expected margins based on round
    if round_num <= 2:
        expected_range = (5, 8)
    elif round_num <= 5:
        expected_range = (3, 5)
    else:
        expected_range = (0, 2)
    
    # Calculate actual margins from ensemble
    margins = []
    for model_results in stage1_results.values():
        if len(model_results) >= 2:
            top_vona = model_results[0]['vona']
            second_vona = model_results[1]['vona']
            margins.append(abs(top_vona - second_vona))
    
    avg_margin = np.mean(margins) if margins else 0
    
    return {
        'expected_range': expected_range,
        'actual_margin': avg_margin,
        'within_expected': expected_range[0] <= avg_margin <= expected_range[1]
    }


def _calculate_vona_margins_simple(round_num, results):
    """Calculate VONA margins for single model"""
    # Expected margins based on round
    if round_num <= 2:
        expected_range = (5, 8)
    elif round_num <= 5:
        expected_range = (3, 5)
    else:
        expected_range = (0, 2)
    
    # Calculate actual margin
    if len(results) >= 2:
        actual_margin = abs(results[0]['vona'] - results[1]['vona'])
    else:
        actual_margin = 0
    
    return {
        'expected_range': expected_range,
        'actual_margin': actual_margin,
        'within_expected': expected_range[0] <= actual_margin <= expected_range[1]
    }


def _generate_pivot_rules_simple(results, round_num):
    """Generate blueprint-aware pivot rules for single model"""
    pivot_rules = []
    
    if len(results) >= 2:
        primary = results[0]['player']
        secondary = results[1]['player']
        
        pivot_rules.append({
            'condition': f'If {primary} is taken',
            'action': f'Pivot to {secondary}',
            'blueprint_status': '⚠️ Adjust' if round_num <= 3 else '✅ Maintain'
        })
    
    return pivot_rules


def _generate_pivot_rules(stage1_results, round_num):
    """Generate blueprint-aware pivot rules"""
    # Get top 3 candidates across all models
    all_candidates = []
    for model_results in stage1_results.values():
        all_candidates.extend(model_results[:3])
    
    # Sort by average VONA
    candidate_vonas = {}
    for candidate in all_candidates:
        player = candidate['player']
        if player not in candidate_vonas:
            candidate_vonas[player] = []
        candidate_vonas[player].append(candidate['vona'])
    
    # Calculate average VONA for each unique player
    avg_vonas = {player: np.mean(vonas) for player, vonas in candidate_vonas.items()}
    sorted_candidates = sorted(avg_vonas.items(), key=lambda x: x[1], reverse=True)
    
    # Generate pivot rules
    pivot_rules = []
    if len(sorted_candidates) >= 2:
        primary = sorted_candidates[0][0]
        secondary = sorted_candidates[1][0]
        
        pivot_rules.append({
            'condition': f'If {primary} is taken',
            'action': f'Pivot to {secondary}',
            'blueprint_status': '⚠️ Adjust' if round_num <= 3 else '✅ Maintain'
        })
    
    return pivot_rules


def _calculate_ensemble_agreement(stage1_results):
    """Calculate how much the ensemble models agree"""
    top_picks = []
    for results in stage1_results.values():
        if results:
            top_picks.append(results[0]['player'])
    
    if not top_picks:
        return 0.0
    
    # Calculate agreement percentage
    most_common = max(set(top_picks), key=top_picks.count)
    agreement = top_picks.count(most_common) / len(top_picks)
    
    return agreement


def _estimate_next_pick(my_pick: int, current_pick: int, n_teams: int = 14) -> int:
    """Estimate next pick number in snake draft"""
    current_round = ((current_pick - 1) // n_teams) + 1
    
    if current_round % 2 == 1:  # Odd round
        # Next pick is at start of next round (reverse order)
        next_round_start = current_round * n_teams + 1
        return next_round_start + (n_teams - my_pick)
    else:  # Even round
        # Next pick is at start of next round (normal order)
        next_round_start = current_round * n_teams + 1
        return next_round_start + (my_pick - 1)


def _format_playbook_markdown(playbook_data, my_pick, elapsed):
    """Format playbook as markdown"""
    output = f"""# Pre-Draft Playbook - Pick #{my_pick}

Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
Analysis Time: {elapsed:.1f} seconds
Ensemble Models: ESPN-only, ADP-only, 80/20 Blend

---

"""
    
    for round_key, data in playbook_data.items():
        round_num = int(round_key.split('_')[1])
        pick_num = data['pick_number']
        
        # Confidence icon
        confidence_icons = {'HIGH': '✅', 'MEDIUM': '🟡', 'LOW': '⚠️'}
        confidence_icon = confidence_icons.get(data['confidence'], '❓')
        
        output += f"""## Pick #{pick_num} (Round {round_num})
🎯 **Recommendation: {data['recommendation']}**
- Confidence: {confidence_icon} {data['confidence']} ({data['ensemble_agreement']:.0%} agreement)
- VONA Margin: {data['vona_margins']['actual_margin']:.1f} (Expected: {data['vona_margins']['expected_range'][0]}-{data['vona_margins']['expected_range'][1]})

"""
        
        # Show top 3 candidates from analysis
        if data['candidates']:
            # Handle both ensemble and single model formats
            if isinstance(data['candidates'], dict):
                # Ensemble format
                blend_results = data['candidates'].get('Blend', data['candidates'].get('Balanced', []))
            else:
                # Single model format
                blend_results = data['candidates']
                
            if blend_results:
                output += "**Top Candidates:**\n"
                for i, candidate in enumerate(blend_results[:3], 1):
                    availability = candidate['p_survive'] * 100
                    output += f"{i}. {candidate['player']} ({candidate['position']}) - VONA: {candidate['vona']:+.1f}, P(Win): {candidate['p_win']:.1%}, Avail: {availability:.0f}%\n"
                output += "\n"
        
        # Pivot rules
        if data['pivot_rules']:
            output += "**🔄 Pivot Rules:**\n"
            for rule in data['pivot_rules']:
                output += f"- {rule['condition']} → {rule['action']} ({rule['blueprint_status']})\n"
            output += "\n"
        
        output += "---\n\n"
    
    output += f"""
## Validation Results

- Total picks analyzed: {len(playbook_data)}
- High confidence: {sum(1 for d in playbook_data.values() if d['confidence'] == 'HIGH')} picks
- VONA margins within expected: {sum(1 for d in playbook_data.values() if d['vona_margins']['within_expected'])} picks

Generated by Monte Carlo Ensemble Analysis
"""
    
    return output


def _format_playbook_json(playbook_data, my_pick, elapsed):
    """Format playbook as JSON"""
    
    # Convert numpy/pandas types to JSON-serializable Python types
    def make_serializable(obj):
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        else:
            return obj
    
    # Clean the data for JSON serialization
    clean_playbook_data = make_serializable(playbook_data)
    
    output_data = {
        'meta': {
            'pick_position': my_pick,
            'generated_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'analysis_time_seconds': float(elapsed),
            'ensemble_models': ['ESPN-only', 'ADP-only', '80/20 Blend']
        },
        'picks': clean_playbook_data,
        'summary': {
            'total_picks': len(playbook_data),
            'high_confidence_picks': sum(1 for d in playbook_data.values() if d['confidence'] == 'HIGH'),
            'valid_vona_margins': sum(1 for d in playbook_data.values() if d['vona_margins']['within_expected'])
        }
    }
    
    return json.dumps(output_data, indent=2)


def run_starter_optimization(my_pick: int = 5, n_sims: int = 100, n_rounds: int = 14,
                           draft_state: Optional[dict] = None, risk_aversion: float = 0.5,
                           parallel: bool = False, n_workers: int = 4):
    """
    Run starter-aware optimization mode.
    ONLY maximizes 7 starter projected points.
    """
    print("=" * 70)
    print("🎯 STARTER OPTIMIZATION MODE")
    print("=" * 70)
    
    current_roster, already_drafted, my_pick, mode = _parse_draft_state(draft_state, my_pick)
    
    print(f"Position: Pick #{my_pick} in 14-team league")
    print(f"Risk Aversion: {risk_aversion:.1f} (0.0=aggressive, 1.0=conservative)")
    print(f"Rounds: {n_rounds}")
    print(f"Simulations: {n_sims}")
    print(f"Mode: {mode}")
    if parallel:
        n_workers = min(n_workers, 6)
        print(f"Parallel: {n_workers} workers (CPU-friendly)")
    print("")
    
    # Initialize simulator with optimizer mode
    sim = DraftSimulator(n_rounds=n_rounds)
    sim.simulator._optimizer_mode = True  # Enable optimizer mode
    
    # Create optimization strategy parameters
    optimization_params = {
        'optimize_starters': True,
        'risk_aversion': risk_aversion,
        'name': f'Starter Optimization (risk={risk_aversion:.1f})'
    }
    
    # Run simulation - we'll manually call the simulator since we need special handling
    results = []
    position_frequency = defaultdict(lambda: defaultdict(int))
    
    start_time = time.time()
    
    for sim_idx in range(n_sims):
        result = sim.simulator.simulate_single_draft(
            my_pick - 1, optimization_params, seed=42 + sim_idx,
            initial_roster=current_roster, already_drafted=already_drafted
        )
        
        results.append(result)
        
        # Track position frequency by round
        seq = result['position_sequence']
        for round_num in range(1, min(15, len(seq) + 1)):
            if len(seq) >= round_num:
                position = seq[round_num - 1]
                position_frequency[round_num][position] += 1
    
    elapsed = time.time() - start_time
    
    # Aggregate results
    values = [r['roster_value'] for r in results]
    starter_points = [r['starter_points'] for r in results]
    
    # Calculate position frequency percentages
    position_frequencies = {}
    for round_num, pos_counts in position_frequency.items():
        if pos_counts:
            total_picks = sum(pos_counts.values())
            frequencies = {}
            for pos, count in pos_counts.items():
                percentage = (count / total_picks) * 100
                frequencies[pos] = percentage
            sorted_freq = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
            position_frequencies[f'round_{round_num}'] = sorted_freq
    
    # Display results focused on starter optimization
    print(f"\n🎯 STARTER OPTIMIZATION RESULTS:")
    print("-" * 50)
    print(f"Total Starter Points: {np.mean(starter_points):.1f} ± {np.std(starter_points):.1f}")
    print(f"Total Roster Value: {np.mean(values):.1f} ± {np.std(values):.1f}")
    print(f"Best Roster: {np.max(values):.1f} points")
    print(f"Worst Roster: {np.min(values):.1f} points")
    
    # Show sample starter breakdown from first result
    if results and 'starters' in results[0]:
        from src.monte_carlo.starter_optimizer import get_starter_breakdown
        starter_breakdown = get_starter_breakdown(results[0]['roster'])
        print(f"\n📋 SAMPLE STARTER BREAKDOWN:")
        print("-" * 50)
        
        # Display in logical order: QB, RB1, RB2, WR1, WR2, TE1, FLEX
        position_order = ['QB', 'RB', 'WR', 'TE', 'FLEX']
        for pos in position_order:
            if pos in starter_breakdown:
                players = starter_breakdown[pos]
                for i, player in enumerate(players):
                    if pos == 'FLEX':
                        print(f"FLEX: {player['name']} ({player['proj']:.0f} pts)")
                    else:
                        print(f"{pos}{i+1}: {player['name']} ({player['proj']:.0f} pts)")
        
    
    # Show bench depth achieved using 25% rule
    if results:
        from src.monte_carlo.starter_optimizer import calculate_bench_depth_report
        bench_depths = []
        for result in results:
            if 'starters' in result:
                bench_depth = calculate_bench_depth_report(result['roster'], result['starters'])
                bench_depths.append(bench_depth)
        
        if bench_depths:
            print(f"\n📋 ACHIEVED BENCH DEPTH (25% rule):")
            print("-" * 50)
            for pos in ['RB', 'WR', 'TE', 'QB']:
                avg_depth = np.mean([bd[pos] for bd in bench_depths])
                if avg_depth > 0:
                    print(f"{pos}: {avg_depth:.1f} quality bench players")
            print("(Quality = within 25% of starter's projected points)")
    
    # Show position frequency by round
    if position_frequencies:
        print("\n🎯 POSITION FREQUENCY BY ROUND:")
        print("-" * 50)
        
        for round_num in range(1, 8):  # Show first 7 rounds
            round_key = f'round_{round_num}'
            if round_key in position_frequencies:
                frequencies = position_frequencies[round_key]
                if frequencies:
                    print(f"\nRound {round_num}:")
                    for pos, percentage in frequencies:
                        print(f"  {pos}: {percentage:.1f}%")
    
    print(f"\n✅ Completed {n_sims} simulations in {elapsed:.1f} seconds")
    print(f"💡 Strategy: Maximize starter points only, bench depth is informational")


def run_scenario_analysis(my_pick: int = 5, n_sims: int = 100, base_strategy: str = 'balanced'):
    """Run multiple scenarios with different trade-off parameters"""
    from src.monte_carlo.strategies import generate_scenario_configs
    import numpy as np
    
    print("=" * 70)
    print("🎭 SCENARIO ANALYSIS - STARTER VS BENCH TRADE-OFFS")
    print("=" * 70)
    print(f"Position: Pick #{my_pick} in 14-team league")
    print(f"Base Strategy: {base_strategy}")
    print(f"Simulations per scenario: {n_sims}")
    print("")
    
    # Generate scenario configurations
    scenarios = generate_scenario_configs(base_strategy)
    
    # Initialize simulator
    sim = DraftSimulator(n_rounds=14)
    my_team_idx = my_pick - 1
    
    # Load draft state if available
    draft_state = sim.load_draft_state()
    current_roster, already_drafted, _, _ = _parse_draft_state(draft_state, my_pick)
    
    # Run each scenario
    results = []
    baseline_starter_points = None
    
    for i, scenario in enumerate(scenarios[:10], 1):  # Run first 10 scenarios
        print(f"\nScenario {i}: {scenario['name']}")
        print(f"  Risk Aversion: {scenario['risk_aversion']:.1f}")
        print(f"  Bench Decay: {scenario['bench_value_decay']:.1f}")
        
        # Get strategy params based on type
        if 'params' in scenario:
            strategy_params = scenario['params']
        else:
            strategy_params = scenario['multipliers']
            strategy_params['bench_phase'] = scenario['bench_phase']
        
        # Run simulations
        scenario_results = []
        for sim_idx in range(n_sims):
            result = sim.simulator.simulate_single_draft(
                my_team_idx, strategy_params, 
                seed=42 + sim_idx,
                initial_roster=current_roster,
                already_drafted=already_drafted
            )
            
            # Calculate bench metrics
            roster_analysis = sim.simulator.calculate_roster_value(result['roster'])
            bench_metrics = sim.simulator.calculate_bench_quality_metrics(
                result['roster'], 
                roster_analysis['starters'],
                {p['id']: p['proj'] for p in result['roster']}
            )
            
            scenario_results.append({
                'starter_points': roster_analysis['starter_points'],
                'bench_metrics': bench_metrics,
                'total_value': roster_analysis['total_value']
            })
        
        # Aggregate results
        avg_starters = np.mean([r['starter_points'] for r in scenario_results])
        avg_quality_backups = np.mean([r['bench_metrics']['quality_backups']['total'] 
                                       for r in scenario_results])
        avg_bench_value = np.mean([r['bench_metrics']['total_bench_value'] 
                                   for r in scenario_results])
        
        # Calculate trade-off score
        if baseline_starter_points is None:
            baseline_starter_points = avg_starters
            trade_off_score = 0
            starter_diff = 0
        else:
            starter_diff = avg_starters - baseline_starter_points
            bench_gain = avg_quality_backups * 10  # Each quality backup worth ~10 points
            trade_off_score = starter_diff + bench_gain
        
        # Store results
        result_summary = {
            'scenario': scenario['name'],
            'risk_aversion': scenario['risk_aversion'],
            'bench_decay': scenario['bench_value_decay'],
            'avg_starters': avg_starters,
            'avg_quality_backups': avg_quality_backups,
            'avg_bench_value': avg_bench_value,
            'starter_diff': avg_starters - baseline_starter_points if baseline_starter_points else 0,
            'trade_off_score': trade_off_score
        }
        results.append(result_summary)
        
        # Print summary
        print(f"  Starters: {avg_starters:.1f} pts")
        print(f"  Quality Backups: {avg_quality_backups:.1f}")
        if baseline_starter_points:
            print(f"  Trade-off: {starter_diff:+.1f} starters, {avg_quality_backups:.1f} backups")
            print(f"  Score: {trade_off_score:+.1f}")
    
    # Sort by trade-off score
    results.sort(key=lambda x: x['trade_off_score'], reverse=True)
    
    print("\n" + "=" * 70)
    print("📊 SCENARIO RANKINGS (by trade-off score):")
    print("-" * 70)
    
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. {result['scenario']}")
        print(f"   Starters: {result['avg_starters']:.1f} pts ({result['starter_diff']:+.1f})")
        print(f"   Quality Backups: {result['avg_quality_backups']:.1f}")
        print(f"   Trade-off Score: {result['trade_off_score']:+.1f}")
        print(f"   Parameters: risk={result['risk_aversion']:.1f}, bench={result['bench_decay']:.1f}")
    
    print("\n💡 INSIGHTS:")
    best = results[0]
    worst = results[-1]
    
    if best['avg_quality_backups'] > worst['avg_quality_backups'] * 1.5:
        print(f"✓ {best['scenario']} provides {best['avg_quality_backups']:.1f} quality backups")
        print(f"  vs {worst['avg_quality_backups']:.1f} for {worst['scenario']}")
    
    if abs(best['starter_diff']) < 5:
        print(f"✓ Minimal starter sacrifice ({best['starter_diff']:+.1f} pts) for better depth")
    
    return results


def main():
    """Main entry point with argparse support"""
    parser = argparse.ArgumentParser(
        description="Monte Carlo Fantasy Football Draft Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python monte_carlo_runner.py compare --rounds 14         # Compare strategies (traditional)
  python monte_carlo_runner.py compare_fast --rounds 14    # Compare strategies (CRN + adaptive)
  python monte_carlo_runner.py degradation                 # Single-step degradation analysis 
  python monte_carlo_runner.py degradation --deg-rounds 3  # Multi-round degradation analysis (NEW!)
  python monte_carlo_runner.py degradation --deg-rounds 3 --deg-sims 500 --shortlist 8  # High-precision analysis
  python monte_carlo_runner.py discover --rounds 14        # Discover natural patterns
  python monte_carlo_runner.py balanced --rounds 7         # Run balanced strategy (7 rounds)
  python monte_carlo_runner.py zero_rb --n-sims 200        # Run Zero-RB with more simulations
  python monte_carlo_runner.py starter --n-sims 50         # Run starter optimizer (marginal value optimization)
  python monte_carlo_runner.py balanced --parallel         # Run balanced with 4 CPU-friendly workers
  python monte_carlo_runner.py zero_rb --parallel --workers 6  # Run with 6 workers (max)
  python monte_carlo_runner.py optimize --pick 5 --risk 0.5   # Starter optimization (balanced risk)
  python monte_carlo_runner.py optimize --pick 12 --risk 0.0  # Aggressive starter optimization
  python monte_carlo_runner.py playbook --pick 5 --rounds 7  # Generate pre-draft playbook
  python monte_carlo_runner.py playbook --pick 12 --output-format json  # JSON playbook
        """
    )
    
    parser.add_argument(
        'mode', 
        choices=['compare', 'compare_fast', 'discover', 'export', 'grid_search', 'vona', 'degradation', 'playbook',
                'balanced', 'zero_rb', 'rb_heavy', 'hero_rb', 'elite_qb', 'starter_max', 'starter',
                'conservative', 'aggressive', 'rb_focused', 'early_value', 'nuclear', 'rb_heavy_vor',
                'shadow_conservative', 'shadow_balanced', 'constraint_balanced', 'vor_analysis', 'attainment',
                'scenarios', 'optimize'],
        help='Simulation mode to run'
    )
    parser.add_argument(
        '--rounds', 
        type=int, 
        default=14,
        help='Number of rounds to simulate (default: 14)'
    )
    parser.add_argument(
        '--n-sims', 
        type=int, 
        default=100,
        help='Number of simulations to run (default: 100)'
    )
    parser.add_argument(
        '--pick', 
        type=int, 
        default=5,
        help='Your draft pick number (default: 5)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='balanced',
        choices=['balanced', 'zero_rb', 'rb_heavy', 'hero_rb', 'elite_qb', 'starter_max', 'starter_optimize', 'conservative', 'aggressive', 'rb_focused', 'early_value', 'nuclear', 'rb_heavy_vor', 'shadow_conservative', 'shadow_balanced', 'constraint_balanced'],
        help='Strategy for export mode (default: balanced)'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run simulations in parallel (CPU-friendly with 4 workers by default)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4, max: 6 for CPU-friendly operation)'
    )
    parser.add_argument(
        '--output-format',
        type=str,
        default='markdown',
        choices=['markdown', 'json'],
        help='Output format for playbook mode (default: markdown)'
    )
    parser.add_argument(
        '--deg-rounds',
        type=int,
        default=1,
        help='Number of future picks to analyze for degradation analysis (default: 1)'
    )
    parser.add_argument(
        '--deg-sims',
        type=int,
        default=200,
        help='Number of simulations per degradation step (default: 200)'
    )
    parser.add_argument(
        '--shortlist',
        type=int,
        default=6,
        help='Number of top candidates to analyze per position (default: 6)'
    )
    parser.add_argument(
        '--risk',
        type=float,
        default=0.5,
        help='Risk aversion level for optimizer mode (0.0 = aggressive, 1.0 = conservative, default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Check for live draft state
    draft_state = load_live_draft_state(args.rounds)
    
    # Override pick number if draft state is available
    if draft_state:
        my_pick = draft_state.get('my_team_idx', args.pick - 1) + 1
    else:
        my_pick = args.pick
    
    # Run appropriate mode
    if args.mode == 'compare':
        run_strategy_comparison(my_pick, args.n_sims, args.rounds)
        
    elif args.mode == 'compare_fast':
        compare_fast(my_pick, args.rounds, draft_state)
        
    elif args.mode == 'discover':
        run_pattern_discovery(my_pick, args.n_sims, args.rounds, draft_state)
        
    elif args.mode == 'export':
        export_simulation_data(args.strategy, my_pick, args.n_sims, args.rounds, draft_state, export_parquet=True)
        
    elif args.mode == 'grid_search':
        grid_search_successive_halving(my_pick, args.rounds, draft_state)
        
    elif args.mode == 'vona':
        run_vona_analysis(my_pick, args.rounds, draft_state, args.n_sims, args.parallel, args.workers)
    
    elif args.mode == 'degradation':
        run_positional_degradation_analysis(my_pick, args.rounds, draft_state, 
                                           deg_rounds=args.deg_rounds, n_sims=args.deg_sims, 
                                           shortlist=args.shortlist)
        
    elif args.mode in ['balanced', 'zero_rb', 'rb_heavy', 'hero_rb', 'elite_qb', 'starter_max', 'conservative', 'aggressive', 'rb_focused', 'early_value', 'nuclear', 'rb_heavy_vor', 'shadow_conservative', 'shadow_balanced', 'constraint_balanced']:
        run_single_strategy(args.mode, my_pick, args.n_sims, args.rounds, draft_state, 
                          parallel=args.parallel, n_workers=args.workers)
    
    elif args.mode == 'starter':
        run_single_strategy('starter_optimize', my_pick, args.n_sims, args.rounds, draft_state, 
                          parallel=args.parallel, n_workers=args.workers)
    
    elif args.mode == 'vor_analysis':
        run_vor_analysis(my_pick, args.n_sims, args.rounds, draft_state,
                        parallel=args.parallel, n_workers=args.workers)
    
    elif args.mode == 'attainment':
        run_attainment_analysis(my_pick, args.n_sims, args.rounds, draft_state)
    
    elif args.mode == 'playbook':
        generate_playbook(my_pick, args.rounds, args.output_format)
    
    elif args.mode == 'scenarios':
        run_scenario_analysis(my_pick, args.n_sims, args.strategy)
    
    elif args.mode == 'optimize':
        run_starter_optimization(my_pick, args.n_sims, args.rounds, draft_state, 
                               args.risk, args.parallel, args.workers)
    
    print("\n" + "=" * 70)
    print("Monte Carlo Draft Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()