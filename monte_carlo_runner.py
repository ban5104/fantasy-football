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
from pathlib import Path
from typing import Optional
from src.monte_carlo import (DraftSimulator, compare_all_strategies, 
                           quick_simulation, discover_patterns, PatternDetector)


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
        
        # Show top pattern if available
        if '2_round' in stats['patterns'] and stats['patterns']['2_round']:
            top_pattern = stats['patterns']['2_round'][0]
            print(f"   Most common start: {top_pattern[0]} ({top_pattern[1]} times)")
    
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
    
    # Show common patterns
    if 'patterns' in result:
        print("\n🎯 COMMON DRAFT PATTERNS:")
        print("-" * 40)
        
        for pattern_type, patterns in result['patterns'].items():
            if patterns:
                print(f"\n{pattern_type.replace('_', ' ').title()}:")
                for pattern, count in patterns[:3]:
                    freq = count / n_sims * 100
                    print(f"  {pattern}: {freq:.1f}% frequency")
    
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
    


def main():
    """Main entry point with argparse support"""
    parser = argparse.ArgumentParser(
        description="Monte Carlo Fantasy Football Draft Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python monte_carlo_runner.py compare --rounds 14         # Compare strategies (traditional)
  python monte_carlo_runner.py compare_fast --rounds 14    # Compare strategies (CRN + adaptive)
  python monte_carlo_runner.py discover --rounds 14        # Discover natural patterns
  python monte_carlo_runner.py balanced --rounds 7         # Run balanced strategy (7 rounds)
  python monte_carlo_runner.py zero_rb --n-sims 200        # Run Zero-RB with more simulations
  python monte_carlo_runner.py balanced --parallel         # Run balanced with 4 CPU-friendly workers
  python monte_carlo_runner.py zero_rb --parallel --workers 6  # Run with 6 workers (max)
        """
    )
    
    parser.add_argument(
        'mode', 
        choices=['compare', 'compare_fast', 'discover', 'export', 'balanced', 'zero_rb', 'rb_heavy', 'hero_rb', 'elite_qb'],
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
        choices=['balanced', 'zero_rb', 'rb_heavy', 'hero_rb', 'elite_qb'],
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
        
    elif args.mode in ['balanced', 'zero_rb', 'rb_heavy', 'hero_rb', 'elite_qb']:
        run_single_strategy(args.mode, my_pick, args.n_sims, args.rounds, draft_state, 
                          parallel=args.parallel, n_workers=args.workers)
    
    print("\n" + "=" * 70)
    print("Monte Carlo Draft Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()