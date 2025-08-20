#!/usr/bin/env python3
"""
Extract Championship DNA from your Monte Carlo simulations
This is what you actually want - not strategy labels, but winning patterns!
"""

import sys
import os
sys.path.append('..')

from src.monte_carlo import DraftSimulator
from championship_dna_analyzer import ChampionshipDNAAnalyzer
import numpy as np
import pandas as pd
import json
from collections import defaultdict

def collect_detailed_rosters(my_pick=5, n_sims=2000):
    """
    Run simulations and collect DETAILED roster data
    We need the actual rosters, not just the final scores
    """
    print(f"🎲 Collecting {n_sims} detailed roster simulations...")
    print(f"   Draft position: #{my_pick}")
    print("-"*50)
    
    sim = DraftSimulator()
    all_rosters = []
    
    # Run balanced strategy with many simulations
    # (or run multiple strategies if you want variety)
    strategies_to_test = ['balanced']  # Keep it simple
    sims_per_strategy = n_sims // len(strategies_to_test)
    
    for strategy in strategies_to_test:
        print(f"\n📊 Running {strategy} strategy ({sims_per_strategy} sims)...")
        
        # We need to modify the simulator to return full roster details
        # For now, let's run simulations and extract what we can
        for sim_idx in range(sims_per_strategy):
            if sim_idx % 100 == 0:
                print(f"   Progress: {sim_idx}/{sims_per_strategy}")
            
            # Run single simulation with detailed tracking
            result = sim.simulator.simulate_single_draft(
                my_team_idx=my_pick - 1,
                strategy_multipliers=sim.simulator.prob_model.get_strategy(strategy)['multipliers'],
                seed=42 + sim_idx
            )
            
            # Extract roster composition data
            roster_data = {
                'roster_value': result['roster_value'],
                'players': [],
                'position_counts': defaultdict(int),
                'strategy': strategy,
                'draft_position': my_pick,
                'sim_id': sim_idx
            }
            
            # Get player details
            for player in result['roster']:
                player_info = {
                    'name': player.get('player_name', 'Unknown'),
                    'pos': player['pos'],
                    'proj': player.get('proj', 0),
                    'rank': player.get('espn_rank', 999),
                    'position_rank': 0  # We'll calculate this
                }
                roster_data['players'].append(player_info)
                roster_data['position_counts'][player['pos']] += 1
            
            # Store first 5 round positions as sequence
            if 'position_sequence' in result:
                roster_data['draft_sequence'] = [
                    {'pos': pos} for pos in result['position_sequence'][:5]
                ]
            
            all_rosters.append(roster_data)
    
    print(f"\n✅ Collected {len(all_rosters)} roster simulations")
    
    # Calculate position ranks within collected data
    all_players = []
    for roster in all_rosters:
        for player in roster['players']:
            all_players.append(player)
    
    if all_players:
        df = pd.DataFrame(all_players)
        df['position_rank'] = df.groupby('pos')['proj'].rank(method='min', ascending=False)
        
        # Update rosters with position ranks
        pos_rank_map = {}
        for _, row in df.iterrows():
            key = f"{row['name']}_{row['pos']}"
            if key not in pos_rank_map:
                pos_rank_map[key] = row['position_rank']
        
        for roster in all_rosters:
            for player in roster['players']:
                key = f"{player['name']}_{player['pos']}"
                if key in pos_rank_map:
                    player['position_rank'] = pos_rank_map[key]
    
    return all_rosters

def find_your_championship_dna(my_pick=5, n_sims=1000):
    """
    The main function - find what winning teams look like!
    """
    print("\n" + "="*60)
    print("🧬 DISCOVERING YOUR CHAMPIONSHIP DNA")
    print("="*60)
    
    # Step 1: Collect roster data
    rosters = collect_detailed_rosters(my_pick, n_sims)
    
    # Step 2: Analyze for patterns
    analyzer = ChampionshipDNAAnalyzer(percentile_cutoff=90)  # Top 10%
    dna = analyzer.analyze_simulation_results(rosters)
    
    # Step 3: Generate report
    analyzer.print_championship_report(dna)
    
    # Step 4: Create actionable draft sheet
    print("\n" + "="*60)
    print("📝 YOUR PERSONAL DRAFT CHEAT SHEET")
    print("="*60)
    
    # Position targets based on DNA
    pos_comp = dna['position_composition']
    tier_dist = dna['tier_distribution']
    
    print("\n🎯 MUST-HAVE TARGETS:")
    print("-"*40)
    
    # Find critical positions (where elite matters)
    critical_positions = []
    for pos in ['RB', 'WR', 'TE', 'QB']:
        if pos in tier_dist:
            elite_freq = tier_dist[pos].get('elite', {}).get('frequency', 0)
            tier1_freq = tier_dist[pos].get('tier1', {}).get('frequency', 0)
            combined = elite_freq + tier1_freq
            
            if combined >= 0.5:  # 50%+ of winners have top-tier
                critical_positions.append((pos, combined))
    
    critical_positions.sort(key=lambda x: x[1], reverse=True)
    
    for i, (pos, freq) in enumerate(critical_positions[:3], 1):
        print(f"{i}. {pos}: Get top-8 player ({freq*100:.0f}% of winners have one)")
    
    print("\n📊 ROSTER CONSTRUCTION TARGETS:")
    print("-"*40)
    for pos in ['QB', 'RB', 'WR', 'TE']:
        if pos in pos_comp:
            avg = pos_comp[pos]['avg']
            mode = pos_comp[pos]['mode']
            print(f"{pos}: Draft {mode} total (winners avg {avg:.1f})")
    
    print("\n🔥 WINNING DRAFT PATTERNS:")
    print("-"*40)
    if dna['common_sequences']:
        print("Most successful first 5 rounds:")
        for i, seq in enumerate(dna['common_sequences'][:3], 1):
            positions = seq['pattern'].split('-')
            print(f"{i}. Rounds 1-5: {' → '.join(positions)} ({seq['frequency']*100:.1f}% success rate)")
    
    print("\n💡 KEY INSIGHTS:")
    print("-"*40)
    
    # Generate specific insights
    insights = []
    
    # RB insights
    if 'RB' in tier_dist:
        rb_elite = tier_dist['RB'].get('elite', {}).get('frequency', 0)
        if rb_elite >= 0.5:
            insights.append(f"• {rb_elite*100:.0f}% of winners have an ELITE RB - prioritize this!")
        
        rb_count = pos_comp.get('RB', {}).get('avg', 0)
        if rb_count >= 3:
            insights.append(f"• Winners draft {rb_count:.1f} RBs on average - don't neglect depth")
    
    # WR insights  
    if 'WR' in tier_dist:
        wr_tier2 = tier_dist['WR'].get('tier2', {}).get('frequency', 0)
        if wr_tier2 >= 0.6:
            insights.append(f"• WR depth matters - {wr_tier2*100:.0f}% have tier-2 WRs")
    
    # TE insights
    if 'TE' in tier_dist:
        te_elite = tier_dist['TE'].get('elite', {}).get('frequency', 0)
        if te_elite >= 0.4:
            insights.append(f"• Elite TE strategy viable - {te_elite*100:.0f}% of winners have one")
    
    for insight in insights[:5]:
        print(insight)
    
    # Save to file for reference during draft
    output = {
        'draft_position': my_pick,
        'simulations_run': n_sims,
        'championship_dna': dna,
        'critical_positions': [(p, float(f)) for p, f in critical_positions],
        'insights': insights
    }
    
    with open('my_championship_dna.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print("\n✅ Saved analysis to 'my_championship_dna.json'")
    print("\n🏆 Now go draft like a champion! Focus on the DNA, not the strategy names.")
    
    return dna

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Find your Championship DNA')
    parser.add_argument('--pick', type=int, default=5, help='Your draft position (1-14)')
    parser.add_argument('--sims', type=int, default=1000, help='Number of simulations')
    
    args = parser.parse_args()
    
    dna = find_your_championship_dna(args.pick, args.sims)