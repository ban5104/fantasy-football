#!/usr/bin/env python3
"""
Show actual successful roster examples from Championship DNA analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_simulation_results(strategy='balanced', pick=5):
    """Load simulation results for a specific strategy and pick"""
    data_dir = Path('data/cache')
    
    # Find most recent simulation file
    pattern = f"{strategy}_pick{pick}_n*_r14.parquet"
    files = list(data_dir.glob(pattern))
    
    if not files:
        print(f"No simulation files found for {strategy} at pick {pick}")
        return None
    
    # Use file with most simulations
    files_with_sims = []
    for f in files:
        try:
            parts = f.stem.split('_')
            for part in parts:
                if part.startswith('n'):
                    n_sims = int(part[1:])
                    files_with_sims.append((f, n_sims))
                    break
        except:
            continue
    
    if files_with_sims:
        file_to_use = max(files_with_sims, key=lambda x: x[1])[0]
        print(f"Loading: {file_to_use.name}")
        return pd.read_parquet(file_to_use)
    return None

def calculate_optimal_lineup(roster_df):
    """Calculate optimal starting lineup points"""
    # League settings: QB, 2 RB, 3 WR, TE, FLEX (RB/WR/TE)
    lineup = {
        'QB': [],
        'RB': [],
        'WR': [],
        'TE': [],
        'FLEX': []
    }
    
    # Sort by sampled points descending
    roster_df = roster_df.sort_values('sampled_points', ascending=False)
    
    # Fill starting positions
    for pos in ['QB', 'RB', 'WR', 'TE']:
        pos_players = roster_df[roster_df['pos'] == pos]
        
        if pos == 'QB':
            lineup['QB'] = pos_players.head(1)
        elif pos == 'RB':
            lineup['RB'] = pos_players.head(2)
        elif pos == 'WR':
            lineup['WR'] = pos_players.head(2)  # Only 2 WR spots, not 3
        elif pos == 'TE':
            lineup['TE'] = pos_players.head(1)
    
    # Determine FLEX (best remaining RB/WR/TE)
    used_players = pd.concat([
        lineup['QB'], lineup['RB'], lineup['WR'], lineup['TE']
    ])
    
    remaining = roster_df[~roster_df.index.isin(used_players.index)]
    flex_eligible = remaining[remaining['pos'].isin(['RB', 'WR', 'TE'])]
    
    if not flex_eligible.empty:
        lineup['FLEX'] = flex_eligible.head(1)
    
    # Calculate total starter points
    starters = pd.concat([v for v in lineup.values() if not v.empty])
    starter_points = starters['sampled_points'].sum()
    
    # Identify bench
    bench = roster_df[~roster_df.index.isin(starters.index)]
    
    return lineup, bench, starter_points

def show_successful_rosters(strategy='balanced', pick=5, n_examples=5):
    """Show detailed examples of successful rosters"""
    
    df = load_simulation_results(strategy, pick)
    if df is None:
        return
    
    print("\n" + "="*80)
    print("HOW SUCCESS IS DETERMINED:")
    print("="*80)
    print("• Success = Top 10% of rosters by STARTER POINTS only")
    print("• Starters = Optimal weekly lineup (1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX)")
    print("• Bench shown separately for your subjective evaluation")
    print("• Each simulation varies player projections ±20% (Beta-PERT)")
    
    # Get top 10% by starter points (not total roster value)
    starter_values = df.groupby('sim')['starter_points'].first()
    threshold = starter_values.quantile(0.9)
    winning_sims = starter_values[starter_values >= threshold].index
    
    print(f"\n📊 Found {len(winning_sims)} successful rosters out of {df['sim'].nunique()} simulations")
    print(f"   Success rate: {len(winning_sims)/df['sim'].nunique():.1%}")
    print(f"   Starter points threshold: {threshold:.0f} points")
    
    # Show examples
    print("\n" + "="*80)
    print(f"TOP {n_examples} SUCCESSFUL ROSTER EXAMPLES:")
    print("="*80)
    
    # Get top rosters sorted by starter points
    top_rosters = starter_values[winning_sims].sort_values(ascending=False)
    top_sims = top_rosters.head(n_examples).index
    
    for i, sim_id in enumerate(top_sims, 1):
        roster = df[df['sim'] == sim_id].sort_values('draft_pick')
        starter_pts = roster['starter_points'].iloc[0]
        
        print(f"\n{'='*40}")
        print(f"EXAMPLE #{i} - Starter Points: {starter_pts:.0f}")
        print(f"{'='*40}")
        
        # Calculate optimal lineup
        lineup, bench, starter_points = calculate_optimal_lineup(roster)
        
        # Show draft order
        print("\n📝 DRAFT ORDER:")
        for _, player in roster.iterrows():
            round_num = player['draft_round']
            pick_num = player['draft_pick']
            print(f"  R{round_num:2d}.{pick_num:3d}: {player['player_name']:20s} {player['pos']:3s} - {player['sampled_points']:.0f} pts")
        
        # Show starting lineup
        print("\n⭐ OPTIMAL STARTING LINEUP (7 total):")
        total_starter_pts = 0
        for pos_name, players in lineup.items():
            if not players.empty:
                for _, p in players.iterrows():
                    print(f"  {pos_name:4s}: {p['player_name']:20s} - {p['sampled_points']:.0f} pts")
                    total_starter_pts += p['sampled_points']
        
        print(f"  {'─'*35}")
        print(f"  STARTER TOTAL: {total_starter_pts:.0f} pts")
        
        # Show bench
        print("\n🪑 BENCH:")
        bench_total = 0
        if not bench.empty:
            for _, p in bench.iterrows():
                print(f"  {p['pos']:3s}: {p['player_name']:20s} - {p['sampled_points']:.0f} pts")
                bench_total += p['sampled_points']
        else:
            print("  (No bench players)")
        
        print(f"  {'─'*35}")
        print(f"  BENCH TOTAL: {bench_total:.0f} pts")
        
        # Show roster composition
        pos_counts = roster.groupby('pos').size()
        print("\n📊 ROSTER COMPOSITION:")
        print(f"  RB: {pos_counts.get('RB', 0)}  WR: {pos_counts.get('WR', 0)}  " +
              f"QB: {pos_counts.get('QB', 0)}  TE: {pos_counts.get('TE', 0)}")
        
        # Analyze strategy type
        rb_early = len(roster[(roster['pos'] == 'RB') & (roster['draft_round'] <= 3)])
        wr_early = len(roster[(roster['pos'] == 'WR') & (roster['draft_round'] <= 3)])
        
        print("\n🎯 STRATEGY TYPE:")
        if rb_early >= 2:
            print("  RB-Heavy (2+ RBs in first 3 rounds)")
        elif rb_early == 0:
            print("  Zero-RB (No RBs in first 3 rounds)")
        elif wr_early >= 2:
            print("  WR-Heavy (2+ WRs in first 3 rounds)")
        else:
            print("  Balanced (Mixed positions early)")
    
    # Show key patterns
    print("\n" + "="*80)
    print("KEY PATTERNS FROM SUCCESSFUL ROSTERS:")
    print("="*80)
    
    winners = df[df['sim'].isin(winning_sims)]
    
    # Position counts
    avg_by_pos = {}
    for pos in ['RB', 'WR', 'QB', 'TE']:
        counts = []
        for sim in winning_sims:
            sim_roster = winners[winners['sim'] == sim]
            counts.append(len(sim_roster[sim_roster['pos'] == pos]))
        avg_by_pos[pos] = np.mean(counts)
    
    print("\n📊 Average Position Counts in Successful Rosters:")
    for pos, avg in sorted(avg_by_pos.items()):
        print(f"  {pos}: {avg:.1f} players")
    
    # Early round patterns
    print("\n📈 Common Early Round Patterns (Rounds 1-3):")
    early_patterns = {}
    for sim in winning_sims[:20]:  # Sample first 20 winners
        sim_roster = winners[(winners['sim'] == sim) & (winners['draft_round'] <= 3)]
        pattern = '-'.join(sim_roster.sort_values('draft_pick')['pos'].values)
        early_patterns[pattern] = early_patterns.get(pattern, 0) + 1
    
    for pattern, count in sorted(early_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {pattern}: {count} times")
    
    print("\n💡 STARTER vs BENCH TRADE-OFFS:")
    print("  • Rankings based on STARTER POINTS only (what wins weeks)")
    print("  • Bench quality is subjective - you evaluate based on your preferences")
    print("  • High starter rosters may have weak benches (injury risk)")
    print("  • Balanced rosters sacrifice 20-40 starter pts for quality depth")
    print("  • Your choice: Max ceiling (all starters) vs safety (bench depth)")

if __name__ == "__main__":
    import sys
    
    # Parse command line args
    strategy = sys.argv[1] if len(sys.argv) > 1 else 'balanced'
    pick = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    n_examples = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    show_successful_rosters(strategy, pick, n_examples)