#!/usr/bin/env python3
"""
SIMPLE Championship DNA Runner
This is what you actually want - find what WINNING rosters look like!
No complex strategies, just patterns in the top teams.
"""

import sys
import os
sys.path.append('..')

from src.monte_carlo import DraftSimulator
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

def run_championship_dna_analysis(my_pick=5, n_sims=2000, top_percent=10):
    """
    The simplest possible version - just find winning patterns
    
    Args:
        my_pick: Your draft position (1-14)
        n_sims: How many simulations to run
        top_percent: What percentage to consider "winners" (10 = top 10%)
    """
    
    print("\n" + "="*70)
    print("🧬 CHAMPIONSHIP DNA ANALYSIS")
    print("="*70)
    print(f"Position: Pick #{my_pick}")
    print(f"Simulations: {n_sims}")
    print(f"Analyzing: Top {top_percent}% of teams")
    print("-"*70)
    
    # Step 1: Run simulations
    print("\n🎲 Running simulations...")
    sim = DraftSimulator()
    
    # Run with balanced strategy (or mix strategies for variety)
    result = sim.simulator.run_simulations(
        my_team_idx=my_pick - 1,
        strategy_name='balanced',
        n_sims=n_sims,
        base_seed=42
    )
    
    all_rosters = result.get('all_results', [])
    print(f"   Generated {len(all_rosters)} rosters")
    
    # Step 2: Find the winners
    values = [r['roster_value'] for r in all_rosters]
    cutoff = np.percentile(values, 100 - top_percent)
    winners = [r for r in all_rosters if r['roster_value'] >= cutoff]
    
    print(f"\n🏆 Analyzing {len(winners)} winning rosters (value >= {cutoff:.1f})")
    
    # Step 3: Analyze winning roster composition
    print("\n" + "="*70)
    print("📊 WHAT WINNING ROSTERS LOOK LIKE:")
    print("="*70)
    
    # Position counts
    position_counts = defaultdict(list)
    for roster in winners:
        pos_count = defaultdict(int)
        for player in roster['roster']:
            pos_count[player['pos']] += 1
        
        for pos, count in pos_count.items():
            position_counts[pos].append(count)
    
    print("\n📋 POSITION COMPOSITION (winners average):")
    print("-"*40)
    for pos in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
        if pos in position_counts:
            counts = position_counts[pos]
            avg = np.mean(counts)
            mode = max(set(counts), key=counts.count)
            print(f"{pos:4} → {avg:.1f} players (most common: {mode})")
    
    # Early round patterns
    print("\n🔄 MOST SUCCESSFUL DRAFT STARTS (first 3 rounds):")
    print("-"*40)
    early_patterns = []
    for roster in winners:
        if 'position_sequence' in roster and len(roster['position_sequence']) >= 3:
            pattern = '-'.join(roster['position_sequence'][:3])
            early_patterns.append(pattern)
    
    if early_patterns:
        pattern_counts = Counter(early_patterns)
        total = len(early_patterns)
        
        for i, (pattern, count) in enumerate(pattern_counts.most_common(5), 1):
            freq = count / total * 100
            positions = pattern.split('-')
            print(f"{i}. {' → '.join(positions):20} ({freq:.1f}% of winners)")
    
    # Tier analysis (simplified - based on projection values)
    print("\n🎯 VALUE TIERS IN WINNING ROSTERS:")
    print("-"*40)
    
    # Collect all player projections by position
    position_values = defaultdict(list)
    for roster in winners:
        for player in roster['roster']:
            position_values[player['pos']].append(player['proj'])
    
    # Show percentiles for each position
    for pos in ['QB', 'RB', 'WR', 'TE']:
        if pos in position_values and position_values[pos]:
            values = sorted(position_values[pos], reverse=True)
            
            # Get the best, 2nd best, 3rd best typical values
            best = values[0] if len(values) > 0 else 0
            second = values[1] if len(values) > 1 else 0
            third = values[2] if len(values) > 2 else 0
            
            print(f"{pos}: Best={best:.0f}, 2nd={second:.0f}, 3rd={third:.0f} pts")
    
    # Key insights
    print("\n" + "="*70)
    print("💡 KEY INSIGHTS FOR YOUR DRAFT:")
    print("="*70)
    
    insights = []
    
    # RB insights
    if 'RB' in position_counts:
        rb_avg = np.mean(position_counts['RB'])
        rb_mode = max(set(position_counts['RB']), key=position_counts['RB'].count)
        if rb_avg >= 3.5:
            insights.append(f"• Draft {rb_mode} RBs total - depth matters! (winners avg {rb_avg:.1f})")
        
        # Check if early RB matters
        rb_first_2 = sum(1 for p in early_patterns if 'RB' in p[:5])
        if rb_first_2 / len(early_patterns) > 0.6:
            insights.append(f"• {rb_first_2/len(early_patterns)*100:.0f}% of winners take RB in first 2 rounds")
    
    # WR insights
    if 'WR' in position_counts:
        wr_avg = np.mean(position_counts['WR'])
        if wr_avg >= 3:
            insights.append(f"• Load up on WRs - winners average {wr_avg:.1f}")
    
    # TE insights
    early_te = sum(1 for p in early_patterns if 'TE' in p)
    if early_te / len(early_patterns) > 0.3:
        insights.append(f"• {early_te/len(early_patterns)*100:.0f}% of winners take TE early (rounds 1-3)")
    
    # QB insights
    early_qb = sum(1 for p in early_patterns if 'QB' in p)
    if early_qb / len(early_patterns) < 0.2:
        insights.append(f"• Only {early_qb/len(early_patterns)*100:.0f}% take QB early - wait on QB")
    
    for insight in insights:
        print(insight)
    
    # Draft rules
    print("\n🎯 YOUR DRAFT RULES:")
    print("-"*40)
    
    # Generate simple, actionable rules
    rules = []
    
    # Most common successful pattern
    if pattern_counts:
        top_pattern = pattern_counts.most_common(1)[0][0]
        positions = top_pattern.split('-')
        rules.append(f"Consider starting: {' → '.join(positions)} (highest win rate)")
    
    # Position requirements
    for pos in ['RB', 'WR']:
        if pos in position_counts:
            mode = max(set(position_counts[pos]), key=position_counts[pos].count)
            rules.append(f"Target {mode} {pos}s total")
    
    # Value thresholds
    if 'QB' in position_values and position_values['QB']:
        qb_threshold = np.percentile(position_values['QB'], 75)
        rules.append(f"QB with {qb_threshold:.0f}+ projected points is sufficient")
    
    for i, rule in enumerate(rules[:5], 1):
        print(f"{i}. {rule}")
    
    print("\n" + "="*70)
    print("✅ Analysis complete! Focus on these patterns, not strategy labels.")
    print("="*70)
    
    return {
        'winners': winners,
        'position_counts': position_counts,
        'early_patterns': pattern_counts if early_patterns else {},
        'insights': insights
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Find Championship DNA - What do winning teams look like?'
    )
    parser.add_argument('--pick', type=int, default=5, 
                       help='Your draft position (1-14)')
    parser.add_argument('--sims', type=int, default=1000,
                       help='Number of simulations to run')
    parser.add_argument('--top', type=int, default=10,
                       help='Top X percent to analyze as winners')
    
    args = parser.parse_args()
    
    results = run_championship_dna_analysis(args.pick, args.sims, args.top)