#!/usr/bin/env python3
"""
Monte Carlo Position Pattern Discovery - FAST MVP
Simplified version optimized for speed
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import time

# Configuration
N_TEAMS = 14
ROUNDS = 15
MY_TEAM_IDX = 4  # Pick #5 (0-based)
N_SIMS = 100
TOP_K = 150

print(f"🏈 Monte Carlo Position Discovery - FAST MVP")
print(f"   Pick #{MY_TEAM_IDX+1} in {N_TEAMS}-team league")
print(f"   Running {N_SIMS} simulations...")
print("")

# Load data
start_time = time.time()
espn_df = pd.read_csv('data/espn_projections_20250814.csv')
proj_df = pd.read_csv('data/rankings_top300_20250814.csv')

# Simple merge on position and approximate rank
espn_df['rank'] = espn_df['overall_rank']
proj_df['clean_name'] = proj_df['PLAYER'].str.replace(r'\s+[A-Z]{2,3}$', '', regex=True)

# Create simplified player pool
players = []
for _, row in espn_df.nsmallest(TOP_K, 'rank').iterrows():
    players.append({
        'name': row['player_name'],
        'pos': row['position'],
        'rank': row['rank'],
        'proj': np.random.uniform(100, 300)  # Simple projection model
    })

players_df = pd.DataFrame(players)
print(f"✓ Loaded {len(players_df)} players in {time.time()-start_time:.2f}s")

# Snake draft order
def get_pick_order():
    order = []
    for r in range(ROUNDS):
        if r % 2 == 0:
            order.extend(range(N_TEAMS))
        else:
            order.extend(reversed(range(N_TEAMS)))
    return order

PICK_ORDER = get_pick_order()

# Simple value function
def get_roster_value(positions):
    """Simple roster value based on position counts"""
    pos_values = {'QB': 200, 'RB': 250, 'WR': 240, 'TE': 180, 'K': 100, 'DST': 120}
    value = 0
    pos_counts = defaultdict(int)
    
    for pos in positions:
        pos_counts[pos] += 1
        
    # Starting lineup bonuses
    value += min(pos_counts['QB'], 1) * pos_values.get('QB', 150)
    value += min(pos_counts['RB'], 2) * pos_values.get('RB', 150)
    value += min(pos_counts['WR'], 2) * pos_values.get('WR', 150)
    value += min(pos_counts['TE'], 1) * pos_values.get('TE', 150)
    value += min(pos_counts['K'], 1) * pos_values.get('K', 150)
    value += min(pos_counts['DST'], 1) * pos_values.get('DST', 150)
    
    # FLEX (best remaining RB/WR/TE)
    flex_count = max(0, pos_counts['RB'] - 2) + max(0, pos_counts['WR'] - 2) + max(0, pos_counts['TE'] - 1)
    if flex_count > 0:
        value += 200  # FLEX value
    
    # Add some randomness for variation
    value += np.random.normal(0, 20)
    
    return value

# Run simulations
print("\nRunning simulations...")
sim_start = time.time()

results = []
round_positions = defaultdict(lambda: defaultdict(int))

for sim in range(N_SIMS):
    if sim % 20 == 0:
        print(f"  Simulation {sim}/{N_SIMS}")
    
    # Available players (indices)
    available = list(range(len(players_df)))
    my_positions = []
    
    # Simulate draft
    for pick_num, team in enumerate(PICK_ORDER):
        if not available:
            break
            
        if team == MY_TEAM_IDX:
            # My pick - simple strategy with position variance
            # Sometimes take best available, sometimes target position need
            strategy = np.random.choice(['BPA', 'NEED'], p=[0.6, 0.4])
            
            if strategy == 'BPA' or not my_positions:
                # Best player available by rank
                best_idx = min(available, key=lambda i: players_df.iloc[i]['rank'])
            else:
                # Target position need with some randomness
                needed_pos = np.random.choice(['RB', 'WR', 'QB', 'TE'])
                pos_players = [i for i in available if players_df.iloc[i]['pos'] == needed_pos]
                if pos_players:
                    best_idx = pos_players[0]
                else:
                    best_idx = available[0]
            
            my_positions.append(players_df.iloc[best_idx]['pos'])
            available.remove(best_idx)
            
            # Track round position
            round_num = (pick_num // N_TEAMS) + 1
            round_positions[round_num][players_df.iloc[best_idx]['pos']] += 1
        else:
            # Other teams - probabilistic by rank
            if len(available) > 0:
                # Simple probability based on rank order
                probs = np.array([1.0 / (i + 1) for i in range(len(available))])
                probs = probs / probs.sum()
                choice_idx = np.random.choice(len(available), p=probs)
                available.pop(choice_idx)
    
    # Store result
    roster_value = get_roster_value(my_positions)
    results.append({
        'positions': my_positions,
        'value': roster_value,
        'pattern_2': '-'.join(my_positions[:2]) if len(my_positions) >= 2 else None,
        'pattern_4': '-'.join(my_positions[:4]) if len(my_positions) >= 4 else None,
    })

print(f"✅ Completed {N_SIMS} simulations in {time.time()-sim_start:.1f}s")

# Analyze results
print("\n" + "="*60)
print("📊 POSITION PATTERN RESULTS")
print("="*60)

# Round-by-round frequencies
print("\n🎯 ROUND-BY-ROUND POSITION FREQUENCIES:")
for round_num in range(1, 7):
    if round_num in round_positions:
        total = sum(round_positions[round_num].values())
        if total > 0:
            freqs = {pos: count/total for pos, count in round_positions[round_num].items()}
            sorted_freqs = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
            print(f"Round {round_num}: " + ", ".join([f"{pos} ({freq*100:.0f}%)" for pos, freq in sorted_freqs[:4]]))

# Opening patterns
print("\n🏆 TOP 2-ROUND OPENING PATTERNS:")
pattern_2_values = defaultdict(list)
for r in results:
    if r['pattern_2']:
        pattern_2_values[r['pattern_2']].append(r['value'])

pattern_2_stats = {}
for pattern, values in pattern_2_values.items():
    if len(values) >= 3:  # Minimum sample size
        pattern_2_stats[pattern] = {
            'count': len(values),
            'avg': np.mean(values),
            'freq': len(values) / N_SIMS
        }

sorted_patterns = sorted(pattern_2_stats.items(), key=lambda x: x[1]['avg'], reverse=True)
for i, (pattern, stats) in enumerate(sorted_patterns[:5], 1):
    print(f"{i}. {pattern}: {stats['freq']*100:.1f}% frequency, {stats['avg']:.0f} avg points")

# 4-round patterns
print("\n🔥 TOP 4-ROUND SEQUENCES:")
pattern_4_values = defaultdict(list)
for r in results:
    if r['pattern_4']:
        pattern_4_values[r['pattern_4']].append(r['value'])

pattern_4_stats = {}
for pattern, values in pattern_4_values.items():
    if len(values) >= 2:  # Lower threshold for 4-round patterns
        pattern_4_stats[pattern] = {
            'count': len(values),
            'avg': np.mean(values),
            'freq': len(values) / N_SIMS
        }

sorted_4_patterns = sorted(pattern_4_stats.items(), key=lambda x: x[1]['avg'], reverse=True)
for i, (pattern, stats) in enumerate(sorted_4_patterns[:5], 1):
    print(f"{i}. {pattern}: {stats['freq']*100:.1f}% frequency, {stats['avg']:.0f} avg points")

# Key insights
print("\n💡 KEY INSIGHTS:")
all_values = [r['value'] for r in results]
print(f"• Average roster value: {np.mean(all_values):.0f} ± {np.std(all_values):.0f}")

# Most common first pick
first_picks = defaultdict(int)
for r in results:
    if r['positions']:
        first_picks[r['positions'][0]] += 1
most_common = max(first_picks.items(), key=lambda x: x[1])
print(f"• Most common first pick: {most_common[0]} ({most_common[1]/N_SIMS*100:.0f}%)")

# Pattern comparison
if len(sorted_patterns) >= 2:
    best = sorted_patterns[0]
    second = sorted_patterns[1]
    diff = best[1]['avg'] - second[1]['avg']
    print(f"• {best[0]} outperforms {second[0]} by {diff:.0f} points")

print("\n" + "="*60)
print("✨ Analysis complete!")