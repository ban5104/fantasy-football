#!/usr/bin/env python3
"""
Monte Carlo Position Pattern Discovery - MVP Implementation
Discovers optimal draft position sequences through simulation
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
    'rounds': 15,
    'my_team_idx': 4,  # Pick #5 (0-based indexing)
    'n_sims': 100,  # Reduced for testing - change to 500 for production
    'top_k': 150,
    'espn_weight': 0.8,
    'adp_weight': 0.2,
}

# Roster requirements
STARTER_REQUIREMENTS = {
    'QB': 1,
    'RB': 2,
    'WR': 2,
    'TE': 1,
    'FLEX': 1,  # Best remaining RB/WR/TE
    'K': 1,
    'DST': 1
}

def load_player_data():
    """Load and merge ESPN rankings, ADP data, and projections"""
    # Load ESPN rankings
    espn_df = pd.read_csv('data/espn_projections_20250814.csv')
    espn_df['espn_rank'] = espn_df['overall_rank']
    
    # Load ADP data
    adp_df = pd.read_csv('data/fantasypros_adp_20250815.csv')
    adp_df['adp_rank'] = adp_df['RANK']
    adp_df['player_name'] = adp_df['PLAYER']
    
    # Load projections
    proj_df = pd.read_csv('data/rankings_top300_20250814.csv')
    proj_df['player_name'] = proj_df['PLAYER'].str.replace(r'\s+[A-Z]{2,3}$', '', regex=True).str.strip()
    proj_df['proj'] = proj_df['FANTASY_PTS'].fillna(100)
    
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
    
    # Calculate combined scores for probability
    players_df['espn_score'] = 1.0 / (players_df['espn_rank'] + 1e-6)
    players_df['adp_score'] = 1.0 / (players_df['adp_rank'] + 1e-6)
    players_df['base_score'] = (
        CONFIG['espn_weight'] * players_df['espn_score'] + 
        CONFIG['adp_weight'] * players_df['adp_score']
    )
    
    return players_df

def get_snake_draft_order(n_teams, rounds):
    """Generate snake draft pick order"""
    order = []
    for r in range(rounds):
        if r % 2 == 0:
            order.extend(range(n_teams))
        else:
            order.extend(reversed(range(n_teams)))
    return order

def compute_team_value(chosen_ids, players_df):
    """Calculate total fantasy points from optimal starting lineup"""
    if len(chosen_ids) == 0:
        return 0.0
    
    df = players_df.loc[chosen_ids]
    bypos = {
        p: df[df['pos'] == p].sort_values('proj', ascending=False) 
        for p in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
    }
    
    total = 0.0
    
    # QB x1
    if len(bypos['QB']) > 0:
        total += float(bypos['QB'].iloc[0]['proj'])
    
    # RB x2
    for i in range(min(2, len(bypos['RB']))):
        total += float(bypos['RB'].iloc[i]['proj'])
    
    # WR x2
    for i in range(min(2, len(bypos['WR']))):
        total += float(bypos['WR'].iloc[i]['proj'])
    
    # TE x1
    if len(bypos['TE']) > 0:
        total += float(bypos['TE'].iloc[0]['proj'])
    
    # FLEX: best remaining RB/WR/TE
    flex_pool = []
    if len(bypos['RB']) > 2:
        flex_pool += list(bypos['RB'].iloc[2:]['proj'])
    if len(bypos['WR']) > 2:
        flex_pool += list(bypos['WR'].iloc[2:]['proj'])
    if len(bypos['TE']) > 1:
        flex_pool += list(bypos['TE'].iloc[1:]['proj'])
    if len(flex_pool) > 0:
        total += float(max(flex_pool))
    
    # K x1
    if len(bypos['K']) > 0:
        total += float(bypos['K'].iloc[0]['proj'])
    
    # DST x1
    if len(bypos['DST']) > 0:
        total += float(bypos['DST'].iloc[0]['proj'])
    
    return total

def get_best_player_for_lineup(pool, my_chosen, players_df):
    """Select player who maximizes total starting lineup value"""
    if not pool:
        return None
    
    # If roster is empty, just take highest projected player
    if not my_chosen:
        return max(pool, key=lambda pid: players_df.loc[pid]['proj'])
    
    current_value = compute_team_value(my_chosen, players_df)
    
    best_marginal_value = -1000
    best_player_id = None
    
    # Test each available player
    for player_id in pool:
        test_roster = my_chosen + [player_id]
        new_value = compute_team_value(test_roster, players_df)
        marginal_value = new_value - current_value
        
        if marginal_value > best_marginal_value:
            best_marginal_value = marginal_value
            best_player_id = player_id
    
    # If no player improves lineup, take highest projection
    if best_player_id is None:
        best_player_id = max(pool, key=lambda pid: players_df.loc[pid]['proj'])
    
    return best_player_id

def build_pick_probs(pool, players_df):
    """Build pick probabilities using 80/20 ESPN/ADP weighting"""
    if len(pool) == 0:
        return np.array([])
    
    scores = []
    for pid in pool:
        if pid in players_df.index:
            scores.append(players_df.loc[pid]['base_score'])
        else:
            scores.append(0.0001)  # Tiny probability for missing players
    
    scores = np.array(scores)
    
    # Add small random noise to break ties
    scores = scores + np.random.uniform(0, 0.0001, len(scores))
    
    # Normalize to probabilities
    if scores.sum() > 0:
        probs = scores / scores.sum()
    else:
        probs = np.ones(len(scores)) / len(scores)  # Uniform if all zero
    
    return probs

def simulate_single_draft(sim_idx, my_team_idx, players_df, rng):
    """Execute one complete 15-round draft simulation"""
    
    # Get available player pool (top 150)
    available_pool = list(players_df.nsmallest(CONFIG['top_k'], 'espn_rank').index)
    
    # Track my picks
    my_roster_ids = []
    my_position_sequence = []
    
    # Simulate each pick in snake draft order
    pick_order = get_snake_draft_order(CONFIG['n_teams'], CONFIG['rounds'])
    
    for global_pick_idx, picking_team in enumerate(pick_order):
        if not available_pool:
            break
            
        if picking_team == my_team_idx:
            # MY PICK: Optimize for total starting lineup value
            best_player_id = get_best_player_for_lineup(
                available_pool, my_roster_ids, players_df
            )
            if best_player_id:
                my_roster_ids.append(best_player_id)
                my_position_sequence.append(players_df.loc[best_player_id]['pos'])
                available_pool.remove(best_player_id)
            
        else:
            # OPPONENT PICK: Probabilistic selection
            probabilities = build_pick_probs(available_pool, players_df)
            if len(probabilities) > 0 and probabilities.sum() > 0:
                chosen_idx = rng.choice(len(available_pool), p=probabilities)
                chosen_player_id = available_pool[chosen_idx]
                available_pool.remove(chosen_player_id)
    
    # Calculate final roster value
    final_value = compute_team_value(my_roster_ids, players_df)
    
    return {
        'simulation_id': sim_idx,
        'position_sequence': my_position_sequence,
        'roster_value': final_value,
        'round_1_pos': my_position_sequence[0] if my_position_sequence else None,
        'round_2_pos': my_position_sequence[1] if len(my_position_sequence) > 1 else None,
        'roster_ids': my_roster_ids
    }

def run_position_discovery_simulation():
    """Main execution function"""
    print(f"🏈 Starting Position Pattern Discovery ({CONFIG['n_sims']} simulations)")
    print(f"   Your position: Pick #{CONFIG['my_team_idx']+1} in {CONFIG['n_teams']}-team league")
    print("")
    
    # Load data
    players_df = load_player_data()
    print(f"✓ Loaded {len(players_df)} players")
    
    # Results containers
    all_simulations = []
    round_position_counts = defaultdict(lambda: defaultdict(int))
    pattern_values = defaultdict(list)
    
    # Progress tracking
    start_time = time.time()
    
    for sim_idx in range(CONFIG['n_sims']):
        if sim_idx % 100 == 0 and sim_idx > 0:
            elapsed = time.time() - start_time
            rate = sim_idx / elapsed
            eta = (CONFIG['n_sims'] - sim_idx) / rate
            print(f"  Simulation {sim_idx}/{CONFIG['n_sims']} - ETA: {eta:.0f}s")
        
        # Create RNG for this simulation
        rng = np.random.default_rng(42 + sim_idx)
        
        # Run single complete draft simulation
        result = simulate_single_draft(
            sim_idx, CONFIG['my_team_idx'], players_df, rng
        )
        
        all_simulations.append(result)
        
        # Aggregate data for analysis
        for round_num, position in enumerate(result['position_sequence'], 1):
            round_position_counts[round_num][position] += 1
        
        # Track pattern performance
        if result['round_1_pos'] and result['round_2_pos']:
            opening_pattern = f"{result['round_1_pos']}-{result['round_2_pos']}"
            pattern_values[opening_pattern].append(result['roster_value'])
        
        # Track full sequence patterns (first 6 rounds)
        if len(result['position_sequence']) >= 6:
            full_pattern = '-'.join(result['position_sequence'][:6])
            pattern_values[full_pattern].append(result['roster_value'])
    
    elapsed = time.time() - start_time
    print(f"\n✅ Completed {CONFIG['n_sims']} simulations in {elapsed:.1f} seconds")
    
    # Analyze and return results
    return analyze_position_patterns(
        all_simulations, round_position_counts, pattern_values
    )

def analyze_position_patterns(all_sims, round_counts, pattern_values):
    """Aggregate simulation results into actionable insights"""
    
    n_sims = len(all_sims)
    
    # 1. Round-by-round position frequencies
    round_frequencies = {}
    for round_num in range(1, 16):
        total_picks = sum(round_counts[round_num].values())
        if total_picks > 0:
            round_frequencies[round_num] = {
                pos: count / total_picks 
                for pos, count in round_counts[round_num].items()
            }
    
    # 2. Opening pattern performance analysis
    opening_patterns = {}
    for pattern, values in pattern_values.items():
        if '-' in pattern and len(pattern.split('-')) == 2:  # Two-round patterns
            opening_patterns[pattern] = {
                'frequency': len(values) / n_sims,
                'avg_value': np.mean(values),
                'std_value': np.std(values),
                'count': len(values)
            }
    
    # Sort by average value
    best_opening_patterns = sorted(
        opening_patterns.items(), 
        key=lambda x: x[1]['avg_value'], 
        reverse=True
    )
    
    # 3. Full sequence analysis (6-round patterns)
    sequence_patterns = {}
    for pattern, values in pattern_values.items():
        if len(pattern.split('-')) == 6:  # Six-round patterns
            if len(values) >= 10:  # Minimum sample size
                sequence_patterns[pattern] = {
                    'frequency': len(values) / n_sims,
                    'avg_value': np.mean(values),
                    'std_value': np.std(values),
                    'count': len(values)
                }
    
    best_sequences = sorted(
        sequence_patterns.items(),
        key=lambda x: x[1]['avg_value'],
        reverse=True
    )[:10]
    
    # 4. Summary statistics
    roster_values = [sim['roster_value'] for sim in all_sims]
    summary_stats = {
        'total_simulations': len(all_sims),
        'avg_roster_value': np.mean(roster_values),
        'std_roster_value': np.std(roster_values),
        'min_roster_value': np.min(roster_values),
        'max_roster_value': np.max(roster_values),
        'value_range': np.max(roster_values) - np.min(roster_values)
    }
    
    return {
        'round_frequencies': round_frequencies,
        'opening_patterns': dict(best_opening_patterns[:10]),
        'best_sequences': dict(best_sequences),
        'summary_stats': summary_stats
    }

def display_results(results):
    """Display formatted results"""
    
    print("\n" + "=" * 70)
    print("🏈 POSITION PATTERN DISCOVERY RESULTS")
    print("=" * 70)
    
    # Summary stats
    stats = results['summary_stats']
    print(f"\nTotal Simulations: {stats['total_simulations']}")
    print(f"Average Roster Value: {stats['avg_roster_value']:.1f} ± {stats['std_roster_value']:.1f} points")
    print(f"Value Range: {stats['min_roster_value']:.1f} - {stats['max_roster_value']:.1f} points")
    
    # Round-by-round frequencies
    print("\n📊 ROUND-BY-ROUND POSITION FREQUENCY")
    print("-" * 40)
    for round_num in range(1, 7):  # Show first 6 rounds
        if round_num in results['round_frequencies']:
            freqs = results['round_frequencies'][round_num]
            sorted_freqs = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
            round_str = f"Round {round_num}: "
            position_strs = [f"{pos} ({freq*100:.0f}%)" for pos, freq in sorted_freqs[:4]]
            print(round_str + ", ".join(position_strs))
    
    # Top opening strategies
    print("\n🏆 TOP OPENING STRATEGIES (2-Round Patterns)")
    print("-" * 40)
    for i, (pattern, data) in enumerate(list(results['opening_patterns'].items())[:5], 1):
        print(f"{i}. {pattern}: {data['frequency']*100:.1f}% frequency, "
              f"{data['avg_value']:.1f} ± {data['std_value']:.1f} points")
    
    # Best full sequences
    print("\n🔥 OPTIMAL 6-ROUND SEQUENCES")
    print("-" * 40)
    for i, (pattern, data) in enumerate(list(results['best_sequences'].items())[:5], 1):
        print(f"{i}. {pattern}: {data['frequency']*100:.1f}% frequency, "
              f"{data['avg_value']:.1f} ± {data['std_value']:.1f} points")
    
    # Key insights
    print("\n💡 KEY INSIGHTS")
    print("-" * 40)
    
    # Find best and worst opening strategies
    opening_items = list(results['opening_patterns'].items())
    if len(opening_items) >= 2:
        best_opening = opening_items[0]
        worst_opening = opening_items[-1]
        advantage = best_opening[1]['avg_value'] - worst_opening[1]['avg_value']
        print(f"• {best_opening[0]} outperforms {worst_opening[0]} by {advantage:.1f} points on average")
    
    # QB timing analysis
    qb_rounds = []
    for round_num, freqs in results['round_frequencies'].items():
        if 'QB' in freqs and freqs['QB'] > 0.2:  # QB picked >20% of time
            qb_rounds.append(round_num)
    if qb_rounds:
        print(f"• QB most frequently drafted in rounds: {qb_rounds[:3]}")
    
    # Position diversity
    top_sequences = list(results['best_sequences'].keys())[:10]
    if top_sequences:
        unique_starts = set([seq.split('-')[0] for seq in top_sequences])
        print(f"• Top strategies start with: {', '.join(unique_starts)}")
    
    print("\n" + "=" * 70)

def save_results(results, filename='position_discovery_results.json'):
    """Save results to JSON file"""
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj
    
    serializable_results = convert_types(results)
    
    with open(f'data/draft/{filename}', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✓ Results saved to ../data/draft/{filename}")

def main():
    """Main entry point"""
    results = run_position_discovery_simulation()
    display_results(results)
    save_results(results)
    
    return results

if __name__ == "__main__":
    results = main()