#!/usr/bin/env python3
"""
Enhanced Championship DNA Analysis with Additional Insights
Adds: Scarcity Cliffs, Draft Paths, Position Runs, Recovery Patterns, Correlations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

class EnhancedChampionshipDNA:
    """Enhanced DNA analysis with actionable draft insights"""
    
    def __init__(self):
        self.scarcity_cliffs = {}
        self.winning_paths = []
        self.position_runs = {}
        self.recovery_patterns = {}
        self.correlations = {}
        
    def analyze_scarcity_cliffs(self, df: pd.DataFrame, tiers: Dict) -> Dict:
        """
        Find when each position tier typically becomes unavailable
        Returns pick numbers where 80% of tier is gone
        """
        cliffs = defaultdict(dict)
        
        # Group by position and tier
        for pos in ['RB', 'WR', 'QB', 'TE']:
            if pos not in tiers:
                continue
                
            pos_df = df[df['pos'] == pos].copy()
            
            for tier_name in ['tier_1', 'tier_2', 'tier_3']:
                tier_threshold = tiers[pos].get(tier_name, 0)
                if tier_threshold == float('inf') or tier_threshold == 0:
                    continue
                    
                # Find players in this tier
                tier_players = pos_df[pos_df['sampled_points'] >= tier_threshold]
                if len(tier_players) == 0:
                    continue
                
                # Find when 80% are drafted
                picks = tier_players.groupby('sim')['draft_pick'].apply(
                    lambda x: np.percentile(x, 80) if len(x) > 0 else 200
                )
                
                cliff_pick = int(picks.median())
                if cliff_pick < 200:  # Valid pick
                    cliffs[pos][tier_name] = {
                        'pick': cliff_pick,
                        'round': (cliff_pick - 1) // 14 + 1,
                        'pct_gone': 80
                    }
        
        self.scarcity_cliffs = dict(cliffs)
        return self.scarcity_cliffs
    
    def analyze_winning_paths(self, df: pd.DataFrame, my_pick: int = 5) -> List[Dict]:
        """
        Identify most common successful draft paths (first 4 rounds)
        """
        # Get top 10% rosters
        threshold = df.groupby('sim')['roster_value'].first().quantile(0.9)
        winning_sims = df[df['roster_value'] >= threshold]['sim'].unique()
        winners = df[df['sim'].isin(winning_sims)]
        
        paths = []
        for sim in winning_sims:
            sim_picks = winners[winners['sim'] == sim].sort_values('draft_pick')
            # Get first 4 rounds for this team
            first_4 = sim_picks[sim_picks['draft_round'] <= 4]['pos'].tolist()
            if len(first_4) == 4:
                paths.append('→'.join(first_4))
        
        # Count path frequencies
        path_counts = Counter(paths)
        total = len(paths)
        
        self.winning_paths = [
            {
                'path': path,
                'count': count,
                'win_rate': count / total * 100,
                'positions': path.split('→')
            }
            for path, count in path_counts.most_common(10)
            if count >= total * 0.05  # At least 5% frequency
        ]
        
        return self.winning_paths
    
    def analyze_position_runs(self, df: pd.DataFrame) -> Dict:
        """
        Detect when position runs typically occur and their magnitude
        """
        runs = defaultdict(list)
        
        # Analyze each simulation
        for sim in df['sim'].unique():
            sim_df = df[df['sim'] == sim].sort_values('draft_pick')
            
            for pos in ['RB', 'WR', 'QB', 'TE']:
                pos_picks = sim_df[sim_df['pos'] == pos]['draft_pick'].tolist()
                
                # Detect runs (3+ of same position in 8 picks)
                for i in range(len(pos_picks) - 2):
                    window_start = pos_picks[i]
                    window_end = window_start + 8
                    
                    picks_in_window = [p for p in pos_picks[i:i+5] if p <= window_end]
                    if len(picks_in_window) >= 3:
                        runs[pos].append({
                            'trigger_pick': window_start,
                            'count': len(picks_in_window),
                            'window': 8
                        })
        
        # Aggregate patterns
        self.position_runs = {}
        for pos, run_list in runs.items():
            if run_list:
                trigger_picks = [r['trigger_pick'] for r in run_list]
                avg_count = np.mean([r['count'] for r in run_list])
                
                self.position_runs[pos] = {
                    'typical_trigger': int(np.median(trigger_picks)),
                    'trigger_round': (int(np.median(trigger_picks)) - 1) // 14 + 1,
                    'avg_picks_in_run': round(avg_count, 1),
                    'frequency': len(run_list) / df['sim'].nunique() * 100
                }
        
        return self.position_runs
    
    def analyze_correlations(self, df: pd.DataFrame) -> Dict:
        """
        Find correlated draft patterns in winning rosters
        """
        # Get top 10% rosters
        threshold = df.groupby('sim')['roster_value'].first().quantile(0.9)
        winning_sims = df[df['roster_value'] >= threshold]['sim'].unique()
        winners = df[df['sim'].isin(winning_sims)]
        
        correlations = []
        
        for sim in winning_sims:
            roster = winners[winners['sim'] == sim]
            
            # Check various patterns
            early_te = any((roster['pos'] == 'TE') & (roster['draft_round'] <= 4))
            late_qb = any((roster['pos'] == 'QB') & (roster['draft_round'] >= 6))
            early_qb = any((roster['pos'] == 'QB') & (roster['draft_round'] <= 5))
            rb_heavy = len(roster[roster['pos'] == 'RB']) >= 5
            
            correlations.append({
                'early_te_late_qb': early_te and late_qb,
                'early_qb_no_early_te': early_qb and not early_te,
                'rb_heavy_early_te': rb_heavy and early_te
            })
        
        # Calculate correlation percentages
        corr_df = pd.DataFrame(correlations)
        self.correlations = {
            'Early TE → Late QB': corr_df['early_te_late_qb'].mean() * 100,
            'Early QB → No Early TE': corr_df['early_qb_no_early_te'].mean() * 100,
            'RB-Heavy → Early TE': corr_df['rb_heavy_early_te'].mean() * 100
        }
        
        return self.correlations
    
    def print_enhanced_insights(self):
        """Print all enhanced insights in a clean format"""
        
        # Scarcity Cliffs
        print("\n" + "="*60)
        print("🏔️ SCARCITY CLIFFS (80% Gone)")
        print("="*60)
        for pos, tiers in self.scarcity_cliffs.items():
            for tier, info in sorted(tiers.items()):
                print(f"{pos} {tier}: Pick {info['pick']} (Round {info['round']})")
        
        # Winning Paths
        print("\n" + "="*60)
        print("🛤️ TOP WINNING DRAFT PATHS")
        print("="*60)
        for i, path in enumerate(self.winning_paths[:5], 1):
            stars = "⭐" if path['win_rate'] > 35 else ""
            print(f"{i}. {path['path']} ({path['win_rate']:.0f}% win rate) {stars}")
        
        # Position Runs
        print("\n" + "="*60)
        print("🚨 POSITION RUN PATTERNS")
        print("="*60)
        for pos, info in self.position_runs.items():
            if info['frequency'] > 20:  # Only show common runs
                print(f"{pos} Run: Typically starts pick {info['typical_trigger']} "
                      f"(Round {info['trigger_round']}), "
                      f"avg {info['avg_picks_in_run']:.1f} picks")
        
        # Correlations
        print("\n" + "="*60)
        print("🎯 WINNING CORRELATIONS")
        print("="*60)
        for pattern, pct in sorted(self.correlations.items(), 
                                  key=lambda x: x[1], reverse=True):
            if pct > 50:  # Only show strong correlations
                print(f"{pattern}: {pct:.0f}% of winners")


def run_enhanced_analysis(pick: int = 5, n_sims: int = 200):
    """Run the enhanced Championship DNA analysis"""
    
    print("🧬 Enhanced Championship DNA Analysis")
    print("="*60)
    
    # Load simulation data
    cache_dir = Path('data/cache')
    sim_file = cache_dir / f'balanced_pick{pick}_n{n_sims}_r14.parquet'
    
    if not sim_file.exists():
        print(f"❌ No simulation data found at {sim_file}")
        print(f"Run: python monte_carlo_runner.py export --strategy balanced --pick {pick} --n-sims {n_sims}")
        return None
    
    df = pd.read_parquet(sim_file)
    
    # Load tier definitions
    from src.monte_carlo import DraftSimulator
    sim = DraftSimulator()
    sim.prob_model.load_data()
    
    # Define tiers (simplified for demo)
    tiers = {}
    for pos in ['RB', 'WR', 'QB', 'TE']:
        pos_players = sim.prob_model.players_df[sim.prob_model.players_df['pos'] == pos]
        pos_players = pos_players.sort_values('proj', ascending=False)
        
        n = len(pos_players)
        tiers[pos] = {
            'tier_1': pos_players.iloc[:max(1, int(n*0.05))]['proj'].min() if n > 0 else float('inf'),
            'tier_2': pos_players.iloc[max(1, int(n*0.05)):int(n*0.20)]['proj'].min() if n > 5 else float('inf'),
            'tier_3': pos_players.iloc[int(n*0.20):int(n*0.50)]['proj'].min() if n > 10 else float('inf')
        }
    
    # Run enhanced analysis
    analyzer = EnhancedChampionshipDNA()
    analyzer.analyze_scarcity_cliffs(df, tiers)
    analyzer.analyze_winning_paths(df, pick)
    analyzer.analyze_position_runs(df)
    analyzer.analyze_correlations(df)
    
    # Print insights
    analyzer.print_enhanced_insights()
    
    return analyzer


if __name__ == "__main__":
    analyzer = run_enhanced_analysis(pick=5, n_sims=200)