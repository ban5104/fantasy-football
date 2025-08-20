#!/usr/bin/env python3
"""
Championship DNA Live Draft Assistant
Tells you EXACTLY what rank players you need at each point in the draft
"""

import sys
import os
sys.path.append('..')

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import json
from src.monte_carlo import DraftSimulator

class ChampionshipDNALive:
    """Live draft assistant based on championship patterns"""
    
    def __init__(self, my_pick=5, winner_definition='balanced'):
        self.my_pick = my_pick
        self.sim = DraftSimulator()
        self.winner_definition = winner_definition
        self.championship_dna = None
        self.rank_targets = {}
        
    def define_winners(self, rosters, method='balanced'):
        """
        Multiple ways to define "winning" teams
        
        Methods:
        - 'starter_points': Top teams by starting lineup only
        - 'total_value': Starters + depth bonus (default)
        - 'balanced': Good starters AND depth
        - 'upside': High ceiling (90th percentile outcomes)
        - 'consistency': Low variance teams
        """
        
        if method == 'starter_points':
            scores = [r['starter_points'] for r in rosters]
        elif method == 'total_value':
            scores = [r['roster_value'] for r in rosters]
        elif method == 'balanced':
            # Require both good starters AND depth
            scores = []
            for r in rosters:
                starter_score = r['starter_points']
                depth_score = r.get('depth_bonus', 0)
                # Weight starters more but require some depth
                scores.append(starter_score + min(depth_score, starter_score * 0.2))
        elif method == 'upside':
            # Would need to run multiple projections per roster
            scores = [r['roster_value'] * 1.1 for r in rosters]  # Simplified
        else:
            scores = [r['roster_value'] for r in rosters]
        
        return scores
    
    def analyze_rank_targets(self, n_sims=1000):
        """
        Discover what RANK of players winners actually draft
        """
        print(f"🎯 Analyzing rank targets for pick #{self.my_pick}...")
        
        # Run simulations
        result = self.sim.simulator.run_simulations(
            my_team_idx=self.my_pick - 1,
            strategy_name='balanced',
            n_sims=n_sims,
            base_seed=42
        )
        
        rosters = result.get('all_results', [])
        
        # Define winners based on chosen method
        scores = self.define_winners(rosters, self.winner_definition)
        cutoff = np.percentile(scores, 90)  # Top 10%
        
        winners = []
        for i, roster in enumerate(rosters):
            if scores[i] >= cutoff:
                winners.append(roster)
        
        print(f"📊 Analyzing {len(winners)} winning rosters (top 10%)")
        
        # Extract rank patterns by round
        round_patterns = defaultdict(lambda: defaultdict(list))
        
        for winner in winners:
            if 'roster' not in winner:
                continue
                
            # Sort roster by ESPN rank to approximate draft order
            roster_sorted = sorted(
                winner['roster'], 
                key=lambda x: x.get('espn_rank', 999) if hasattr(x, 'get') else 999
            )
            
            # Assign to rounds (first 14 picks = 14 rounds)
            for round_num, player in enumerate(roster_sorted[:14], 1):
                if isinstance(player, dict):
                    pos = player.get('pos', 'UNKNOWN')
                    rank = player.get('espn_rank', 999)
                    
                    # Track position rank within position
                    pos_rank = self._get_position_rank(player, roster_sorted)
                    round_patterns[round_num][pos].append({
                        'overall_rank': rank,
                        'position_rank': pos_rank
                    })
        
        # Calculate rank ranges for each round/position
        self.rank_targets = {}
        
        for round_num in range(1, 8):  # Focus on first 7 rounds
            self.rank_targets[round_num] = {}
            
            for pos in ['RB', 'WR', 'TE', 'QB']:
                if pos in round_patterns[round_num]:
                    ranks = round_patterns[round_num][pos]
                    if len(ranks) >= 3:  # Need enough data
                        overall_ranks = [r['overall_rank'] for r in ranks]
                        pos_ranks = [r['position_rank'] for r in ranks]
                        
                        self.rank_targets[round_num][pos] = {
                            'overall': {
                                'min': int(np.percentile(overall_ranks, 10)),
                                'max': int(np.percentile(overall_ranks, 90)),
                                'median': int(np.median(overall_ranks))
                            },
                            'position': {
                                'min': int(np.percentile(pos_ranks, 10)),
                                'max': int(np.percentile(pos_ranks, 90)),
                                'median': int(np.median(pos_ranks))
                            },
                            'frequency': len(ranks) / len(winners)
                        }
        
        # Store full DNA analysis
        self.championship_dna = {
            'winners': len(winners),
            'total_sims': n_sims,
            'rank_targets': self.rank_targets,
            'winner_definition': self.winner_definition,
            'cutoff_value': cutoff
        }
        
        return self.rank_targets
    
    def _get_position_rank(self, player, all_players):
        """Calculate position rank for a player"""
        pos = player.get('pos', 'UNKNOWN')
        pos_players = [p for p in all_players if p.get('pos') == pos]
        pos_players.sort(key=lambda x: x.get('proj', 0), reverse=True)
        
        for i, p in enumerate(pos_players, 1):
            if p.get('name') == player.get('name'):
                return i
        return 99
    
    def get_draft_targets(self):
        """
        Generate specific draft targets by round
        """
        if not self.rank_targets:
            self.analyze_rank_targets()
        
        print("\n" + "="*70)
        print("🎯 YOUR DRAFT TARGETS BY ROUND")
        print(f"   Based on top 10% rosters using '{self.winner_definition}' scoring")
        print("="*70)
        
        for round_num in range(1, 8):
            if round_num not in self.rank_targets:
                continue
                
            print(f"\n📍 ROUND {round_num}:")
            print("-"*40)
            
            # Sort by frequency
            round_targets = self.rank_targets[round_num]
            sorted_positions = sorted(
                round_targets.items(),
                key=lambda x: x[1]['frequency'],
                reverse=True
            )
            
            for pos, data in sorted_positions[:3]:  # Top 3 options
                if data['frequency'] >= 0.1:  # At least 10% of winners
                    overall = data['overall']
                    position = data['position']
                    freq = data['frequency'] * 100
                    
                    print(f"{pos:3} → Overall rank {overall['min']}-{overall['max']} "
                          f"(Position rank {position['min']}-{position['max']}) "
                          f"[{freq:.0f}% of winners]")
        
        return self.rank_targets
    
    def get_live_recommendation(self, current_roster=None, already_drafted=None):
        """
        Live draft recommendation based on current state
        """
        print("\n" + "="*70)
        print("🔴 LIVE DRAFT RECOMMENDATION")
        print("="*70)
        
        # Load current draft state
        state = self.sim.load_draft_state()
        if state:
            current_roster = state.get('my_current_roster', current_roster)
            already_drafted = state.get('all_drafted', already_drafted)
            current_pick = state.get('current_global_pick', 0) + 1
            print(f"📍 Current pick: #{current_pick}")
            print(f"📋 Your roster: {current_roster}")
        
        # Determine current round
        if already_drafted:
            round_num = (len(already_drafted) // 14) + 1
        else:
            round_num = 1
        
        print(f"📍 Round {round_num}")
        
        # Run quick simulation from current state
        print("\n🎲 Running 100 simulations from current position...")
        
        # Test each position to see what wins from here
        position_values = {}
        for test_pos in ['RB', 'WR', 'TE', 'QB']:
            # Run mini simulation assuming we take this position
            result = self.sim.simulator.run_simulations(
                my_team_idx=self.my_pick - 1,
                strategy_name='balanced',
                n_sims=50,
                initial_roster=current_roster,
                already_drafted=already_drafted
            )
            position_values[test_pos] = result['mean_value']
        
        # Sort by value
        sorted_positions = sorted(
            position_values.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        print("\n🎯 POSITION VALUES FROM HERE:")
        print("-"*40)
        for pos, value in sorted_positions:
            print(f"{pos}: {value:.1f} expected points")
        
        # Get specific player recommendations
        if self.rank_targets and round_num in self.rank_targets:
            print("\n📊 TARGET RANKS THIS ROUND:")
            print("-"*40)
            
            best_pos = sorted_positions[0][0]
            if best_pos in self.rank_targets[round_num]:
                target = self.rank_targets[round_num][best_pos]
                print(f"✅ RECOMMENDATION: Draft {best_pos}")
                print(f"   Overall rank: {target['overall']['min']}-{target['overall']['max']}")
                print(f"   Position rank: {target['position']['min']}-{target['position']['max']}")
            else:
                print(f"✅ RECOMMENDATION: Draft {best_pos}")
                print(f"   (No specific rank target, pick best available)")
        
        return sorted_positions[0][0] if sorted_positions else 'BPA'
    
    def show_winner_definitions(self, n_sims=500):
        """
        Show how different definitions of 'winning' change recommendations
        """
        print("\n" + "="*70)
        print("🏆 COMPARING WINNER DEFINITIONS")
        print("="*70)
        
        # Run simulations once
        result = self.sim.simulator.run_simulations(
            my_team_idx=self.my_pick - 1,
            strategy_name='balanced', 
            n_sims=n_sims
        )
        rosters = result.get('all_results', [])
        
        # Compare different winner definitions
        definitions = ['starter_points', 'total_value', 'balanced']
        comparisons = {}
        
        for defn in definitions:
            scores = self.define_winners(rosters, defn)
            cutoff = np.percentile(scores, 90)
            
            # Find what positions dominate early rounds
            winners = [r for i, r in enumerate(rosters) if scores[i] >= cutoff]
            
            # Count first 3 picks
            early_positions = []
            for winner in winners:
                if 'position_sequence' in winner:
                    early_positions.extend(winner['position_sequence'][:3])
            
            pos_counts = Counter(early_positions)
            total = sum(pos_counts.values())
            
            comparisons[defn] = {
                'cutoff': cutoff,
                'winner_count': len(winners),
                'early_positions': {
                    pos: count/total*100 
                    for pos, count in pos_counts.most_common(4)
                }
            }
        
        # Display comparison
        for defn, data in comparisons.items():
            print(f"\n📊 {defn.upper()} (cutoff: {data['cutoff']:.0f}):")
            print(f"   Winners: {data['winner_count']}")
            print("   Early rounds focus:")
            for pos, pct in data['early_positions'].items():
                print(f"   - {pos}: {pct:.0f}%")
        
        print("\n💡 INSIGHT:")
        print("- 'starter_points': Focuses on guaranteed production")
        print("- 'total_value': Balances starters and depth")  
        print("- 'balanced': Requires both good starters AND bench")
        
        return comparisons


def run_pre_draft_analysis(my_pick=5):
    """Complete pre-draft preparation"""
    
    print("\n" + "="*70)
    print("🏈 PRE-DRAFT CHAMPIONSHIP DNA ANALYSIS")
    print("="*70)
    
    assistant = ChampionshipDNALive(my_pick=my_pick, winner_definition='balanced')
    
    # 1. Analyze rank targets
    print("\n1️⃣ ANALYZING RANK TARGETS...")
    assistant.analyze_rank_targets(n_sims=500)
    assistant.get_draft_targets()
    
    # 2. Show different winner definitions
    print("\n2️⃣ COMPARING WINNER DEFINITIONS...")
    assistant.show_winner_definitions(n_sims=300)
    
    # 3. Save for draft day
    with open(f'draft_day_targets_pick{my_pick}.json', 'w') as f:
        json.dump(assistant.championship_dna, f, indent=2, default=str)
    
    print(f"\n✅ Saved targets to draft_day_targets_pick{my_pick}.json")
    print("📱 Keep this open during your draft!")
    
    return assistant


def run_live_draft_assistant(my_pick=5):
    """Use during live draft"""
    
    print("\n" + "="*70)
    print("🔴 LIVE DRAFT ASSISTANT")
    print("="*70)
    
    assistant = ChampionshipDNALive(my_pick=my_pick)
    
    # Load pre-computed targets if available
    try:
        with open(f'draft_day_targets_pick{my_pick}.json', 'r') as f:
            assistant.championship_dna = json.load(f)
            assistant.rank_targets = assistant.championship_dna.get('rank_targets', {})
            # Convert string keys back to int
            assistant.rank_targets = {
                int(k): v for k, v in assistant.rank_targets.items()
            }
        print("✅ Loaded pre-computed targets")
    except:
        print("⚠️ No pre-computed targets, running quick analysis...")
        assistant.analyze_rank_targets(n_sims=200)
    
    # Get live recommendation
    recommendation = assistant.get_live_recommendation()
    
    print(f"\n\n🎯 PICK: {recommendation}")
    
    return assistant


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['pre-draft', 'live'], 
                       help='Run pre-draft analysis or live assistant')
    parser.add_argument('--pick', type=int, default=5,
                       help='Your draft position')
    
    args = parser.parse_args()
    
    if args.mode == 'pre-draft':
        run_pre_draft_analysis(args.pick)
    else:
        run_live_draft_assistant(args.pick)