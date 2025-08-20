#!/usr/bin/env python3
"""
Championship DNA Hybrid Approach
Implements the North Star → Tier Windows → Pivot Rules system
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

class HybridDraftSystem:
    """
    Complete hybrid draft system with:
    1. North Star roster targets
    2. Tier-based pick windows
    3. Dynamic pivot rules
    4. Live adaptation logic
    """
    
    def __init__(self, data_dir='data/simulations'):
        self.data_dir = Path(data_dir)
        self.north_star = None
        self.tier_windows = {}
        self.pivot_rules = []
        self.tier_definitions = {}
        
    def define_tiers(self, players_df):
        """
        Define position-specific tiers based on projected points
        Tier 1 = top 5% (clear stars)
        Tier 2 = next 15% (high starters)
        Tier 3 = next 30% (starter-level)
        Tier 4+ = bench/depth
        """
        tiers = {}
        
        for pos in ['RB', 'WR', 'QB', 'TE']:
            pos_players = players_df[players_df['pos'] == pos].copy()
            pos_players = pos_players.sort_values('proj', ascending=False)
            
            n_players = len(pos_players)
            tier_1_cutoff = int(n_players * 0.05)
            tier_2_cutoff = int(n_players * 0.20)
            tier_3_cutoff = int(n_players * 0.50)
            
            tiers[pos] = {
                'tier_1': pos_players.iloc[:tier_1_cutoff]['proj'].min() if tier_1_cutoff > 0 else float('inf'),
                'tier_2': pos_players.iloc[tier_1_cutoff:tier_2_cutoff]['proj'].min() if tier_2_cutoff > tier_1_cutoff else float('inf'),
                'tier_3': pos_players.iloc[tier_2_cutoff:tier_3_cutoff]['proj'].min() if tier_3_cutoff > tier_2_cutoff else float('inf'),
                'tier_4': pos_players.iloc[tier_3_cutoff:]['proj'].min() if tier_3_cutoff < n_players else 0
            }
            
        self.tier_definitions = tiers
        return tiers
    
    def analyze_winning_patterns(self, simulation_file):
        """
        Step 1: Define North Star roster targets from winning teams
        """
        df = pd.read_parquet(simulation_file)
        
        # Get top 10% by total value
        threshold = df.groupby('sim')['roster_value'].first().quantile(0.9)
        winning_sims = df[df['roster_value'] >= threshold]['sim'].unique()
        winners = df[df['sim'].isin(winning_sims)]
        
        # Analyze roster composition with tiers
        roster_blueprint = defaultdict(lambda: defaultdict(list))
        
        for sim in winning_sims:
            sim_roster = winners[winners['sim'] == sim]
            
            # Count by position and tier
            for pos in ['RB', 'WR', 'QB', 'TE']:
                pos_players = sim_roster[sim_roster['pos'] == pos]
                
                # Total count
                roster_blueprint[pos]['total'].append(len(pos_players))
                
                # Tier counts
                if self.tier_definitions and pos in self.tier_definitions:
                    tier_counts = {'tier_1': 0, 'tier_2': 0, 'tier_3': 0, 'tier_4': 0}
                    
                    for _, player in pos_players.iterrows():
                        proj = player['sampled_points']
                        if proj >= self.tier_definitions[pos]['tier_1']:
                            tier_counts['tier_1'] += 1
                        elif proj >= self.tier_definitions[pos]['tier_2']:
                            tier_counts['tier_2'] += 1
                        elif proj >= self.tier_definitions[pos]['tier_3']:
                            tier_counts['tier_3'] += 1
                        else:
                            tier_counts['tier_4'] += 1
                    
                    for tier, count in tier_counts.items():
                        roster_blueprint[pos][tier].append(count)
        
        # Create North Star targets
        self.north_star = {}
        for pos, data in roster_blueprint.items():
            self.north_star[pos] = {
                'total': int(np.median(data['total'])),
                'tier_1_plus': int(np.median([t1 for t1 in data.get('tier_1', [0])])),
                'tier_2_plus': int(np.median([t1 + t2 for t1, t2 in 
                                             zip(data.get('tier_1', [0]), data.get('tier_2', [0]))])),
                'success_rate': len(winning_sims) / df['sim'].nunique()
            }
        
        return self.north_star
    
    def derive_tier_windows(self, simulation_file):
        """
        Step 2: Derive soft pick windows where success rates spike
        """
        df = pd.read_parquet(simulation_file)
        
        # Get winning teams
        threshold = df.groupby('sim')['roster_value'].first().quantile(0.9)
        winning_sims = df[df['roster_value'] >= threshold]['sim'].unique()
        winners = df[df['sim'].isin(winning_sims)]
        
        # Analyze when positions are typically drafted
        pick_windows = defaultdict(lambda: defaultdict(list))
        
        for sim in winning_sims:
            sim_roster = winners[winners['sim'] == sim].sort_values('draft_pick')
            
            for _, player in sim_roster.iterrows():
                round_num = player['draft_round']  # Use the actual draft round
                pos = player['pos']
                
                # Determine tier
                proj = player['sampled_points']
                if self.tier_definitions and pos in self.tier_definitions:
                    if proj >= self.tier_definitions[pos]['tier_1']:
                        tier = 'tier_1'
                    elif proj >= self.tier_definitions[pos]['tier_2']:
                        tier = 'tier_2'
                    elif proj >= self.tier_definitions[pos]['tier_3']:
                        tier = 'tier_3'
                    else:
                        tier = 'tier_4'
                    
                    pick_windows[round_num][(pos, tier)].append(1)
        
        # Calculate probabilities
        self.tier_windows = {}
        for round_num in range(1, 15):  # All 14 rounds
            self.tier_windows[round_num] = {}
            
            for (pos, tier), occurrences in pick_windows[round_num].items():
                probability = len(occurrences) / len(winning_sims)
                if probability > 0.1:  # Only keep significant patterns
                    if pos not in self.tier_windows[round_num]:
                        self.tier_windows[round_num][pos] = {}
                    self.tier_windows[round_num][pos][tier] = probability
        
        return self.tier_windows
    
    def define_pivot_rules(self):
        """
        Step 3: Define simple pivot triggers
        """
        self.pivot_rules = [
            {
                'name': 'RB_scarcity',
                'condition': lambda state: state['tier_1_rbs_remaining'] <= 1 and state['my_rbs'] < 2,
                'action': 'prioritize_rb',
                'message': 'Only 1 Tier-1 RB left - prioritize RB'
            },
            {
                'name': 'TE_cliff',
                'condition': lambda state: state['tier_1_tes_remaining'] <= 2 and state['my_tes'] == 0 and state['round'] <= 4,
                'action': 'prioritize_te',
                'message': 'Elite TE cliff approaching - consider TE'
            },
            {
                'name': 'WR_run',
                'condition': lambda state: state['recent_wr_picks'] >= 4 and state['my_wrs'] < 2,
                'action': 'prioritize_wr',
                'message': 'WR run detected - grab WR value'
            },
            {
                'name': 'roster_imbalance',
                'condition': lambda state: state['roster_completion'] < 0.3 and state['round'] >= 6,
                'action': 'best_available',
                'message': 'Roster off-track - pivot to best available'
            }
        ]
        
        return self.pivot_rules
    
    def get_live_recommendation(self, current_round, my_roster, available_players, recent_picks=[]):
        """
        Step 4: Live draft execution with adaptation
        """
        # Calculate current state
        state = self._calculate_draft_state(current_round, my_roster, available_players, recent_picks)
        
        # Check if on track with North Star
        on_track = self._check_north_star_progress(state)
        
        # Primary recommendation from tier windows
        primary_rec = self._get_tier_window_recommendation(current_round, state)
        
        # Check pivot rules
        active_pivots = []
        for rule in self.pivot_rules:
            if rule['condition'](state):
                active_pivots.append(rule)
        
        # Generate final recommendation
        if active_pivots:
            # Pivot needed
            pivot = active_pivots[0]  # Take highest priority pivot
            recommendation = {
                'action': pivot['action'],
                'reason': pivot['message'],
                'confidence': 0.7,  # Lower confidence during pivot
                'on_track': False
            }
        elif on_track and primary_rec:
            # Follow tier windows
            recommendation = {
                'action': primary_rec['position'],
                'tier_target': primary_rec['tier'],
                'probability': primary_rec['probability'],
                'reason': f"Tier window suggests {primary_rec['position']} (Tier {primary_rec['tier'][-1]})",
                'confidence': primary_rec['probability'],
                'on_track': True
            }
        else:
            # Off track but no specific pivot
            recommendation = {
                'action': 'best_available',
                'reason': 'Maximize value to get back on track',
                'confidence': 0.5,
                'on_track': False
            }
        
        # Add regret calculation
        recommendation['regret_if_skip'] = self._calculate_regret(state, recommendation['action'])
        
        return recommendation
    
    def _calculate_draft_state(self, round_num, my_roster, available_players, recent_picks):
        """Calculate current draft state metrics"""
        state = {
            'round': round_num,
            'my_rbs': sum(1 for p in my_roster if p.get('pos') == 'RB'),
            'my_wrs': sum(1 for p in my_roster if p.get('pos') == 'WR'),
            'my_qbs': sum(1 for p in my_roster if p.get('pos') == 'QB'),
            'my_tes': sum(1 for p in my_roster if p.get('pos') == 'TE'),
            'recent_wr_picks': sum(1 for p in recent_picks[-8:] if p == 'WR'),
            'roster_completion': len(my_roster) / 14  # Assuming 14 rounds
        }
        
        # Count remaining tier players
        for pos in ['RB', 'WR', 'QB', 'TE']:
            tier_1_remaining = 0
            tier_2_remaining = 0
            
            if self.tier_definitions and pos in self.tier_definitions:
                for player in available_players:
                    if player.get('pos') == pos:
                        proj = player.get('proj', 0)
                        if proj >= self.tier_definitions[pos]['tier_1']:
                            tier_1_remaining += 1
                        elif proj >= self.tier_definitions[pos]['tier_2']:
                            tier_2_remaining += 1
            
            state[f'tier_1_{pos.lower()}s_remaining'] = tier_1_remaining
            state[f'tier_2_{pos.lower()}s_remaining'] = tier_2_remaining
        
        return state
    
    def _check_north_star_progress(self, state):
        """Check if roster is on track with North Star targets"""
        if not self.north_star:
            return True  # No targets defined
        
        on_track = True
        expected_progress = state['roster_completion']
        
        for pos in ['RB', 'WR', 'QB', 'TE']:
            current = state[f'my_{pos.lower()}s']
            target = self.north_star.get(pos, {}).get('total', 0)
            expected = target * expected_progress
            
            if current < expected - 1:  # Allow 1 player buffer
                on_track = False
                break
        
        return on_track
    
    def _get_tier_window_recommendation(self, round_num, state):
        """Get recommendation from tier windows"""
        if round_num not in self.tier_windows:
            return None
        
        round_windows = self.tier_windows[round_num]
        best_option = None
        best_prob = 0
        
        for pos, tiers in round_windows.items():
            # Check if we need more of this position
            current = state[f'my_{pos.lower()}s']
            target = self.north_star.get(pos, {}).get('total', 0)
            
            if current < target:
                for tier, prob in tiers.items():
                    if prob > best_prob:
                        best_prob = prob
                        best_option = {
                            'position': pos,
                            'tier': tier,
                            'probability': prob
                        }
        
        return best_option
    
    def _calculate_regret(self, state, recommended_action):
        """Calculate expected regret if recommendation is not followed"""
        # Simplified regret calculation
        if recommended_action in ['RB', 'WR', 'TE', 'QB']:
            # Position-specific recommendation
            remaining = state[f'tier_1_{recommended_action.lower()}s_remaining']
            if remaining <= 2:
                return 15.0  # High regret if scarce
            elif remaining <= 5:
                return 8.0   # Medium regret
            else:
                return 3.0   # Low regret if plenty available
        else:
            return 5.0  # Default regret
    
    def generate_draft_blueprint(self):
        """
        Generate complete pre-draft blueprint with all components
        """
        blueprint = {
            'north_star': self.north_star,
            'tier_windows': self.tier_windows,
            'pivot_rules': [{'name': r['name'], 'message': r['message']} for r in self.pivot_rules],
            'tier_definitions': self.tier_definitions
        }
        
        return blueprint
    
    def print_draft_cards(self):
        """Print draft cards for notebook display"""
        
        # Blueprint Card
        print("="*60)
        print("📋 ROSTER BLUEPRINT (North Star)")
        print("="*60)
        if self.north_star:
            for pos, targets in self.north_star.items():
                print(f"{pos}: {targets['total']} total (≥{targets['tier_2_plus']} Tier-2+)")
            print(f"\nSuccess Rate: {self.north_star.get('RB', {}).get('success_rate', 0)*100:.1f}%")
        
        # Pick Windows Card
        print("\n" + "="*60)
        print("🎯 PICK WINDOWS (All 14 Rounds)")
        print("="*60)
        for round_num in range(1, 15):
            if round_num in self.tier_windows:
                print(f"\nRound {round_num}:")
                for pos, tiers in self.tier_windows[round_num].items():
                    for tier, prob in sorted(tiers.items(), key=lambda x: x[1], reverse=True)[:1]:
                        print(f"  {pos} ({tier}): {prob*100:.0f}% of winners")
        
        # Pivot Rules Card
        print("\n" + "="*60)
        print("⚡ PIVOT TRIGGERS")
        print("="*60)
        for rule in self.pivot_rules[:3]:  # Show top 3 rules
            print(f"• {rule['message']}")


def run_hybrid_analysis(pick=5, n_sims=200):
    """Complete hybrid analysis workflow"""
    print("🎯 Running Hybrid Championship DNA Analysis...")
    print("-"*60)
    
    # Initialize system
    system = HybridDraftSystem()
    
    # Find simulation files - look in the correct directory
    cache_dir = Path('data/cache')
    sim_files = list(cache_dir.glob(f'balanced_pick{pick}_n{n_sims}*.parquet'))
    if not sim_files:
        print("❌ No simulation data found!")
        print(f"Run: python monte_carlo_runner.py export --strategy balanced --pick {pick} --n-sims {n_sims}")
        return None
    
    # Load players for tier definitions
    from src.monte_carlo import DraftSimulator
    sim = DraftSimulator()
    sim.prob_model.load_data()
    
    # Define tiers
    system.define_tiers(sim.prob_model.players_df)
    
    # Analyze patterns
    system.analyze_winning_patterns(sim_files[0])
    
    # Derive windows
    system.derive_tier_windows(sim_files[0])
    
    # Define pivot rules
    system.define_pivot_rules()
    
    # Print cards
    system.print_draft_cards()
    
    return system


if __name__ == "__main__":
    system = run_hybrid_analysis(pick=5, n_sims=200)