"""
Live Draft Integration Example

Shows how to automatically update Dynamic VBD rankings as players are drafted
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import yaml
import time
from typing import List, Dict, Set
from dynamic_vbd import DynamicVBDTransformer, create_probability_forecast, create_draft_state


class LiveDraftTracker:
    """
    Tracks live draft state and automatically updates Dynamic VBD rankings
    """
    
    def __init__(self, config_path: str, projections_df: pd.DataFrame):
        """Initialize the live draft tracker"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.all_projections = projections_df.copy()
        self.transformer = DynamicVBDTransformer(self.config, projections_df)
        
        # Track draft state
        self.current_pick = 1
        self.drafted_players: Set[str] = set()
        self.pick_history: List[Dict] = []
        
        print(f"Live Draft Tracker initialized")
        print(f"Total players: {len(self.all_projections)}")
        print(f"Dynamic VBD enabled: {self.transformer.enabled}")
    
    def draft_player(self, player_name: str, pick_number: int = None) -> pd.DataFrame:
        """
        Record a player being drafted and return updated rankings
        
        Args:
            player_name: Name of drafted player
            pick_number: Optional pick number (auto-increments if not provided)
            
        Returns:
            Updated Dynamic VBD rankings for remaining players
        """
        if pick_number:
            self.current_pick = pick_number
        
        # Record the pick
        self.drafted_players.add(player_name)
        self.pick_history.append({
            'pick': self.current_pick,
            'player': player_name,
            'timestamp': time.time()
        })
        
        print(f"\n📝 PICK {self.current_pick}: {player_name} drafted")
        
        # Get updated draft state
        draft_state = create_draft_state(
            current_pick=self.current_pick,
            drafted_players=list(self.drafted_players)
        )
        
        # Filter to available players only
        available_players = self.get_available_players()
        
        if available_players.empty:
            print("⚠️ No players remaining!")
            return pd.DataFrame()
        
        # Recalculate position probabilities based on remaining players
        forecast = self.calculate_updated_probabilities(available_players)
        
        # Get updated Dynamic VBD rankings
        updated_rankings = self.transformer.transform(
            available_players, forecast, draft_state
        )
        
        # Increment pick counter
        self.current_pick += 1
        
        # Display update summary
        self.display_update_summary(updated_rankings)
        
        return updated_rankings
    
    def get_available_players(self) -> pd.DataFrame:
        """Get DataFrame of players still available for draft"""
        return self.all_projections[
            ~self.all_projections['Player'].isin(self.drafted_players)
        ].copy().reset_index(drop=True)
    
    def calculate_updated_probabilities(self, available_df: pd.DataFrame) -> 'ProbabilityForecast':
        """
        Calculate updated position probabilities based on remaining players
        
        In a real system, this would call your probability forecasting model
        """
        # Simple mock calculation - replace with your actual probability system
        position_counts = available_df['POSITION'].value_counts()
        total_available = len(available_df)
        
        if total_available == 0:
            return create_probability_forecast(1, {})
        
        # Calculate probabilities based on scarcity and draft patterns
        position_probs = {}
        
        # Base weights for typical draft behavior
        base_weights = {
            'QB': 0.8,   # Lower weight - usually drafted less frequently
            'RB': 1.3,   # Higher weight - premium position
            'WR': 1.2,   # High weight - lots of depth needed
            'TE': 0.9,   # Medium weight
            'DEF': 0.4,  # Lower weight - drafted late
            'K': 0.3     # Lowest weight - drafted very late
        }
        
        # Adjust weights based on draft stage
        stage_multiplier = self.get_draft_stage_multiplier()
        
        for pos, count in position_counts.items():
            if count > 0:
                # Base probability from availability
                base_prob = count / total_available
                
                # Apply position weight and stage multiplier
                weight = base_weights.get(pos, 1.0) * stage_multiplier.get(pos, 1.0)
                position_probs[pos] = base_prob * weight
        
        # Normalize to sum to 1
        total_prob = sum(position_probs.values())
        if total_prob > 0:
            position_probs = {k: v/total_prob for k, v in position_probs.items()}
        
        # Calculate horizon picks (next 3-7 picks depending on draft stage)
        total_picks = self.config.get('basic_settings', {}).get('teams', 14) * 16  # 14 teams x 16 rounds
        remaining_picks = total_picks - self.current_pick + 1
        horizon_picks = min(max(3, remaining_picks // 20), 7)  # 3-7 picks ahead
        
        return create_probability_forecast(
            horizon_picks=horizon_picks,
            position_probs=position_probs
        )
    
    def get_draft_stage_multiplier(self) -> Dict[str, float]:
        """
        Adjust position weights based on draft stage
        
        Early draft: Premium positions more likely
        Late draft: Kicker/Defense more likely
        """
        total_picks = self.config.get('basic_settings', {}).get('teams', 14) * 16
        draft_progress = self.current_pick / total_picks
        
        if draft_progress < 0.3:  # Early draft (first ~4 rounds)
            return {
                'QB': 1.2, 'RB': 1.3, 'WR': 1.2, 'TE': 1.1,
                'DEF': 0.2, 'K': 0.1
            }
        elif draft_progress < 0.7:  # Mid draft (rounds 5-11)
            return {
                'QB': 1.0, 'RB': 1.1, 'WR': 1.1, 'TE': 1.0,
                'DEF': 0.6, 'K': 0.3
            }
        else:  # Late draft (rounds 12+)
            return {
                'QB': 0.8, 'RB': 0.9, 'WR': 0.9, 'TE': 0.8,
                'DEF': 1.5, 'K': 1.8
            }
    
    def display_update_summary(self, rankings: pd.DataFrame):
        """Display summary of ranking updates"""
        if 'VBD_BLENDED' not in rankings.columns:
            print("⚠️ VBD calculations not available")
            return
        
        available_count = len(rankings)
        print(f"🔄 Rankings updated - {available_count} players remaining")
        
        # Show top 5 available players
        top_5 = rankings.nlargest(5, 'VBD_BLENDED')
        print("\n🏆 Top 5 Available Players:")
        for i, (_, player) in enumerate(top_5.iterrows(), 1):
            print(f"  {i}. {player['Player']} ({player['POSITION']}) - VBD: {player['VBD_BLENDED']:.1f}")
        
        # Show position scarcity
        pos_counts = rankings['POSITION'].value_counts()
        print(f"\n📊 Remaining by position: {dict(pos_counts)}")
        
        # Show cache stats
        cache_stats = self.transformer.get_cache_stats()
        print(f"💾 Cache size: {cache_stats['cache_size']}")
    
    def get_top_available(self, n: int = 10) -> pd.DataFrame:
        """Get top N available players by Dynamic VBD"""
        available = self.get_available_players()
        if available.empty:
            return pd.DataFrame()
        
        draft_state = create_draft_state(self.current_pick, list(self.drafted_players))
        forecast = self.calculate_updated_probabilities(available)
        rankings = self.transformer.transform(available, forecast, draft_state)
        
        if 'VBD_BLENDED' in rankings.columns:
            return rankings.nlargest(n, 'VBD_BLENDED')
        return rankings.head(n)
    
    def simulate_draft_run(self, position: str, num_picks: int = 3):
        """
        Simulate a position run and show how rankings adjust
        
        Args:
            position: Position being drafted (e.g., 'RB')
            num_picks: Number of consecutive picks at that position
        """
        print(f"\n🔥 Simulating {position} run ({num_picks} picks)...")
        
        available = self.get_available_players()
        position_players = available[available['POSITION'] == position]['Player'].tolist()
        
        for i in range(min(num_picks, len(position_players))):
            player = position_players[i]
            self.draft_player(player)
            
            # Show how this affects remaining players of same position
            remaining_pos = self.get_available_players()
            remaining_pos = remaining_pos[remaining_pos['POSITION'] == position]
            if not remaining_pos.empty:
                print(f"   Remaining {position}s: {remaining_pos['Player'].tolist()[:3]}...")


def demo_live_draft_updates():
    """Demonstrate how Dynamic VBD updates during a live draft"""
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'league-config.yaml')
    
    # Create sample projections (in practice, load from your data pipeline)
    projections = pd.DataFrame({
        'Player': [
            # QBs
            'Josh Allen', 'Lamar Jackson', 'Mahomes', 'Burrow', 'Herbert', 'Hurts',
            # RBs
            'CMC', 'Bijan Robinson', 'Saquon', 'Henry', 'Jacobs', 'Cook', 'Mixon', 'Jones',
            # WRs  
            'Jefferson', 'Hill', 'Diggs', 'Adams', 'Brown', 'DK', 'Evans', 'Hopkins',
            # TEs
            'Kelce', 'Andrews', 'Kittle', 'Hockenson', 'Goedert', 'Waller',
            # DEF
            'Ravens DST', 'Bills DST', 'Steelers DST', 'Cowboys DST',
            # K
            'Tucker', 'Bass', 'McManus', 'Boswell'
        ],
        'POSITION': (
            ['QB'] * 6 + ['RB'] * 8 + ['WR'] * 8 + ['TE'] * 6 + ['DEF'] * 4 + ['K'] * 4
        ),
        'FANTASY_PTS': (
            # QBs: 340-290
            [340, 335, 330, 320, 315, 310] +
            # RBs: 290-220  
            [290, 285, 280, 275, 270, 265, 260, 250] +
            # WRs: 270-200
            [270, 265, 260, 255, 250, 245, 240, 230] +
            # TEs: 190-140
            [190, 185, 180, 175, 170, 165] +
            # DEF: 130-115
            [130, 125, 120, 115] +
            # K: 135-120
            [135, 130, 125, 120]
        )
    })
    
    print("=== Live Draft Dynamic VBD Demo ===\n")
    
    # Initialize tracker
    tracker = LiveDraftTracker(config_path, projections)
    
    print("\n🎯 Initial Rankings (Top 10):")
    initial_top_10 = tracker.get_top_available(10)
    if 'VBD_BLENDED' in initial_top_10.columns:
        for i, (_, player) in enumerate(initial_top_10.iterrows(), 1):
            print(f"  {i}. {player['Player']} ({player['POSITION']}) - VBD: {player['VBD_BLENDED']:.1f}")
    
    # Simulate draft picks and show updates
    draft_sequence = [
        'CMC',           # Pick 1: Top RB goes first
        'Jefferson',     # Pick 2: Top WR
        'Josh Allen',    # Pick 3: Top QB 
        'Bijan Robinson', # Pick 4: RB run starts
        'Saquon',        # Pick 5: RB run continues
        'Hill',          # Pick 6: Back to WR
        'Henry'          # Pick 7: More RB scarcity
    ]
    
    print(f"\n📋 Simulating draft picks 1-{len(draft_sequence)}:")
    print("=" * 50)
    
    for i, player in enumerate(draft_sequence, 1):
        updated_rankings = tracker.draft_player(player, i)
        
        # Show how this pick affected the top available players
        if not updated_rankings.empty and 'VBD_BLENDED' in updated_rankings.columns:
            time.sleep(0.5)  # Pause for readability
    
    # Show the power of position run detection
    print("\n" + "=" * 60)
    print("🔥 Demonstrating RB Run Detection:")
    tracker.simulate_draft_run('RB', 3)
    
    print("\n✅ Demo complete!")
    print("\nKey insights from Dynamic VBD:")
    print("1. Rankings automatically adjust as players are drafted")
    print("2. Position runs are detected and remaining players get VBD boosts")
    print("3. Cache system ensures fast updates during live drafts")
    print("4. Probability forecasts adapt to changing draft dynamics")


if __name__ == "__main__":
    demo_live_draft_updates()