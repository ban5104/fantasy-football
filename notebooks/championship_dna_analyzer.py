"""
Championship DNA hybrid draft system - minimal implementation
Identifies winning roster compositions and provides real-time draft guidance
"""

import pandas as pd
import numpy as np
from pathlib import Path

class ChampionshipDNA:
    """Extract championship patterns from simulation data"""
    
    def __init__(self, cache_dir=None):
        # FIX #6: Improve path resolution
        if cache_dir is None:
            project_root = Path(__file__).parent.parent
            self.cache_dir = project_root / 'data' / 'cache'
        else:
            self.cache_dir = Path(cache_dir)
            if not self.cache_dir.is_absolute():
                project_root = Path(__file__).parent.parent
                # Use proper path resolution instead of fragile string manipulation
                self.cache_dir = project_root / Path(cache_dir).as_posix().lstrip('../')
        
    def load_champions(self, strategy='balanced', top_pct=0.1):
        """Load top performing rosters from simulation data"""
        # Find strategy files
        if not self.cache_dir.exists():
            print(f"❌ Cache directory not found: {self.cache_dir}")
            return None
            
        strategy_files = list(self.cache_dir.glob(f"{strategy}_pick*_n*_r*.parquet"))
        
        if not strategy_files:
            print(f"❌ No files found for strategy: {strategy} in {self.cache_dir}")
            return None
            
        # Use the largest simulation file
        best_file = max(strategy_files, key=lambda f: self._extract_n_sims(f.name))
        print(f"📂 Loading champions from: {best_file.name}")
        
        try:
            df = pd.read_parquet(best_file)
        except Exception as e:
            print(f"❌ Error loading Parquet file: {e}")
            return None
            
        # Validate required columns
        required_cols = ['sim', 'roster_value', 'pos', 'player_name']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            print(f"   Available columns: {df.columns.tolist()}")
            return None
        
        # Get top performers by roster value (FIX #1: use max() instead of first())
        roster_values = df.groupby('sim')['roster_value'].max().sort_values(ascending=False)
        # FIX #2: Guard against zero champions
        n_champions = max(1, int(len(roster_values) * top_pct))
        champion_sims = roster_values.head(n_champions).index
        
        champions = df[df['sim'].isin(champion_sims)].copy()
        print(f"🏆 Extracted {len(champion_sims)} champions (top {top_pct:.0%})")
        
        return champions
    
    def get_north_star(self, champions):
        """Extract the ideal roster composition from champions"""
        # Count position distribution in champion rosters
        pos_counts = champions.groupby(['sim', 'pos']).size().unstack(fill_value=0)
        
        # Get the most common composition with safe access
        north_star = {}
        for pos in pos_counts.columns:
            mode_values = pos_counts[pos].mode()
            if len(mode_values) > 0:
                north_star[pos] = int(mode_values.iloc[0])
            else:
                # Fallback to median if no mode exists
                north_star[pos] = int(pos_counts[pos].median())
            
        return north_star
    def create_tiers(self, df, position):
        """Create player tiers based on championship frequency"""
        pos_df = df[df['pos'] == position].copy()
        if len(pos_df) == 0:
            return {}
            
        # FIX #3: Handle different column names for points
        points_col = 'sampled_points'
        if points_col not in df.columns:
            if 'proj' in df.columns:
                points_col = 'proj'
            elif 'roster_value' in df.columns:
                points_col = 'roster_value'
            else:
                print(f"⚠️  Warning: No points column found for {position}, using sim count only")
                points_col = None
        
        # Calculate draft frequency
        agg_dict = {'sim': 'nunique'}
        if points_col:
            agg_dict[points_col] = 'mean'
        
        player_freq = pos_df.groupby('player_name').agg(agg_dict)
        
        total_champion_sims = df['sim'].nunique()
        player_freq['champion_rate'] = player_freq['sim'] / total_champion_sims
        player_freq = player_freq.sort_values('champion_rate', ascending=False)
        
        # Define tier cutoffs with minimum sizes
        n_players = len(player_freq)
        # FIX #9: Add warning for sparse data
        if n_players < 10:
            print(f"⚠️  Warning: Only {n_players} {position} players found - tier analysis may be unreliable")
        
        tier_cutoffs = {
            1: max(1, int(n_players * 0.05)),  # Top 5%, minimum 1
            2: max(2, int(n_players * 0.20)),  # Next 15% (5-20%), minimum 2
            3: max(3, int(n_players * 0.40)),  # Next 20% (20-40%), minimum 3
            4: n_players                       # Rest (40%+)
        }
        
        # Assign tiers
        tiers = {}
        for i, (player, stats) in enumerate(player_freq.iterrows()):
            if i < tier_cutoffs[1]:
                tier = 1
            elif i < tier_cutoffs[2]:
                tier = 2
            elif i < tier_cutoffs[3]:
                tier = 3
            else:
                tier = 4
                
            tiers[player] = {
                'tier': tier,
                'champion_rate': stats['champion_rate'],
                'avg_points': stats[points_col] if points_col else 0
            }
            
        return tiers
    
    def calculate_windows(self, champions, round_num, n_teams=14):
        """Calculate position pick probabilities for a round"""
        champions_copy = champions.copy()
        
        # Check if we have actual draft data or need to use fallback
        if 'draft_round' in champions_copy.columns:
            # Use actual draft round data
            champions_copy['round'] = champions_copy['draft_round']
        elif 'draft_pick' in champions_copy.columns:
            # Calculate from draft pick
            champions_copy['round'] = ((champions_copy['draft_pick'] - 1) // n_teams) + 1
        else:
            # Fallback: estimate from roster position (will only show round 1)
            print("⚠️  Warning: No draft sequence data available, using roster position")
            champions_copy = champions_copy.sort_values('sim')
            champions_copy['pick_order'] = champions_copy.groupby('sim').cumcount()
            champions_copy['round'] = (champions_copy['pick_order'] // n_teams) + 1
        
        # Check available rounds
        max_round = champions_copy['round'].max()
        if round_num > max_round:
            print(f"⚠️  Note: Only {max_round} round(s) of data available (requested round {round_num})")
            return {}
        
        # Filter for requested round
        round_picks = champions_copy[champions_copy['round'] == round_num]
        
        if len(round_picks) == 0:
            return {}
            
        # Calculate position probabilities
        pos_counts = round_picks['pos'].value_counts()
        total_picks = len(round_picks)
        
        windows = {}
        for pos, count in pos_counts.items():
            windows[pos] = count / total_picks
            
        return windows
    
    def generate_pivots(self, current_roster, available_tiers, north_star):
        """Generate pivot alerts based on tier scarcity"""
        alerts = []
        
        # Count current roster by position
        current_counts = {}
        for pos in north_star.keys():
            # FIX #7: Handle both dict and object formats for flexibility
            if isinstance(current_roster, list):
                if len(current_roster) > 0 and isinstance(current_roster[0], dict):
                    current_counts[pos] = len([p for p in current_roster if p.get('pos') == pos])
                else:
                    # Handle objects with pos attribute
                    current_counts[pos] = len([p for p in current_roster if getattr(p, 'pos', None) == pos])
            else:
                current_counts[pos] = 0
        
        # Check for urgent needs
        for pos, target in north_star.items():
            current = current_counts.get(pos, 0)
            remaining_need = target - current
            
            if remaining_need > 0 and pos in available_tiers:
                tier_counts = {}
                for player, data in available_tiers[pos].items():
                    tier = data['tier']
                    tier_counts[tier] = tier_counts.get(tier, 0) + 1
                
                # Alert for tier scarcity
                tier1_left = tier_counts.get(1, 0)
                tier2_left = tier_counts.get(2, 0)
                
                if tier1_left <= 2:
                    alerts.append(f"• Only {tier1_left} Tier-1 {pos}s left → Prioritize {pos} now")
                elif tier2_left <= 3 and remaining_need >= 2:
                    alerts.append(f"• Only {tier2_left} Tier-2 {pos}s left → Consider {pos} soon")
                    
        return alerts
    
    def display_blueprint(self, north_star, champions):
        """Display championship blueprint card"""
        total_sims = champions['sim'].nunique()
        
        print("🎯 CHAMPIONSHIP BLUEPRINT")
        print("=" * 37)
        
        for pos, count in sorted(north_star.items()):
            # Calculate tier requirements
            pos_champions = champions[champions['pos'] == pos]
            if len(pos_champions) > 0:
                tiers = self.create_tiers(champions, pos)
                tier2_plus = len([p for p, data in tiers.items() if data['tier'] <= 2])
                min_tier2 = min(2, tier2_plus) if count >= 2 else 0
                print(f"{pos}: {count} players (≥{min_tier2} Tier-2+)")
            else:
                print(f"{pos}: {count} players")
        
        # FIX #8: Show actual support fraction instead of meaningless success rate
        print(f"Support: {total_sims} champion rosters analyzed")
        print()
    
    def display_windows(self, windows, round_num):
        """Display pick windows card"""
        print(f"📊 ROUND {round_num} PICK WINDOWS")
        print("=" * 37)
        
        for pos, prob in sorted(windows.items(), key=lambda x: x[1], reverse=True):
            # FIX #4: Fix tier percentage calculation bug
            tier1_pct = int(prob * 100 * 0.3)  # Convert to percentage first
            tier2_pct = int(prob * 100 * 0.7)
            print(f"{pos}: {prob:.0%} chance (Tier-1: {tier1_pct}%, Tier-2: {tier2_pct}%)")
        print()
    
    def display_pivots(self, alerts):
        """Display pivot alerts card"""
        print("⚠️  PIVOT ALERTS")
        print("=" * 37)
        
        if alerts:
            for alert in alerts:
                print(alert)
        else:
            print("• No urgent pivots needed")
        print()
    
    def _extract_n_sims(self, filename):
        """Extract simulation count from filename"""
        try:
            # Pattern: strategy_pick5_n200_r14.parquet
            parts = filename.split('_')
            for part in parts:
                if part.startswith('n') and part[1:].isdigit():
                    return int(part[1:])
        except:
            pass
        return 0

def run_championship_analysis(strategy='balanced', round_num=3, n_teams=14):
    """Main function to run complete championship DNA analysis
    
    Args:
        strategy: Draft strategy to analyze
        round_num: Round number to analyze windows for
        n_teams: Number of teams in the league (default 14 from config)
    """
    
    analyzer = ChampionshipDNA()
    
    # 1. Load champions
    champions = analyzer.load_champions(strategy=strategy, top_pct=0.1)
    if champions is None:
        return
    
    # 2. Extract North Star composition
    north_star = analyzer.get_north_star(champions)
    
    # 3. Calculate pick windows (with proper n_teams)
    windows = analyzer.calculate_windows(champions, round_num, n_teams=n_teams)
    
    # 4. Create tier systems for each position
    all_tiers = {}
    for pos in north_star.keys():
        all_tiers[pos] = analyzer.create_tiers(champions, pos)
    
    # 5. Generate pivot alerts (simplified - no current roster)
    current_roster = []  # Empty for demo
    alerts = analyzer.generate_pivots(current_roster, all_tiers, north_star)
    
    # 6. Display all three cards
    analyzer.display_blueprint(north_star, champions)
    analyzer.display_windows(windows, round_num)
    analyzer.display_pivots(alerts)
    
    return {
        'north_star': north_star,
        'windows': windows,
        'tiers': all_tiers,
        'alerts': alerts
    }

if __name__ == "__main__":
    # Demo run
    run_championship_analysis(strategy='balanced', round_num=3)