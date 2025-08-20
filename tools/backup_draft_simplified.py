#!/usr/bin/env python3
"""
Simplified Emergency Backup Draft Tracker with Dynamic VBD
----------------------------------------------------------
Terminal-based draft tracker with streamlined Dynamic VBD integration.

Usage: 
  python backup_draft_simplified.py                 # Use config setting
  python backup_draft_simplified.py --dynamic-vbd   # Force enable Dynamic VBD
  python backup_draft_simplified.py --no-dynamic-vbd # Force disable Dynamic VBD
"""

import pandas as pd
import os
import sys
import argparse
import yaml
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

# Configuration
NUM_TEAMS = 14
NUM_ROUNDS = 16
TOTAL_PICKS = NUM_TEAMS * NUM_ROUNDS  # 224 picks

class SimplifiedDraftTracker:
    """Simplified draft tracker with streamlined Dynamic VBD."""
    
    def __init__(self, force_dynamic_vbd: Optional[bool] = None):
        self.picks = []
        self.current_pick = 1
        self.players_df = None
        self.output_dir = "data/draft"
        self.player_col = None
        self.position_col = None
        self.team_col = None
        
        # Dynamic VBD components - simplified
        self.config = None
        self.dynamic_vbd_enabled = False
        self.force_dynamic_vbd = force_dynamic_vbd
        self.vbd_cache = {}  # Simple in-memory cache
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load configuration
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """Load league configuration with simplified precedence."""
        config_path = "config/league-config.yaml"
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                
                # Simple precedence: command line > config > default
                if self.force_dynamic_vbd is not None:
                    self.dynamic_vbd_enabled = self.force_dynamic_vbd
                    print(f"🔧 Dynamic VBD: {self.dynamic_vbd_enabled} (command line)")
                else:
                    self.dynamic_vbd_enabled = self.config.get('dynamic_vbd', {}).get('enabled', False)
                    print(f"🔧 Dynamic VBD: {self.dynamic_vbd_enabled} (config file)")
            else:
                self.dynamic_vbd_enabled = self.force_dynamic_vbd if self.force_dynamic_vbd is not None else False
                print(f"⚠️  No config found, Dynamic VBD: {self.dynamic_vbd_enabled}")
        except Exception as e:
            print(f"⚠️  Config error: {e}, Dynamic VBD disabled")
            self.dynamic_vbd_enabled = False
    
    def _find_columns(self, df: pd.DataFrame) -> bool:
        """Find player, position, and team columns."""
        # Find player column
        for col in df.columns:
            if any(x in col.lower() for x in ['player', 'name']):
                self.player_col = col
                break
        else:
            self.player_col = df.columns[0]
        
        # Find position column
        for col in df.columns:
            if any(x in col.lower() for x in ['position', 'pos']):
                self.position_col = col
                break
        
        # Find team column
        for col in df.columns:
            if any(x in col.lower() for x in ['team', 'nfl', 'pro']):
                self.team_col = col
                break
        
        return self.player_col is not None
    
    def load_player_database(self) -> bool:
        """Load player data from available CSV files."""
        # Try VBD ranking files first
        vbd_sources = []
        if os.path.exists("data/output"):
            import glob
            vbd_patterns = [
                "data/output/vbd_rankings_top300_*.csv",
                "data/output/rankings_vbd_*_top300_*.csv"
            ]
            for pattern in vbd_patterns:
                vbd_sources.extend(sorted(glob.glob(pattern), reverse=True))
        
        # Fallback sources
        fallback_sources = [
            "data/CSG Fantasy Football Sheet - 2025 v13.01.csv",
            "CSG Fantasy Football Sheet - 2025 v13.01.csv",
            "data/output/draft_cheat_sheet.csv",
            "draft_cheat_sheet.csv"
        ]
        
        # Try all sources
        for source in vbd_sources + fallback_sources:
            if not os.path.exists(source):
                continue
                
            try:
                print(f"📂 Loading {source}...")
                try:
                    self.players_df = pd.read_csv(source, encoding='utf-8')
                except UnicodeDecodeError:
                    self.players_df = pd.read_csv(source, encoding='latin-1')
                
                if self.players_df.empty or not self._find_columns(self.players_df):
                    continue
                    
                # Clean data
                self.players_df = self.players_df[self.players_df[self.player_col].notna()].copy()
                self.players_df = self.players_df[self.players_df[self.player_col].str.strip() != ''].copy()
                
                print(f"✅ Loaded {len(self.players_df)} players")
                print(f"   Columns: {self.player_col}, {self.position_col}, {self.team_col}")
                
                # Check for VBD data
                has_vbd = any('VBD' in col.upper() for col in self.players_df.columns)
                has_fantasy = 'FANTASY_PTS' in self.players_df.columns
                
                if (has_vbd or has_fantasy) and self.dynamic_vbd_enabled:
                    print("   🚀 Dynamic VBD ready")
                elif self.dynamic_vbd_enabled:
                    print("   📊 Dynamic VBD disabled (no VBD data)")
                    self.dynamic_vbd_enabled = False
                
                return True
                    
            except Exception as e:
                print(f"⚠️  Failed to load {source}: {e}")
                continue
        
        print("❌ No player database could be loaded!")
        return False
    
    def load_existing_picks(self) -> int:
        """Load existing picks if resuming."""
        latest_file = os.path.join(self.output_dir, "draft_picks_latest.csv")
        
        if not os.path.exists(latest_file):
            return 1
        
        try:
            try:
                existing_df = pd.read_csv(latest_file, encoding='utf-8')
            except UnicodeDecodeError:
                existing_df = pd.read_csv(latest_file, encoding='latin-1')
            
            if existing_df.empty:
                return 1
            
            self.picks = []
            for _, row in existing_df.iterrows():
                pick = {
                    'overall_pick': int(row['overall_pick']),
                    'player_name': str(row['player_name']).strip(),
                    'position': str(row.get('position', 'Unknown')).strip(),
                    'team_name': str(row.get('team_name', '')).strip(),
                    'pro_team': str(row.get('pro_team', '')).strip()
                }
                self.picks.append(pick)
            
            self.picks.sort(key=lambda x: x['overall_pick'])
            self.current_pick = len(self.picks) + 1
            
            print(f"✅ Resumed with {len(self.picks)} picks at pick #{self.current_pick}")
            return self.current_pick
                
        except Exception as e:
            print(f"⚠️  Could not load existing picks: {e}")
            return 1
    
    def get_snake_draft_team(self, pick_number: int) -> Tuple[int, int]:
        """Calculate team and round for snake draft."""
        round_num = ((pick_number - 1) // NUM_TEAMS) + 1
        
        if round_num % 2 == 1:  # Odd rounds: 1->14
            team_num = ((pick_number - 1) % NUM_TEAMS) + 1
        else:  # Even rounds: 14->1
            team_num = NUM_TEAMS - ((pick_number - 1) % NUM_TEAMS)
        
        return team_num, round_num
    
    def update_dynamic_vbd(self) -> None:
        """Simplified Dynamic VBD update."""
        if not self.dynamic_vbd_enabled or self.config is None:
            return
        
        try:
            # Get available players
            drafted_names = [p['player_name'] for p in self.picks]
            available = self.players_df[~self.players_df[self.player_col].isin(drafted_names)].copy()
            
            if available.empty:
                return
            
            # Simple cache key
            cache_key = f"{self.current_pick}_{len(drafted_names)}"
            
            if cache_key in self.vbd_cache:
                adjustments = self.vbd_cache[cache_key]
            else:
                adjustments = self._calculate_simple_adjustments(available)
                self.vbd_cache[cache_key] = adjustments
            
            # Apply adjustments to available players
            self._apply_vbd_adjustments(available, adjustments)
            
        except Exception as e:
            print(f"⚠️  Dynamic VBD error: {e}")
    
    def _calculate_simple_adjustments(self, available_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate simple position scarcity adjustments."""
        if self.position_col not in available_df.columns:
            return {}
        
        position_counts = available_df[self.position_col].value_counts()
        total_available = len(available_df)
        
        if total_available <= 0:
            return {}
        
        # Simple scarcity calculation
        adjustments = {}
        draft_progress = self.current_pick / TOTAL_PICKS
        
        for pos, count in position_counts.items():
            scarcity = 1.0 - (count / total_available)  # Higher when fewer available
            
            # Simple draft stage adjustment
            if draft_progress < 0.3:  # Early draft
                stage_mult = 1.2 if pos in ['RB', 'WR'] else 0.8
            elif draft_progress > 0.7:  # Late draft
                stage_mult = 1.5 if pos in ['DEF', 'K'] else 0.9
            else:  # Mid draft
                stage_mult = 1.0
            
            adjustments[pos] = scarcity * stage_mult
        
        return adjustments
    
    def _apply_vbd_adjustments(self, available_df: pd.DataFrame, adjustments: Dict[str, float]) -> None:
        """Apply simple VBD adjustments to main DataFrame."""
        # Find VBD column to adjust
        vbd_col = None
        for col in ['VBD_BEER', 'VBD_BLENDED', 'FANTASY_PTS']:
            if col in available_df.columns:
                vbd_col = col
                break
        
        if vbd_col is None:
            return
        
        # Apply adjustments
        for _, row in available_df.iterrows():
            player_name = row[self.player_col]
            position = row[self.position_col]
            
            if position in adjustments:
                mask = self.players_df[self.player_col] == player_name
                if mask.any():
                    current_value = self.players_df.loc[mask, vbd_col].iloc[0]
                    adjustment = adjustments[position] * 2.0  # Simple scaling
                    self.players_df.loc[mask, vbd_col] = current_value + adjustment
    
    def show_rankings(self, top_n: int = 10) -> None:
        """Show top available players."""
        if self.players_df is None:
            print("❌ No player data available")
            return
        
        # Get available players
        drafted_names = [p['player_name'] for p in self.picks]
        available = self.players_df[~self.players_df[self.player_col].isin(drafted_names)].copy()
        
        if available.empty:
            print("⚠️  No players remaining!")
            return
        
        # Find ranking column
        rank_col = None
        for col in ['VBD_BEER', 'VBD_BLENDED', 'FANTASY_PTS']:
            if col in available.columns:
                rank_col = col
                break
        
        if rank_col:
            top_players = available.nlargest(top_n, rank_col)
            print(f"\n🏆 Top {top_n} Available Players (by {rank_col}):")
        else:
            top_players = available.head(top_n)
            print(f"\n📋 Top {top_n} Available Players:")
        
        print("=" * 60)
        
        for i, (_, player) in enumerate(top_players.iterrows(), 1):
            name = player[self.player_col]
            pos = player[self.position_col] if self.position_col else 'N/A'
            team = player[self.team_col] if self.team_col else 'N/A'
            
            if rank_col:
                value = player[rank_col]
                print(f"  {i:2d}. {name:<25} ({pos:3s}, {team:4s}) - {value:6.1f}")
            else:
                print(f"  {i:2d}. {name:<25} ({pos:3s}, {team:4s})")
    
    def show_position_runs(self) -> None:
        """Show simple position run analysis."""
        if not self.picks or not self.dynamic_vbd_enabled:
            return
        
        print(f"\n📈 Position Analysis:")
        
        # Recent picks
        recent_picks = self.picks[-5:] if len(self.picks) >= 5 else self.picks
        pos_counts = {}
        for pick in recent_picks:
            pos = pick.get('position', 'Unknown')
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        
        print(f"   Recent picks: {dict(pos_counts)}")
        
        # Position run detection
        if len(recent_picks) >= 3:
            last_3_positions = [p.get('position') for p in recent_picks[-3:]]
            if len(set(last_3_positions)) == 1:
                run_pos = last_3_positions[0]
                print(f"   🔥 {run_pos} RUN detected!")
    
    def find_player(self, query: str) -> Optional[Dict]:
        """Find player with simple string matching."""
        if not query or len(query) < 2 or self.players_df is None:
            return None
        
        query_lower = query.lower().strip()
        
        # Get available players
        drafted_names = [p['player_name'] for p in self.picks]
        available = self.players_df[~self.players_df[self.player_col].isin(drafted_names)].copy()
        
        if available.empty:
            return None
        
        # Simple string matching
        matches = available[
            available[self.player_col].str.lower().str.contains(query_lower, na=False, regex=False)
        ]
        
        if matches.empty:
            return None
        
        if len(matches) == 1:
            row = matches.iloc[0]
            return {
                'Player': row[self.player_col],
                'Position': row[self.position_col] if self.position_col else 'Unknown',
                'Team': row[self.team_col] if self.team_col else 'N/A'
            }
        
        # Multiple matches - show selection
        print(f"\n🔍 Multiple players found:")
        display_matches = matches.head(5)  # Limit to 5
        
        for idx, (_, row) in enumerate(display_matches.iterrows(), 1):
            name = row[self.player_col]
            pos = row[self.position_col] if self.position_col else 'N/A'
            team = row[self.team_col] if self.team_col else 'N/A'
            print(f"  {idx}. {name} ({pos}, {team})")
        
        try:
            choice = input(f"\nSelect player (1-{len(display_matches)}) or 'c' to cancel: ").strip()
            if choice.lower() == 'c':
                return None
            
            num = int(choice)
            if 1 <= num <= len(display_matches):
                row = display_matches.iloc[num - 1]
                return {
                    'Player': row[self.player_col],
                    'Position': row[self.position_col] if self.position_col else 'Unknown',
                    'Team': row[self.team_col] if self.team_col else 'N/A'
                }
        except (ValueError, KeyboardInterrupt):
            return None
        
        return None
    
    def save_picks(self) -> None:
        """Save picks to CSV files."""
        if not self.picks:
            return
        
        df = pd.DataFrame(self.picks)
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_file = os.path.join(self.output_dir, f"draft_picks_{timestamp}.csv")
        df.to_csv(timestamped_file, index=False)
        
        # Save as latest
        latest_file = os.path.join(self.output_dir, "draft_picks_latest.csv")
        df.to_csv(latest_file, index=False)
        print(f"💾 Saved {len(self.picks)} picks")
    
    def undo_last_pick(self) -> bool:
        """Remove the last pick."""
        if not self.picks:
            return False
        
        removed = self.picks.pop()
        self.current_pick -= 1
        
        player_name = removed.get('player_name', 'Unknown')
        print(f"↩️  Removed: {player_name}")
        
        # Clear cache and update VBD
        self.vbd_cache.clear()
        if self.dynamic_vbd_enabled:
            self.update_dynamic_vbd()
        
        self.save_picks()
        return True
    
    def add_pick(self, player_name: str, position: str, pro_team: str) -> bool:
        """Add a pick to the draft."""
        team_num, round_num = self.get_snake_draft_team(self.current_pick)
        
        pick_info = {
            'overall_pick': self.current_pick,
            'player_name': str(player_name).strip(),
            'position': str(position).strip() if position else 'Unknown',
            'team_name': f"Team {team_num}",
            'pro_team': str(pro_team).strip() if pro_team and pro_team != 'N/A' else ''
        }
        
        # Check for duplicate
        existing_names = [p.get('player_name', '').lower() for p in self.picks]
        if pick_info['player_name'].lower() in existing_names:
            print(f"⚠️  {player_name} already drafted!")
            confirm = input("Continue anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                return False
        
        self.picks.append(pick_info)
        self.current_pick += 1
        
        print(f"✅ Pick #{pick_info['overall_pick']}: {pick_info['team_name']} selects {player_name} ({position})")
        
        # Update Dynamic VBD
        if self.dynamic_vbd_enabled:
            self.update_dynamic_vbd()
            self.show_position_runs()
        
        self.save_picks()
        return True
    
    def show_status(self) -> None:
        """Display current draft status."""
        if self.current_pick > TOTAL_PICKS:
            print("\n🏁 DRAFT COMPLETE!")
            return
        
        team_num, round_num = self.get_snake_draft_team(self.current_pick)
        
        print(f"\n📊 DRAFT STATUS")
        print("=" * 40)
        print(f"Next Pick: #{self.current_pick} (Round {round_num}, Team {team_num})")
        print(f"Progress: {self.current_pick - 1}/{TOTAL_PICKS} picks completed")
        print(f"Dynamic VBD: {'ENABLED' if self.dynamic_vbd_enabled else 'DISABLED'}")
        
        # Show recent picks
        if self.picks:
            print(f"\n📋 Recent Picks:")
            recent = self.picks[-3:] if len(self.picks) >= 3 else self.picks
            
            for pick in recent:
                pick_num = pick.get('overall_pick', 'N/A')
                player_name = pick.get('player_name', 'Unknown')
                position = pick.get('position', 'N/A')
                print(f"  {pick_num}. {player_name} ({position})")
    
    def run_interactive(self) -> None:
        """Run the simplified interactive draft session."""
        print("\n🏈 SIMPLIFIED BACKUP DRAFT TRACKER")
        print("=" * 50)
        print("Commands: UNDO, STATUS, RANKINGS, QUIT")
        print("=" * 50)
        
        # Load data
        if not self.load_player_database():
            print("❌ Cannot continue without player database!")
            return
        
        self.load_existing_picks()
        print(f"\n✅ Ready! Starting at pick #{self.current_pick}")
        
        # Main loop
        while self.current_pick <= TOTAL_PICKS:
            team_num, round_num = self.get_snake_draft_team(self.current_pick)
            
            print(f"\n─────────────────────────────────────")
            print(f"📍 Pick #{self.current_pick} (Round {round_num}, Team {team_num})")
            
            try:
                player_input = input("Enter player name (or command): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n⚠️  Interrupted! Type 'QUIT' to save and exit.")
                continue
            
            # Handle commands
            if player_input.upper() == 'QUIT':
                self.save_picks()
                print("✅ Draft saved!")
                break
            elif player_input.upper() == 'UNDO':
                if not self.undo_last_pick():
                    print("No picks to undo!")
                continue
            elif player_input.upper() == 'STATUS':
                self.show_status()
                continue
            elif player_input.upper() == 'RANKINGS':
                self.show_rankings(10)
                continue
            elif not player_input:
                continue
            
            # Find player
            player = self.find_player(player_input)
            
            if player is None:
                print(f"❌ No player found matching '{player_input}'")
                continue
            
            # Confirm and add pick
            player_name = player.get('Player', player_input)
            position = player.get('Position', 'Unknown')
            pro_team = player.get('Team', '')
            
            print(f"\n📋 Found: {player_name} ({position}, {pro_team})")
            try:
                confirm = input("Confirm? (y/n): ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                continue
            
            if confirm == 'y':
                self.add_pick(player_name, position, pro_team)
        
        # Draft complete
        if self.current_pick > TOTAL_PICKS:
            print("\n🎉 DRAFT COMPLETE!")
            print(f"📁 Final draft saved to: {self.output_dir}/draft_picks_latest.csv")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Simplified Emergency Backup Draft Tracker",
        epilog="Examples:\n  python backup_draft_simplified.py --dynamic-vbd"
    )
    
    parser.add_argument('--dynamic-vbd', action='store_true', help='Force enable Dynamic VBD')
    parser.add_argument('--no-dynamic-vbd', action='store_true', help='Force disable Dynamic VBD')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    try:
        args = parse_arguments()
        
        # Determine Dynamic VBD setting
        force_dynamic_vbd = None
        if args.dynamic_vbd and args.no_dynamic_vbd:
            print("❌ Cannot specify both --dynamic-vbd and --no-dynamic-vbd")
            sys.exit(1)
        elif args.dynamic_vbd:
            force_dynamic_vbd = True
        elif args.no_dynamic_vbd:
            force_dynamic_vbd = False
        
        # Run tracker
        tracker = SimplifiedDraftTracker(force_dynamic_vbd=force_dynamic_vbd)
        tracker.run_interactive()
        
    except KeyboardInterrupt:
        print("\n⚠️  Draft interrupted!")
        if 'tracker' in locals():
            tracker.save_picks()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        if 'tracker' in locals():
            tracker.save_picks()


if __name__ == "__main__":
    main()