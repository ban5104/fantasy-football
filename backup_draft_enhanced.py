#!/usr/bin/env python3
"""
Enhanced Backup Draft Tracker with Dynamic VBD
----------------------------------------------
Combines the reliability of manual draft tracking with Dynamic VBD intelligence.

Usage: python backup_draft_enhanced.py [--dynamic-vbd|--no-dynamic-vbd]
"""

import pandas as pd
import os
import sys
import yaml
import logging
from datetime import datetime
from typing import Optional, Dict
import argparse

# Configuration
NUM_TEAMS = 14
NUM_ROUNDS = 16
TOTAL_PICKS = NUM_TEAMS * NUM_ROUNDS  # 224 picks

class EnhancedDraftTracker:
    """Manual draft tracker with optional Dynamic VBD integration."""
    
    def __init__(self, enable_dynamic_vbd: Optional[bool] = None):
        self.picks = []
        self.current_pick = 1
        self.players_df = None
        self.output_dir = "data/draft"
        self.player_col = None
        self.position_col = None
        self.team_col = None
        
        # Dynamic VBD components
        self.dynamic_vbd_enabled = False
        self.dynamic_transformer = None
        self.last_vbd_rankings = None
        self.config = None
        self.enable_dynamic_vbd_override = enable_dynamic_vbd
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging for Dynamic VBD
        logging.basicConfig(level=logging.WARNING)
    
    def _find_columns(self, df: pd.DataFrame) -> bool:
        """Find player, position, and team columns."""
        cols = df.columns.str.lower()
        
        # Find player column
        for col in df.columns:
            if any(x in col.lower() for x in ['player', 'name']):
                self.player_col = col
                break
        else:
            self.player_col = df.columns[0]  # Use first column as fallback
        
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
        # Try multiple sources in order of preference
        sources = [
            "data/CSG Fantasy Football Sheet - 2025 v13.01.csv",
            "data/output/vbd_rankings_top300_*.csv",
            "data/output/draft_cheat_sheet.csv",
            "data/output/rankings_*.csv",
            "CSG Fantasy Football Sheet - 2025 v13.01.csv",
            "draft_cheat_sheet.csv", 
            "rankings.csv"
        ]
        
        # Expand glob patterns
        import glob
        expanded_sources = []
        for source in sources:
            if '*' in source:
                expanded_sources.extend(glob.glob(source))
            else:
                expanded_sources.append(source)
        
        for source in expanded_sources:
            if not os.path.exists(source):
                continue
                
            try:
                print(f"📂 Loading {source}...")
                
                # Try UTF-8 first, then latin-1 as fallback
                try:
                    self.players_df = pd.read_csv(source, encoding='utf-8')
                except UnicodeDecodeError:
                    self.players_df = pd.read_csv(source, encoding='latin-1')
                
                if self.players_df.empty:
                    continue
                
                # Find columns
                if not self._find_columns(self.players_df):
                    continue
                    
                # Remove rows with empty player names
                self.players_df = self.players_df[self.players_df[self.player_col].notna()].copy()
                self.players_df = self.players_df[self.players_df[self.player_col].str.strip() != ''].copy()
                
                print(f"✅ Loaded {len(self.players_df)} players from {source}")
                print(f"   Player column: {self.player_col}")
                if self.position_col:
                    print(f"   Position column: {self.position_col}")
                if self.team_col:
                    print(f"   Team column: {self.team_col}")
                
                # Check for fantasy points (needed for Dynamic VBD)
                fantasy_cols = [col for col in self.players_df.columns 
                              if any(x in col.upper() for x in ['FANTASY', 'PTS', 'POINTS', 'VBD'])]
                if fantasy_cols:
                    print(f"   Fantasy data: {fantasy_cols[:3]}")  # Show first 3
                
                return True
                    
            except Exception as e:
                print(f"⚠️  Failed to load {source}: {e}")
                continue
        
        print("❌ ERROR: No player database could be loaded!")
        print("Please ensure one of these files exists:")
        for source in sources[:3]:  # Show main sources
            print(f"  - {source}")
        return False
    
    def initialize_dynamic_vbd(self) -> bool:
        """Initialize Dynamic VBD if enabled and data available."""
        try:
            # Check override first
            if self.enable_dynamic_vbd_override is False:
                print("🔧 Dynamic VBD disabled by command line")
                return False
            
            # Load config
            config_path = "config/league-config.yaml"
            if not os.path.exists(config_path):
                print("⚠️ Config file not found - Dynamic VBD disabled")
                return False
            
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Check if enabled in config (unless overridden)
            config_enabled = self.config.get('dynamic_vbd', {}).get('enabled', False)
            if self.enable_dynamic_vbd_override is None and not config_enabled:
                print("📋 Dynamic VBD disabled in config")
                return False
            
            # Need fantasy points for VBD
            fantasy_cols = [col for col in self.players_df.columns 
                          if 'FANTASY' in col.upper() and 'PTS' in col.upper()]
            
            if not fantasy_cols:
                print("⚠️ Dynamic VBD requires fantasy points column (FANTASY_PTS)")
                print(f"   Available columns: {list(self.players_df.columns)}")
                return False
            
            # Standardize column name
            fantasy_col = fantasy_cols[0]
            if fantasy_col != 'FANTASY_PTS':
                self.players_df['FANTASY_PTS'] = self.players_df[fantasy_col]
            
            # Import and initialize Dynamic VBD
            sys.path.append('src')
            from dynamic_vbd import DynamicVBDTransformer
            
            self.dynamic_transformer = DynamicVBDTransformer(self.config, self.players_df)
            self.dynamic_vbd_enabled = True
            
            print("✅ Dynamic VBD initialized and ready!")
            print(f"   Scale: {self.dynamic_transformer.scale}")
            print(f"   Methods: {self.dynamic_transformer.methods}")
            return True
            
        except Exception as e:
            print(f"⚠️ Could not initialize Dynamic VBD: {e}")
            print("   Continuing with standard draft tracking...")
            return False
    
    def load_existing_picks(self) -> int:
        """Load existing picks if resuming from a crash."""
        latest_file = os.path.join(self.output_dir, "draft_picks_latest.csv")
        
        if not os.path.exists(latest_file):
            return 1
        
        try:
            print(f"📂 Loading existing picks...")
            
            # Try UTF-8 first, then latin-1 as fallback
            try:
                existing_df = pd.read_csv(latest_file, encoding='utf-8')
            except UnicodeDecodeError:
                existing_df = pd.read_csv(latest_file, encoding='latin-1')
            
            if existing_df.empty:
                print("📄 Existing picks file is empty, starting fresh")
                return 1
            
            # Convert to simple list format
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
            
            # Sort picks by overall_pick and set current pick
            self.picks.sort(key=lambda x: x['overall_pick'])
            self.current_pick = len(self.picks) + 1
            
            print(f"✅ Resumed draft with {len(self.picks)} picks at pick #{self.current_pick}")
            
            # Update Dynamic VBD if enabled
            if self.dynamic_vbd_enabled:
                self.update_dynamic_rankings()
            
            return self.current_pick
                
        except Exception as e:
            print(f"⚠️  Could not load existing picks: {e}, starting fresh")
            return 1
    
    def get_snake_draft_team(self, pick_number: int) -> tuple:
        """Calculate team and round for snake draft."""
        round_num = ((pick_number - 1) // NUM_TEAMS) + 1
        
        if round_num % 2 == 1:  # Odd rounds: 1->14
            team_num = ((pick_number - 1) % NUM_TEAMS) + 1
        else:  # Even rounds: 14->1
            team_num = NUM_TEAMS - ((pick_number - 1) % NUM_TEAMS)
        
        return team_num, round_num
    
    def update_dynamic_rankings(self):
        """Update Dynamic VBD rankings after a pick."""
        if not self.dynamic_vbd_enabled or not self.dynamic_transformer:
            return
        
        try:
            from dynamic_vbd import create_draft_state, create_probability_forecast
            
            # Get available players
            drafted_names = [p['player_name'] for p in self.picks]
            available = self.players_df[
                ~self.players_df[self.player_col].isin(drafted_names)
            ].copy()
            
            if available.empty:
                return
            
            # Create draft state
            draft_state = create_draft_state(
                current_pick=self.current_pick,
                drafted_players=drafted_names
            )
            
            # Calculate position probabilities based on remaining players
            position_counts = available[self.position_col].value_counts(normalize=True) if self.position_col else {}
            
            # Apply draft stage adjustments
            stage_adjustments = self.get_draft_stage_multipliers()
            adjusted_probs = {}
            for pos, prob in position_counts.items():
                multiplier = stage_adjustments.get(pos, 1.0)
                adjusted_probs[pos] = prob * multiplier
            
            # Normalize
            total_prob = sum(adjusted_probs.values())
            if total_prob > 0:
                adjusted_probs = {k: v/total_prob for k, v in adjusted_probs.items()}
            
            # Create forecast
            picks_remaining = TOTAL_PICKS - self.current_pick + 1
            horizon = min(max(3, picks_remaining // 20), 7)
            
            forecast = create_probability_forecast(
                horizon_picks=horizon,
                position_probs=adjusted_probs
            )
            
            # Get updated rankings
            self.last_vbd_rankings = self.dynamic_transformer.transform(
                available, forecast, draft_state
            )
            
        except Exception as e:
            logging.warning(f"Dynamic VBD update failed: {e}")
            self.last_vbd_rankings = None
    
    def get_draft_stage_multipliers(self) -> Dict[str, float]:
        """Get position probability multipliers based on draft stage."""
        draft_progress = self.current_pick / TOTAL_PICKS
        
        if draft_progress < 0.3:  # Early draft
            return {'QB': 1.2, 'RB': 1.3, 'WR': 1.2, 'TE': 1.1, 'DEF': 0.2, 'K': 0.1}
        elif draft_progress < 0.7:  # Mid draft
            return {'QB': 1.0, 'RB': 1.1, 'WR': 1.1, 'TE': 1.0, 'DEF': 0.6, 'K': 0.3}
        else:  # Late draft
            return {'QB': 0.8, 'RB': 0.9, 'WR': 0.9, 'TE': 0.8, 'DEF': 1.5, 'K': 1.8}
    
    def find_player(self, query: str) -> Optional[Dict]:
        """Find player with simple string matching."""
        if not query or len(query) < 2:
            return None
        
        if self.players_df is None:
            print("❌ No player database loaded")
            return None
        
        query_lower = query.lower().strip()
        
        # Get available players (not already drafted)
        drafted_names = [p['player_name'] for p in self.picks]
        available_players = self.players_df[~self.players_df[self.player_col].isin(drafted_names)].copy()
        
        if available_players.empty:
            print("⚠️  No available players remaining")
            return None
        
        # Simple string matching
        matches = available_players[
            available_players[self.player_col].str.lower().str.contains(query_lower, na=False, regex=False)
        ]
        
        if matches.empty:
            return None
        
        if len(matches) == 1:
            # Single match - return immediately
            row = matches.iloc[0]
            return {
                'Player': row[self.player_col],
                'Position': row[self.position_col] if self.position_col else 'Unknown',
                'Team': row[self.team_col] if self.team_col else 'N/A'
            }
        
        # Multiple matches - show selection with VBD info if available
        print(f"\n🔍 Multiple players found:")
        display_matches = matches.head(10)
        
        for idx, (_, row) in enumerate(display_matches.iterrows(), 1):
            player_name = row[self.player_col]
            pos = row[self.position_col] if self.position_col else 'N/A'
            team = row[self.team_col] if self.team_col else 'N/A'
            
            # Add VBD info if available
            vbd_info = ""
            if (self.dynamic_vbd_enabled and self.last_vbd_rankings is not None 
                and not self.last_vbd_rankings.empty):
                player_vbd = self.last_vbd_rankings[
                    self.last_vbd_rankings[self.player_col] == player_name
                ]
                if not player_vbd.empty and 'VBD_BLENDED' in player_vbd.columns:
                    vbd_value = player_vbd['VBD_BLENDED'].iloc[0]
                    vbd_info = f" [VBD: {vbd_value:.1f}]"
            
            print(f"  {idx}. {player_name} ({pos}, {team}){vbd_info}")
        
        while True:
            try:
                choice = input(f"\nSelect player number (1-{len(display_matches)}) or 'c' to cancel: ").strip()
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
                else:
                    print(f"Please enter a number between 1 and {len(display_matches)}")
            except ValueError:
                print("Invalid selection. Try again.")
            except KeyboardInterrupt:
                return None
    
    def save_picks(self):
        """Save picks in ESPN-compatible format."""
        if not self.picks:
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.picks)
        
        # Save with timestamp 
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_file = os.path.join(self.output_dir, f"draft_picks_{timestamp}.csv")
        df.to_csv(timestamped_file, index=False)
        
        # Save as latest (this is what notebooks read)
        latest_file = os.path.join(self.output_dir, "draft_picks_latest.csv")
        df.to_csv(latest_file, index=False)
        print(f"💾 Saved {len(self.picks)} picks to {latest_file}")
    
    def undo_last_pick(self):
        """Remove the last pick."""
        if not self.picks:
            return False
        
        removed = self.picks.pop()
        self.current_pick -= 1
        
        player_name = removed.get('player_name', 'Unknown')
        pick_num = removed.get('overall_pick', 'Unknown')
        print(f"↩️  Removed: {player_name} (pick #{pick_num})")
        
        # Update Dynamic VBD
        if self.dynamic_vbd_enabled:
            self.update_dynamic_rankings()
        
        self.save_picks()
        return True
    
    def add_pick(self, player_name: str, position: str, pro_team: str):
        """Add a pick to the draft."""
        # Calculate team and round
        team_num, round_num = self.get_snake_draft_team(self.current_pick)
        
        # Clean inputs
        player_name = str(player_name).strip()
        position = str(position).strip() if position else 'Unknown'
        pro_team = str(pro_team).strip() if pro_team and pro_team != 'N/A' else ''
        
        pick_info = {
            'overall_pick': self.current_pick,
            'player_name': player_name,
            'position': position,
            'team_name': f"Team {team_num}",
            'pro_team': pro_team
        }
        
        # Check for duplicate player
        existing_names = [p.get('player_name', '').lower() for p in self.picks]
        if player_name.lower() in existing_names:
            print(f"⚠️  Warning: {player_name} appears to already be drafted")
            confirm = input("Continue anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                return False
        
        self.picks.append(pick_info)
        self.current_pick += 1
        
        print(f"✅ Pick #{pick_info['overall_pick']}: {pick_info['team_name']} selects {player_name} ({position}, {pro_team})")
        
        # Update Dynamic VBD rankings
        if self.dynamic_vbd_enabled:
            self.update_dynamic_rankings()
            self.show_vbd_impact(player_name, position)
        
        # Auto-save after each pick
        self.save_picks()
        return True
    
    def show_vbd_impact(self, drafted_player: str, position: str):
        """Show VBD impact after a pick."""
        if not self.dynamic_vbd_enabled or not self.last_vbd_rankings is not None:
            return
        
        try:
            # Show top 3 at same position if available
            if self.position_col and not self.last_vbd_rankings.empty:
                same_pos = self.last_vbd_rankings[
                    self.last_vbd_rankings[self.position_col] == position
                ]
                if not same_pos.empty and 'VBD_BLENDED' in same_pos.columns:
                    top_pos = same_pos.nlargest(3, 'VBD_BLENDED')
                    print(f"📊 Top remaining {position}s:")
                    for i, (_, row) in enumerate(top_pos.iterrows(), 1):
                        player = row[self.player_col]
                        vbd = row['VBD_BLENDED']
                        print(f"   {i}. {player} (VBD: {vbd:.1f})")
        except Exception:
            pass  # Don't let VBD display errors interrupt the draft
    
    def show_dynamic_rankings(self, top_n: int = 10):
        """Display top available players by Dynamic VBD."""
        if not self.dynamic_vbd_enabled:
            print("📋 Dynamic VBD not enabled")
            print("   Use --dynamic-vbd flag to enable")
            return
        
        if self.last_vbd_rankings is None:
            print("🔄 Calculating Dynamic VBD rankings...")
            self.update_dynamic_rankings()
        
        if self.last_vbd_rankings is not None and not self.last_vbd_rankings.empty:
            print(f"\n🏆 Top {top_n} Available (Dynamic VBD):")
            if 'VBD_BLENDED' in self.last_vbd_rankings.columns:
                top = self.last_vbd_rankings.nlargest(top_n, 'VBD_BLENDED')
                for i, (_, row) in enumerate(top.iterrows(), 1):
                    player = row[self.player_col]
                    pos = row.get(self.position_col, 'N/A')
                    vbd = row['VBD_BLENDED']
                    team = row.get(self.team_col, '')
                    team_str = f", {team}" if team and team != 'N/A' else ""
                    print(f"  {i:2d}. {player} ({pos}{team_str}) - VBD: {vbd:5.1f}")
                
                # Show cache info
                if hasattr(self.dynamic_transformer, 'get_cache_stats'):
                    cache_stats = self.dynamic_transformer.get_cache_stats()
                    print(f"\n💾 Cache: {cache_stats['cache_size']} scenarios cached")
            else:
                print("⚠️ VBD calculations not available")
        else:
            print("❌ Could not calculate Dynamic VBD rankings")
    
    def show_status(self):
        """Display current draft status."""
        if self.current_pick > TOTAL_PICKS:
            print("\n🏁 DRAFT COMPLETE!")
            return
        
        team_num, round_num = self.get_snake_draft_team(self.current_pick)
        
        print(f"\n📊 DRAFT STATUS")
        print("=" * 40)
        print(f"Next Pick: #{self.current_pick} (Round {round_num}, Team {team_num})")
        print(f"Progress: {self.current_pick - 1}/{TOTAL_PICKS} picks completed")
        
        # Show Dynamic VBD status
        if self.dynamic_vbd_enabled:
            print(f"🔄 Dynamic VBD: ✅ Active")
        else:
            print(f"📋 Dynamic VBD: ⚫ Disabled")
        
        # Show last 5 picks
        if self.picks:
            print(f"\n📋 Recent Picks:")
            recent_picks = self.picks[-5:] if len(self.picks) >= 5 else self.picks
            
            for pick in recent_picks:
                pick_num = pick.get('overall_pick', 'N/A')
                player_name = pick.get('player_name', 'Unknown')
                position = pick.get('position', 'N/A')
                team_name = pick.get('team_name', 'Unknown Team')
                print(f"  {pick_num}. {player_name} ({position}) - {team_name}")
        
        # Show database status
        if self.players_df is not None:
            print(f"\n📋 Database Status:")
            print(f"  Player Column: {self.player_col}")
            if self.position_col:
                print(f"  Position Column: {self.position_col}")
            if self.team_col:
                print(f"  Team Column: {self.team_col}")
    
    def run_interactive(self):
        """Run the interactive draft session."""
        print("\n🏈 ENHANCED BACKUP DRAFT TRACKER")
        print("=" * 45)
        print("Manual draft tracking with Dynamic VBD intelligence")
        
        if self.dynamic_vbd_enabled:
            print("🔄 Dynamic VBD: ENABLED")
        else:
            print("📋 Dynamic VBD: DISABLED")
        
        print("\nCommands: UNDO, STATUS, RANKINGS, QUIT")
        print("=" * 45)
        
        # Load player database
        print("\n📂 Loading player database...")
        if not self.load_player_database():
            print("❌ Cannot continue without player database!")
            return
        
        # Initialize Dynamic VBD
        if self.enable_dynamic_vbd_override is not False:
            print("\n🔄 Initializing Dynamic VBD...")
            self.initialize_dynamic_vbd()
        
        # Check for existing picks
        print("\n📂 Checking for existing draft progress...")
        self.load_existing_picks()
        
        print(f"\n✅ Draft tracker ready! Starting at pick #{self.current_pick}")
        
        # Main draft loop
        while self.current_pick <= TOTAL_PICKS:
            team_num, round_num = self.get_snake_draft_team(self.current_pick)
            
            print(f"\n─────────────────────────────────────")
            print(f"📍 Pick #{self.current_pick} (Round {round_num}, Team {team_num})")
            
            # Get player input
            try:
                player_input = input("Enter player name (or command): ").strip()
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted! Type 'QUIT' to save and exit.")
                continue
            except EOFError:
                print("\n\n⚠️  Input ended. Saving and exiting...")
                break
            
            # Handle commands
            if player_input.upper() == 'QUIT':
                print("💾 Saving and exiting...")
                self.save_picks()
                print("✅ Draft saved successfully!")
                break
                
            elif player_input.upper() == 'UNDO':
                if self.undo_last_pick():
                    continue
                else:
                    print("No picks to undo!")
                    continue
                    
            elif player_input.upper() == 'STATUS':
                self.show_status()
                continue
                
            elif player_input.upper() == 'RANKINGS':
                self.show_dynamic_rankings()
                continue
                
            elif not player_input:
                continue
            
            # Find player
            player = self.find_player(player_input)
            
            if player is None:
                print(f"❌ No player found matching '{player_input}'")
                print("Try typing fewer characters or check spelling")
                continue
            
            # Extract player info
            player_name = player.get('Player', player_input)
            position = player.get('Position', 'Unknown')
            pro_team = player.get('Team', '')
            
            # Confirm selection
            print(f"\n📋 Found: {player_name} ({position}, {pro_team})")
            try:
                confirm = input("Confirm? (y/n): ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\nCancelled - try again")
                continue
            
            if confirm == 'y':
                if self.add_pick(player_name, position, pro_team):
                    # Success - continue to next pick
                    pass
                else:
                    print("Pick not added - try again")
                    continue
            else:
                print("Cancelled - try again")
        
        # Draft complete
        if self.current_pick > TOTAL_PICKS:
            print("\n🎉 DRAFT COMPLETE! All picks recorded.")
            print(f"📁 Final draft saved to: {self.output_dir}/draft_picks_latest.csv")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Enhanced Backup Draft Tracker')
    parser.add_argument('--dynamic-vbd', action='store_true', 
                       help='Force enable Dynamic VBD')
    parser.add_argument('--no-dynamic-vbd', action='store_true',
                       help='Disable Dynamic VBD')
    args = parser.parse_args()
    
    # Determine Dynamic VBD setting
    enable_dynamic_vbd = None
    if args.dynamic_vbd:
        enable_dynamic_vbd = True
    elif args.no_dynamic_vbd:
        enable_dynamic_vbd = False
    
    try:
        tracker = EnhancedDraftTracker(enable_dynamic_vbd=enable_dynamic_vbd)
        tracker.run_interactive()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Draft interrupted! Saving progress...")
        if 'tracker' in locals():
            tracker.save_picks()
            print("💾 Progress saved. Run again to resume.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        if 'tracker' in locals():
            try:
                tracker.save_picks()
                print("✅ Emergency save successful!")
            except:
                print("❌ Could not save progress")
        sys.exit(1)


if __name__ == "__main__":
    main()