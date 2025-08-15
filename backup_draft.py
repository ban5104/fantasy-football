#!/usr/bin/env python3
"""
Emergency Backup Draft Tracker
-------------------------------
Simple terminal-based draft tracker for when ESPN API fails.
Basic and reliable for live drafts.

Usage: python backup_draft.py
"""

import pandas as pd
import os
import sys
from datetime import datetime
from typing import Optional, Dict

# Configuration
NUM_TEAMS = 14
NUM_ROUNDS = 16
TOTAL_PICKS = NUM_TEAMS * NUM_ROUNDS  # 224 picks

class BackupDraftTracker:
    """Minimal draft tracker focused on essential functionality."""
    
    def __init__(self):
        self.picks = []
        self.current_pick = 1
        self.players_df = None
        self.output_dir = "data/draft"
        self.player_col = None
        self.position_col = None
        self.team_col = None
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
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
            "CSG Fantasy Football Sheet - 2025 v13.01.csv",
            "draft_cheat_sheet.csv", 
            "rankings.csv"
        ]
        
        for source in sources:
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
                
                return True
                    
            except Exception as e:
                print(f"⚠️  Failed to load {source}: {e}")
                continue
        
        print("❌ ERROR: No player database could be loaded!")
        print("Please ensure one of these files exists:")
        for source in sources:
            print(f"  - {source}")
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
        
        # Multiple matches - show selection
        print(f"\n🔍 Multiple players found:")
        display_matches = matches.head(10)
        
        for idx, (_, row) in enumerate(display_matches.iterrows(), 1):
            player_name = row[self.player_col]
            pos = row[self.position_col] if self.position_col else 'N/A'
            team = row[self.team_col] if self.team_col else 'N/A'
            print(f"  {idx}. {player_name} ({pos}, {team})")
        
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
        
        # Auto-save after each pick
        self.save_picks()
        return True
    
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
        print("\n🏈 BACKUP DRAFT TRACKER")
        print("=" * 40)
        print("Emergency draft tracking when ESPN API fails")
        print("\nCommands: UNDO, STATUS, QUIT")
        print("=" * 40)
        
        # Load player database
        print("\n📂 Loading player database...")
        if not self.load_player_database():
            print("❌ Cannot continue without player database!")
            return
        
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
    try:
        tracker = BackupDraftTracker()
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