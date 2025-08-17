#!/usr/bin/env python3
"""
Emergency Backup Draft Tracker
-------------------------------
Terminal-based draft tracker for when ESPN API fails.
Simple, reliable draft tracking with Monte Carlo integration.

Usage: 
  python backup_draft.py
"""

import pandas as pd
import os
import sys
import yaml
import json
from datetime import datetime
from typing import Optional, Dict, List, Tuple


# Configuration
NUM_TEAMS = 14
NUM_ROUNDS = 16
TOTAL_PICKS = NUM_TEAMS * NUM_ROUNDS  # 224 picks


class BackupDraftTracker:
    """Simple, reliable draft tracker with Monte Carlo integration."""
    
    def __init__(self):
        self.picks = []
        self.current_pick = 1
        self.players_df = None
        self.output_dir = "data/draft"
        self.player_col = None
        self.position_col = None
        self.team_col = None
        
        # Monte Carlo integration
        self.my_draft_position = None  # 1-based position (1-14)
        self.team_names = self.load_team_names_from_config()
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    
    def load_team_names_from_config(self) -> List[str]:
        """Load team names from config with fallback."""
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'config', 'league-config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            team_names = config.get('team_names', [])
            if len(team_names) >= 14:
                return team_names[:14]
            
            num_teams = config.get('basic_settings', {}).get('teams', 14)
            return [f"Team {i}" for i in range(1, num_teams + 1)]
                
        except Exception as e:
            print(f"⚠️ Could not load team names from config: {e}")
            return [f"Team {i}" for i in range(1, 15)]
    
    def select_draft_position(self) -> None:
        """Interactive team selection."""
        print("\n🏈 SELECT YOUR DRAFT POSITION")
        print("=" * 40)
        
        for i, name in enumerate(self.team_names, 1):
            print(f"  {i:2d}. {name}")
        
        while True:
            try:
                choice = input(f"\nSelect your position (1-{len(self.team_names)}): ").strip()
                pos = int(choice)
                
                if 1 <= pos <= len(self.team_names):
                    self.my_draft_position = pos
                    print(f"✅ You are: {self.team_names[pos-1]} (Pick #{pos})")
                    print(f"📍 Your picks: #{pos}, #{NUM_TEAMS*2-pos+1}, #{NUM_TEAMS*2+pos-1}, ...")
                    return
                
                print(f"❌ Position must be between 1 and {len(self.team_names)}")
                    
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n❌ Draft cancelled")
                sys.exit(0)
    
    def export_monte_carlo_state(self) -> None:
        """Export state for Monte Carlo simulator."""
        if not self.my_draft_position or not self.team_names:
            return
        
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            
            my_team_name = self.team_names[self.my_draft_position - 1]
            my_picks = [p for p in self.picks if p['team_name'] == my_team_name]
            
            state = {
                'my_team_idx': self.my_draft_position - 1,
                'current_global_pick': self.current_pick - 1,
                'my_current_roster': [p['player_name'] for p in my_picks],
                'all_drafted': [p['player_name'] for p in self.picks],
                'timestamp': datetime.now().isoformat(),
                'team_name': my_team_name,
                'total_teams': len(self.team_names)
            }
            
            state_file = os.path.join(self.output_dir, "monte_carlo_state.json")
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            print(f"📡 Monte Carlo state exported to {state_file}")
            
        except Exception as e:
            print(f"⚠️ Error exporting Monte Carlo state: {e}")
    
    def show_draft_order(self) -> None:
        """Display the snake draft order."""
        print("\n🐍 SNAKE DRAFT ORDER")
        print("=" * 50)
        
        for round_num in range(1, 16):  # 15 rounds
            print(f"Round {round_num:2d}: ", end="")
            
            if round_num % 2 == 1:  # Odd rounds
                teams = [f"T{i}" for i in range(1, 15)]
            else:  # Even rounds (reverse)
                teams = [f"T{i}" for i in range(14, 0, -1)]
            
            print(" → ".join(teams))
        
        print(f"\n✅ You are T{self.my_draft_position}")
    
    
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
        # Try common data sources
        sources = [
            "analysis/CSG Fantasy Football Sheet - 2025 v13.01.csv",
            "data/CSG Fantasy Football Sheet - 2025 v13.01.csv",
            "CSG Fantasy Football Sheet - 2025 v13.01.csv",
            "data/output/draft_cheat_sheet.csv",
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
    
    def get_snake_draft_team(self, pick_number: int) -> Tuple[int, int]:
        """Calculate team and round for snake draft."""
        round_num = ((pick_number - 1) // NUM_TEAMS) + 1
        
        if round_num % 2 == 1:  # Odd rounds: 1->14
            team_num = ((pick_number - 1) % NUM_TEAMS) + 1
        else:  # Even rounds: 14->1
            team_num = NUM_TEAMS - ((pick_number - 1) % NUM_TEAMS)
        
        return team_num, round_num
    
    def show_available_players(self, top_n: int = 10) -> None:
        """Show top available players."""
        if self.players_df is None:
            print("⚠️  No player database loaded")
            return
            
        # Get available players
        drafted_names = [p['player_name'] for p in self.picks]
        available_players = self.players_df[~self.players_df[self.player_col].isin(drafted_names)].copy()
        
        if available_players.empty:
            print("⚠️  No players remaining!")
            return
        
        # Show first N available players
        top_players = available_players.head(top_n)
        
        print(f"\n📋 Top {top_n} Available Players:")
        print("=" * 50)
        
        for i, (_, player) in enumerate(top_players.iterrows(), 1):
            name = player[self.player_col]
            pos = player[self.position_col] if self.position_col else 'N/A'
            team = player[self.team_col] if self.team_col else 'N/A'
            print(f"  {i:2d}. {name:<25} ({pos:3s}, {team:4s})")
    
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
    
    def save_picks(self) -> None:
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
        
        # Export state for Monte Carlo
        self.export_monte_carlo_state()
    
    def undo_last_pick(self) -> bool:
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
    
    def add_pick(self, player_name: str, position: str, pro_team: str) -> bool:
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
            'team_name': self.team_names[team_num - 1] if team_num <= len(self.team_names) else f"Team {team_num}",
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
            print(f"  Total Players: {len(self.players_df)}")
    
    def run_interactive(self) -> None:
        """Run the interactive draft session."""
        print("\n🏈 BACKUP DRAFT TRACKER")
        print("=" * 40)
        print("Emergency draft tracking when ESPN API fails")
        print("\nCommands: UNDO, STATUS, PLAYERS, ORDER, HELP, QUIT")
        print("=" * 40)
        
        # Load player database
        print("\n📂 Loading player database...")
        if not self.load_player_database():
            print("❌ Cannot continue without player database!")
            return
        
        # Check for existing picks
        print("\n📂 Checking for existing draft progress...")
        self.load_existing_picks()
        
        # Team selection at startup
        if not self.my_draft_position:
            self.select_draft_position()
        
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
                
            elif player_input.upper() == 'PLAYERS':
                self.show_available_players(10)
                continue
                
            elif player_input.upper() == 'ORDER':
                self.show_draft_order()
                continue
                
            elif player_input.upper() == 'HELP':
                print("\n📋 Available Commands:")
                print("  UNDO     - Remove the last pick")
                print("  STATUS   - Show current draft status and recent picks")
                print("  PLAYERS  - Show top 10 available players")
                print("  ORDER    - Display snake draft order")
                print("  HELP     - Show this help message")
                print("  QUIT     - Save and exit")
                print("\nTo draft a player, type their name (partial matches work)")
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
        # Initialize tracker
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