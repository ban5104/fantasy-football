#!/usr/bin/env python3
"""
Live Draft Tracker for ESPN Fantasy Football

This script monitors your ESPN draft in real-time and can:
1. Display live pick updates
2. Update CSV files with draft data
3. Send notifications for picks
4. Integrate with your analysis notebooks
"""

import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, Any, List
from .espn_connection import ESPNLeagueConnector


class LiveDraftTracker:
    """
    Tracks live draft data and updates files/notebooks accordingly.
    """
    
    def __init__(self, output_dir: str = "data/draft"):
        """
        Initialize the draft tracker.
        
        Args:
            output_dir: Directory to save draft data files
        """
        self.output_dir = output_dir
        self.connector = None
        self.draft_data = []
        self.available_players = []
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def connect_to_league(self) -> bool:
        """Connect to the ESPN league."""
        self.connector = ESPNLeagueConnector.from_config()
        if self.connector.connect():
            print(f"✅ Connected to {self.connector.league.settings.name}")
            return True
        return False
    
    def save_draft_state(self):
        """Save current draft state to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save draft picks
        if self.draft_data:
            draft_df = pd.DataFrame(self.draft_data)
            draft_file = os.path.join(self.output_dir, f"draft_picks_{timestamp}.csv")
            draft_df.to_csv(draft_file, index=False)
            
            # Also save as latest
            latest_file = os.path.join(self.output_dir, "draft_picks_latest.csv")
            draft_df.to_csv(latest_file, index=False)
            
            print(f"📁 Draft data saved to {draft_file}")
        
        # Save available players
        if self.available_players:
            available_df = pd.DataFrame(self.available_players)
            available_file = os.path.join(self.output_dir, f"available_players_{timestamp}.csv")
            available_df.to_csv(available_file, index=False)
            
            # Also save as latest
            latest_available = os.path.join(self.output_dir, "available_players_latest.csv")
            available_df.to_csv(latest_available, index=False)
            
            print(f"📁 Available players saved to {available_file}")
    
    def update_cheat_sheet(self, new_pick: Dict[str, Any]):
        """
        Update your cheat sheet by removing drafted players.
        
        Args:
            new_pick: Dictionary containing pick information
        """
        try:
            cheat_sheet_file = "draft_cheat_sheet.csv"
            if os.path.exists(cheat_sheet_file):
                df = pd.read_csv(cheat_sheet_file)
                
                # Remove the drafted player
                player_name = new_pick['player']
                df = df[df['name'] != player_name]
                
                # Save updated cheat sheet
                df.to_csv(cheat_sheet_file, index=False)
                print(f"📋 Removed {player_name} from cheat sheet")
                
                # Also save backup
                backup_file = os.path.join(self.output_dir, f"cheat_sheet_after_pick_{new_pick['overall']}.csv")
                df.to_csv(backup_file, index=False)
                
        except Exception as e:
            print(f"⚠️  Error updating cheat sheet: {e}")
    
    def analyze_pick(self, new_pick: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a new pick and provide insights.
        
        Args:
            new_pick: Dictionary containing pick information
            
        Returns:
            Dictionary with pick analysis
        """
        analysis = {
            'pick_info': new_pick,
            'round_analysis': {},
            'position_analysis': {},
            'team_analysis': {}
        }
        
        try:
            # Load your projections for comparison
            projections_file = "data/projections/projections_all_positions_20250814.csv"
            if os.path.exists(projections_file):
                projections = pd.read_csv(projections_file)
                
                # Find the player in projections
                player_proj = projections[projections['name'].str.contains(new_pick['player'], na=False, case=False)]
                if not player_proj.empty:
                    analysis['projection'] = {
                        'projected_points': player_proj.iloc[0].get('projected_points', 'N/A'),
                        'rank': player_proj.index[0] + 1,
                        'adp': player_proj.iloc[0].get('adp', 'N/A')
                    }
        
        except Exception as e:
            print(f"⚠️  Error analyzing pick: {e}")
        
        return analysis
    
    def on_new_pick(self, pick_info: Dict[str, Any]):
        """
        Callback function for when a new pick is detected.
        
        Args:
            pick_info: Dictionary containing pick information
        """
        print(f"\n🏈 NEW PICK DETECTED!")
        print(f"Pick {pick_info['overall']}: {pick_info['team']} selects {pick_info['player']}")
        print(f"Position: {pick_info['position']} | Pro Team: {pick_info['pro_team']}")
        
        # Analyze the pick
        analysis = self.analyze_pick(pick_info)
        if 'projection' in analysis:
            proj = analysis['projection']
            print(f"📊 Projection Rank: #{proj['rank']} | Projected Points: {proj['projected_points']}")
        
        # Update cheat sheet
        self.update_cheat_sheet(pick_info)
        
        # Refresh and save data
        self.refresh_data()
        self.save_draft_state()
        
        print("-" * 50)
    
    def refresh_data(self):
        """Refresh draft and available player data."""
        if not self.connector:
            return
            
        # Get updated draft picks
        draft_picks = self.connector.get_draft_picks()
        if draft_picks:
            self.draft_data = draft_picks
        
        # Get updated available players
        available = self.connector.get_available_players(limit=100)
        if available:
            self.available_players = available
    
    def start_monitoring(self, check_interval: int = 10):
        """
        Start monitoring the draft for changes.
        
        Args:
            check_interval: How often to check for updates (seconds)
        """
        if not self.connect_to_league():
            print("❌ Failed to connect to league")
            return
        
        # Get initial state
        print("📥 Getting initial draft state...")
        self.refresh_data()
        self.save_draft_state()
        
        # Get draft status
        status = self.connector.get_draft_status()
        if status:
            print(f"📊 Draft Status: {status['picks_made']}/{status['total_picks']} picks made")
            if not status['is_complete']:
                print(f"⏭️  Next pick: {status['next_pick_team']}")
            else:
                print("✅ Draft is complete!")
                return
        
        print(f"\n👀 Starting live monitoring (checking every {check_interval} seconds)...")
        print("💡 Tip: Open your draft_cheat_sheet.csv in Excel/Sheets for live updates!")
        
        # Start monitoring
        self.connector.monitor_draft_changes(
            callback_func=self.on_new_pick,
            check_interval=check_interval
        )
    
    def get_current_status(self):
        """Get current draft status and display summary."""
        if not self.connector:
            if not self.connect_to_league():
                return
        
        # Get draft status
        status = self.connector.get_draft_status()
        draft_picks = self.connector.get_draft_picks()
        
        print(f"\n📊 CURRENT DRAFT STATUS")
        print("=" * 40)
        
        if status:
            print(f"Picks Made: {status['picks_made']}/{status['total_picks']}")
            print(f"Next Pick: {status['next_pick_team']}")
            print(f"Complete: {'Yes' if status['is_complete'] else 'No'}")
        
        if draft_picks:
            print(f"\n📋 Recent Picks:")
            recent_picks = draft_picks[-5:] if len(draft_picks) >= 5 else draft_picks
            for pick in recent_picks:
                print(f"  {pick['overall_pick']}. {pick['player_name']} ({pick['position']}) - {pick['team_name']}")
        
        # Show top available players by position
        positions = ['QB', 'RB', 'WR', 'TE']
        print(f"\n🎯 Top Available Players:")
        
        for pos in positions:
            available = self.connector.get_available_players(position=pos, limit=3)
            if available:
                print(f"\n{pos}:")
                for i, player in enumerate(available[:3], 1):
                    print(f"  {i}. {player['name']} ({player['team']})")


def main():
    """Main function to run the live draft tracker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ESPN Fantasy Football Live Draft Tracker")
    parser.add_argument("--status", action="store_true", help="Show current draft status")
    parser.add_argument("--monitor", action="store_true", help="Start live monitoring")
    parser.add_argument("--interval", type=int, default=10, help="Check interval in seconds (default: 10)")
    
    args = parser.parse_args()
    
    tracker = LiveDraftTracker()
    
    if args.status:
        tracker.get_current_status()
    elif args.monitor:
        tracker.start_monitoring(check_interval=args.interval)
    else:
        print("ESPN Fantasy Football Live Draft Tracker")
        print("=" * 40)
        print("Usage:")
        print("  python live_draft_tracker.py --status     # Show current status")
        print("  python live_draft_tracker.py --monitor    # Start live monitoring")
        print("  python live_draft_tracker.py --monitor --interval 5  # Check every 5 seconds")


if __name__ == "__main__":
    main()