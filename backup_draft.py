#!/usr/bin/env python3
"""
Emergency Backup Draft Tracker with Dynamic VBD Integration
-----------------------------------------------------------
Terminal-based draft tracker for when ESPN API fails.
Enhanced with real-time Dynamic VBD adjustments.

Usage: 
  python backup_draft.py                 # Use config setting
  python backup_draft.py --dynamic-vbd   # Force enable Dynamic VBD
  python backup_draft.py --no-dynamic-vbd # Force disable Dynamic VBD
"""

import pandas as pd
import os
import sys
import argparse
import yaml
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
import logging
import importlib.util

# Cache for dynamic imports to avoid sys.path manipulation
_DYNAMIC_VBD_MODULE = None
_VBD_MODULE = None


def _import_dynamic_vbd():
    """Safely import dynamic VBD module using importlib."""
    global _DYNAMIC_VBD_MODULE
    if _DYNAMIC_VBD_MODULE is None:
        try:
            spec = importlib.util.spec_from_file_location(
                "dynamic_vbd", 
                os.path.join(os.path.dirname(__file__), "src", "dynamic_vbd.py")
            )
            if spec and spec.loader:
                _DYNAMIC_VBD_MODULE = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_DYNAMIC_VBD_MODULE)
        except Exception as e:
            logging.error(f"Failed to import dynamic_vbd module: {e}")
            raise ImportError(f"Could not import dynamic_vbd: {e}")
    return _DYNAMIC_VBD_MODULE


def _import_vbd():
    """Safely import VBD module using importlib."""
    global _VBD_MODULE
    if _VBD_MODULE is None:
        try:
            spec = importlib.util.spec_from_file_location(
                "vbd", 
                os.path.join(os.path.dirname(__file__), "src", "vbd.py")
            )
            if spec and spec.loader:
                _VBD_MODULE = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_VBD_MODULE)
        except Exception as e:
            logging.error(f"Failed to import vbd module: {e}")
            raise ImportError(f"Could not import vbd: {e}")
    return _VBD_MODULE


# Configuration
NUM_TEAMS = 14
NUM_ROUNDS = 16
TOTAL_PICKS = NUM_TEAMS * NUM_ROUNDS  # 224 picks

# Draft stage thresholds - defaults (can be overridden by config)
DEFAULT_DRAFT_STAGE_EARLY_THRESHOLD = 0.3  # First 30% of draft picks
DEFAULT_DRAFT_STAGE_LATE_THRESHOLD = 0.7   # Last 30% of draft picks

class BackupDraftTracker:
    """Enhanced draft tracker with Dynamic VBD integration."""
    
    def __init__(self, force_dynamic_vbd: Optional[bool] = None):
        self.picks = []
        self.current_pick = 1
        self.players_df = None
        self.output_dir = "data/draft"
        self.player_col = None
        self.position_col = None
        self.team_col = None
        
        # Dynamic VBD components
        self.config = None
        self.dynamic_vbd_transformer = None
        self.dynamic_vbd_enabled = False
        self.force_dynamic_vbd = force_dynamic_vbd
        
        # Draft stage thresholds (loaded from config or defaults)
        self.draft_stage_early_threshold = DEFAULT_DRAFT_STAGE_EARLY_THRESHOLD
        self.draft_stage_late_threshold = DEFAULT_DRAFT_STAGE_LATE_THRESHOLD
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load configuration for Dynamic VBD
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """
        Load league configuration for Dynamic VBD.
        
        Configuration Precedence (highest to lowest priority):
        1. Command line flags (--dynamic-vbd or --no-dynamic-vbd)
        2. Config file setting (dynamic_vbd.enabled in league-config.yaml)
        3. Default: disabled
        """
        config_path = "config/league-config.yaml"
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                
                # Configuration precedence logic - explicit and documented
                if self.force_dynamic_vbd is not None:
                    # Priority 1: Command line override takes highest precedence
                    self.dynamic_vbd_enabled = self.force_dynamic_vbd
                    source = "command line flag"
                    print(f"🔧 Dynamic VBD setting: {self.dynamic_vbd_enabled} (from {source})")
                else:
                    # Priority 2: Config file setting
                    config_enabled = self.config.get('dynamic_vbd', {}).get('enabled', False)
                    self.dynamic_vbd_enabled = config_enabled
                    source = "configuration file" if config_enabled else "default (disabled)"
                    print(f"🔧 Dynamic VBD setting: {self.dynamic_vbd_enabled} (from {source})")
                
                # Load draft stage thresholds from config if available
                draft_stages = self.config.get('dynamic_vbd', {}).get('draft_stages', {})
                self.draft_stage_early_threshold = draft_stages.get('early_threshold', DEFAULT_DRAFT_STAGE_EARLY_THRESHOLD)
                self.draft_stage_late_threshold = draft_stages.get('late_threshold', DEFAULT_DRAFT_STAGE_LATE_THRESHOLD)
                
                if self.dynamic_vbd_enabled:
                    print("🚀 Dynamic VBD will be used if data supports it")
                    print(f"   Draft stages: Early < {self.draft_stage_early_threshold:.1%}, Late > {self.draft_stage_late_threshold:.1%}")
                else:
                    print("📊 Using static VBD rankings only")
            else:
                # No config file found
                if self.force_dynamic_vbd is not None:
                    self.dynamic_vbd_enabled = self.force_dynamic_vbd
                    print(f"⚠️  No league config found, using command line override: {self.dynamic_vbd_enabled}")
                else:
                    self.dynamic_vbd_enabled = False
                    print("⚠️  No league config found, Dynamic VBD disabled by default")
        except Exception as e:
            print(f"⚠️  Error loading config: {e}")
            # Fallback to command line override if available
            if self.force_dynamic_vbd is not None:
                self.dynamic_vbd_enabled = self.force_dynamic_vbd
                print(f"    Using command line override: {self.dynamic_vbd_enabled}")
            else:
                self.dynamic_vbd_enabled = False
                print("    Dynamic VBD disabled due to configuration error")
    
    def initialize_dynamic_vbd(self, df: pd.DataFrame) -> bool:
        """Initialize Dynamic VBD transformer if data supports it."""
        if not self.dynamic_vbd_enabled or self.config is None:
            return False
        
        # Check if DataFrame has required FANTASY_PTS column
        if 'FANTASY_PTS' not in df.columns:
            print("⚠️  No FANTASY_PTS column found, Dynamic VBD disabled")
            return False
        
        try:
            # Import Dynamic VBD components using safe import
            dynamic_vbd_module = _import_dynamic_vbd()
            
            # Initialize transformer
            self.dynamic_vbd_transformer = dynamic_vbd_module.DynamicVBDTransformer(self.config)
            print("✅ Dynamic VBD transformer initialized")
            return True
            
        except ImportError as e:
            print(f"⚠️  Dynamic VBD module not found: {e}")
            print("    Please ensure src/dynamic_vbd.py exists and is accessible")
            return False
        except Exception as e:
            print(f"⚠️  Error initializing Dynamic VBD: {e}")
            print("    Dynamic VBD will be disabled for this session")
            return False
    
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
        """Load player data from available CSV files, preferring VBD rankings."""
        # Try VBD ranking files first for Dynamic VBD support
        vbd_sources = []
        if os.path.exists("data/output"):
            import glob
            # Look for VBD ranking files with most recent date
            vbd_patterns = [
                "data/output/vbd_rankings_top300_*.csv",
                "data/output/rankings_vbd_*_top300_*.csv", 
                "data/output/rankings_statistical_vbd_top300_*.csv"
            ]
            for pattern in vbd_patterns:
                vbd_sources.extend(sorted(glob.glob(pattern), reverse=True))
        
        # Fallback sources
        fallback_sources = [
            "data/CSG Fantasy Football Sheet - 2025 v13.01.csv",
            "CSG Fantasy Football Sheet - 2025 v13.01.csv",
            "data/output/draft_cheat_sheet.csv",
            "draft_cheat_sheet.csv", 
            "rankings.csv"
        ]
        
        # Combine all sources with VBD files first
        all_sources = vbd_sources + fallback_sources
        
        for source in all_sources:
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
                
                # Check for VBD columns
                vbd_columns = [col for col in self.players_df.columns if 'VBD' in col.upper()]
                fantasy_pts = 'FANTASY_PTS' in self.players_df.columns
                
                if vbd_columns or fantasy_pts:
                    print(f"   📈 VBD data available: {vbd_columns if vbd_columns else ['FANTASY_PTS']}")
                    
                    # Try to initialize Dynamic VBD
                    if self.initialize_dynamic_vbd(self.players_df):
                        print("   🚀 Dynamic VBD ready for live updates")
                    else:
                        print("   📊 Using static VBD rankings")
                
                return True
                    
            except Exception as e:
                print(f"⚠️  Failed to load {source}: {e}")
                continue
        
        print("❌ ERROR: No player database could be loaded!")
        print("Please ensure one of these files exists:")
        for source in fallback_sources:
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
    
    def update_dynamic_rankings(self) -> None:
        """Update Dynamic VBD rankings based on current draft state."""
        if not self.dynamic_vbd_enabled or self.dynamic_vbd_transformer is None:
            return
        
        try:
            # Import Dynamic VBD components using safe import
            dynamic_vbd_module = _import_dynamic_vbd()
            
            # Get available players
            drafted_names = [p['player_name'] for p in self.picks]
            available_players = self.players_df[~self.players_df[self.player_col].isin(drafted_names)].copy()
            
            if available_players.empty:
                return
            
            # Create draft state
            draft_state = dynamic_vbd_module.create_draft_state(
                current_pick=self.current_pick,
                drafted_players=drafted_names
            )
            
            # Calculate position probabilities based on remaining players
            position_probs = self._calculate_position_probabilities(available_players)
            
            # Validate position probabilities to prevent division by zero
            if not position_probs:
                print("⚠️  No valid position probabilities calculated, skipping Dynamic VBD update")
                return
            
            # Create probability forecast
            horizon_picks = min(max(3, (TOTAL_PICKS - self.current_pick) // 20), 7)
            forecast = dynamic_vbd_module.create_probability_forecast(
                horizon_picks=horizon_picks,
                position_probs=position_probs
            )
            
            # Update rankings with Dynamic VBD
            updated_rankings = self.dynamic_vbd_transformer.transform(
                available_players, forecast, draft_state
            )
            
            # Safely update the main DataFrame using merge instead of direct assignment
            self._safely_update_rankings(available_players, updated_rankings)
            
        except Exception as e:
            print(f"⚠️  Error updating Dynamic VBD rankings: {e}")
            print("    Continuing with static rankings")
    
    def _safely_update_rankings(self, original_df: pd.DataFrame, updated_df: pd.DataFrame) -> None:
        """Safely update rankings using merge instead of direct assignment."""
        try:
            # Ensure we have a common merge key
            if self.player_col not in original_df.columns or self.player_col not in updated_df.columns:
                raise ValueError(f"Player column '{self.player_col}' missing in DataFrames")
            
            # Get VBD columns to update
            vbd_columns = [col for col in updated_df.columns if 'VBD' in col.upper()]
            if not vbd_columns:
                print("⚠️  No VBD columns found in updated rankings")
                return
            
            # Create update dictionary for each player
            update_cols = [self.player_col] + vbd_columns
            update_data = updated_df[update_cols].copy()
            
            # Update main DataFrame by merging
            for _, row in update_data.iterrows():
                player_name = row[self.player_col]
                mask = self.players_df[self.player_col] == player_name
                
                # Only update if player exists in main DataFrame
                if mask.any():
                    for vbd_col in vbd_columns:
                        if vbd_col in row:
                            self.players_df.loc[mask, vbd_col] = row[vbd_col]
                            
        except Exception as e:
            print(f"⚠️  Error safely updating rankings: {e}")
            print("    Rankings may not be fully updated")
    
    def _calculate_position_probabilities(self, available_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate position draft probabilities based on available players and draft stage."""
        # Input validation
        if available_df is None or available_df.empty:
            return {}
        
        if self.position_col is None or self.position_col not in available_df.columns:
            print("⚠️  Position column not available for probability calculation")
            return {}
        
        position_counts = available_df[self.position_col].value_counts()
        total_available = len(available_df)
        
        # Critical validation: prevent division by zero
        if total_available <= 0:
            print("⚠️  No available players for probability calculation")
            return {}
        
        # Base weights for typical draft behavior
        base_weights = {
            'QB': 0.8,   'RB': 1.3,   'WR': 1.2,   'TE': 0.9,
            'DEF': 0.4,  'DST': 0.4,  'K': 0.3
        }
        
        # Adjust weights based on draft stage
        try:
            draft_progress = self.current_pick / TOTAL_PICKS
            stage_multiplier = self._get_draft_stage_multiplier(draft_progress)
        except (ZeroDivisionError, TypeError) as e:
            print(f"⚠️  Error calculating draft progress: {e}")
            stage_multiplier = {pos: 1.0 for pos in base_weights.keys()}
        
        position_probs = {}
        for pos, count in position_counts.items():
            if count > 0 and pos:  # Ensure position is valid
                try:
                    base_prob = count / total_available
                    weight = base_weights.get(pos, 1.0) * stage_multiplier.get(pos, 1.0)
                    position_probs[pos] = base_prob * weight
                except (ZeroDivisionError, TypeError) as e:
                    print(f"⚠️  Error calculating probability for {pos}: {e}")
                    continue
        
        # Normalize to sum to 1 with validation
        total_prob = sum(position_probs.values())
        if total_prob > 0:
            try:
                position_probs = {k: v/total_prob for k, v in position_probs.items()}
            except ZeroDivisionError:
                print("⚠️  Error normalizing probabilities")
                return {}
        else:
            print("⚠️  No valid probabilities calculated")
            return {}
        
        return position_probs
    
    def _get_draft_stage_multiplier(self, draft_progress: float) -> Dict[str, float]:
        """Get position weight multipliers based on draft stage."""
        # Input validation
        if not isinstance(draft_progress, (int, float)) or draft_progress < 0:
            print(f"⚠️  Invalid draft progress: {draft_progress}")
            # Return neutral multipliers as fallback
            return {'QB': 1.0, 'RB': 1.0, 'WR': 1.0, 'TE': 1.0, 'DEF': 1.0, 'DST': 1.0, 'K': 1.0}
        
        if draft_progress < self.draft_stage_early_threshold:  # Early draft
            return {'QB': 1.2, 'RB': 1.3, 'WR': 1.2, 'TE': 1.1, 'DEF': 0.2, 'DST': 0.2, 'K': 0.1}
        elif draft_progress < self.draft_stage_late_threshold:  # Mid draft
            return {'QB': 1.0, 'RB': 1.1, 'WR': 1.1, 'TE': 1.0, 'DEF': 0.6, 'DST': 0.6, 'K': 0.3}
        else:  # Late draft
            return {'QB': 0.8, 'RB': 0.9, 'WR': 0.9, 'TE': 0.8, 'DEF': 1.5, 'DST': 1.5, 'K': 1.8}
    
    def show_dynamic_rankings(self, top_n: int = 10) -> None:
        """Show top available players by Dynamic VBD."""
        if not self.dynamic_vbd_enabled or self.players_df is None:
            print("📊 Dynamic VBD not available - showing static rankings")
            self._show_static_rankings(top_n)
            return
        
        # Get available players
        drafted_names = [p['player_name'] for p in self.picks]
        available_players = self.players_df[~self.players_df[self.player_col].isin(drafted_names)].copy()
        
        if available_players.empty:
            print("⚠️  No players remaining!")
            return
        
        # Find best VBD column to sort by
        vbd_col = None
        for col in ['VBD_BLENDED', 'VBD_BEER', 'VBD_VORP', 'VBD_VOLS', 'FANTASY_PTS']:
            if col in available_players.columns:
                vbd_col = col
                break
        
        if vbd_col is None:
            print("⚠️  No VBD data available")
            return
        
        # Sort and display top players
        top_players = available_players.nlargest(top_n, vbd_col)
        
        print(f"\n🏆 Top {top_n} Available Players (by {vbd_col}):")
        print("=" * 60)
        
        for i, (_, player) in enumerate(top_players.iterrows(), 1):
            name = player[self.player_col]
            pos = player[self.position_col] if self.position_col else 'N/A'
            team = player[self.team_col] if self.team_col else 'N/A'
            value = player[vbd_col]
            
            print(f"  {i:2d}. {name:<25} ({pos:3s}, {team:4s}) - {vbd_col}: {value:6.1f}")
    
    def _show_static_rankings(self, top_n: int = 10) -> None:
        """Show top available players by basic ranking."""
        drafted_names = [p['player_name'] for p in self.picks]
        available_players = self.players_df[~self.players_df[self.player_col].isin(drafted_names)].copy()
        
        if available_players.empty:
            print("⚠️  No players remaining!")
            return
        
        # Just show first N available players
        top_players = available_players.head(top_n)
        
        print(f"\n📋 Top {top_n} Available Players:")
        print("=" * 50)
        
        for i, (_, player) in enumerate(top_players.iterrows(), 1):
            name = player[self.player_col]
            pos = player[self.position_col] if self.position_col else 'N/A'
            team = player[self.team_col] if self.team_col else 'N/A'
            print(f"  {i:2d}. {name:<25} ({pos:3s}, {team:4s})")
    
    def show_vbd_impact(self) -> None:
        """Show VBD impact analysis after recent picks."""
        if not self.dynamic_vbd_enabled or not self.picks:
            return
        
        print(f"\n📈 VBD Impact Analysis:")
        
        # Show recent picks by position
        recent_picks = self.picks[-5:] if len(self.picks) >= 5 else self.picks
        pos_counts = {}
        for pick in recent_picks:
            pos = pick.get('position', 'Unknown')
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        
        print(f"   Recent picks by position: {dict(pos_counts)}")
        
        # Show remaining position scarcity
        if self.players_df is not None and self.position_col:
            drafted_names = [p['player_name'] for p in self.picks]
            available = self.players_df[~self.players_df[self.player_col].isin(drafted_names)]
            remaining_counts = available[self.position_col].value_counts()
            print(f"   Remaining by position: {dict(remaining_counts)}")
            
            # Identify position runs
            if recent_picks and len(set(p.get('position') for p in recent_picks[-3:])) == 1:
                run_pos = recent_picks[-1].get('position')
                print(f"   🔥 {run_pos} RUN DETECTED - remaining {run_pos}s have increased value!")
    
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
        
        # Multiple matches - show selection with VBD values if available
        print(f"\n🔍 Multiple players found:")
        display_matches = matches.head(10)
        
        # Check for VBD column to display
        vbd_col = None
        for col in ['VBD_BLENDED', 'VBD_BEER', 'VBD_VORP', 'VBD_VOLS', 'FANTASY_PTS']:
            if col in display_matches.columns:
                vbd_col = col
                break
        
        for idx, (_, row) in enumerate(display_matches.iterrows(), 1):
            player_name = row[self.player_col]
            pos = row[self.position_col] if self.position_col else 'N/A'
            team = row[self.team_col] if self.team_col else 'N/A'
            
            if vbd_col:
                vbd_value = row[vbd_col]
                print(f"  {idx}. {player_name} ({pos}, {team}) - {vbd_col}: {vbd_value:.1f}")
            else:
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
    
    def undo_last_pick(self) -> bool:
        """Remove the last pick."""
        if not self.picks:
            return False
        
        removed = self.picks.pop()
        self.current_pick -= 1
        
        player_name = removed.get('player_name', 'Unknown')
        pick_num = removed.get('overall_pick', 'Unknown')
        print(f"↩️  Removed: {player_name} (pick #{pick_num})")
        
        # Update Dynamic VBD rankings if enabled
        if self.dynamic_vbd_enabled:
            print("🔄 Updating Dynamic VBD rankings...")
            self.update_dynamic_rankings()
        
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
        
        # Update Dynamic VBD rankings if enabled
        if self.dynamic_vbd_enabled:
            print("🔄 Updating Dynamic VBD rankings...")
            self.update_dynamic_rankings()
            self.show_vbd_impact()
        
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
            
            # Show Dynamic VBD status
            if self.dynamic_vbd_enabled:
                print(f"  🚀 Dynamic VBD: ENABLED")
                if self.dynamic_vbd_transformer:
                    cache_stats = self.dynamic_vbd_transformer.get_cache_stats()
                    print(f"  📊 Cache size: {cache_stats['cache_size']}")
            else:
                print(f"  📊 Dynamic VBD: DISABLED")
    
    def run_interactive(self) -> None:
        """Run the interactive draft session."""
        print("\n🏈 BACKUP DRAFT TRACKER")
        print("=" * 40)
        print("Emergency draft tracking when ESPN API fails")
        if self.dynamic_vbd_enabled:
            print("Enhanced with Dynamic VBD real-time adjustments")
            print("\nCommands: UNDO, STATUS, RANKINGS, QUIT")
        else:
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
                
            elif player_input.upper() == 'RANKINGS':
                self.show_dynamic_rankings(10)
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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Emergency Backup Draft Tracker with Dynamic VBD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backup_draft.py                 # Use config setting
  python backup_draft.py --dynamic-vbd   # Force enable Dynamic VBD
  python backup_draft.py --no-dynamic-vbd # Force disable Dynamic VBD
        """
    )
    
    parser.add_argument(
        '--dynamic-vbd', 
        action='store_true',
        help='Force enable Dynamic VBD (overrides config)'
    )
    
    parser.add_argument(
        '--no-dynamic-vbd', 
        action='store_true',
        help='Force disable Dynamic VBD (overrides config)'
    )
    
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
        
        # Initialize tracker
        tracker = BackupDraftTracker(force_dynamic_vbd=force_dynamic_vbd)
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