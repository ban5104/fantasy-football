#!/usr/bin/env python3
"""
Fantasy Football Data Processor
Handles CSV loading with flexible column mapping

Run with: uv run python data_processor.py
"""

import pandas as pd
import yaml
import os
from typing import Dict, Any, List

def load_league_config() -> Dict[str, Any]:
    """Load league configuration from YAML file"""
    config_path = "config/league-config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"League config not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def load_player_data() -> pd.DataFrame:
    """Load and process player projection data"""
    
    # Try to load from multiple possible sources
    data_sources = [
        "data/rankings_20250814.csv",
        "draft_cheat_sheet.csv",
        "CSG Fantasy Football Sheet - 2025 v13.01.csv"
    ]
    
    players_df = None
    
    for source in data_sources:
        if os.path.exists(source):
            try:
                players_df = pd.read_csv(source)
                print(f"Loaded player data from: {source}")
                break
            except Exception as e:
                print(f"Error loading {source}: {e}")
                continue
    
    if players_df is None:
        raise FileNotFoundError("No player data files found. Please ensure you have rankings data.")
    
    # Process the data based on which file we loaded
    if "rankings_" in source:
        players_df = process_rankings_data(players_df)
    elif "cheat_sheet" in source:
        players_df = process_cheat_sheet_data(players_df)
    elif "CSG Fantasy" in source:
        players_df = process_csg_data(players_df)
    
    # Ensure required columns exist
    required_columns = ['UNNAMED:_0_LEVEL_0_PLAYER', 'POSITION', 'FANTASY_PTS']
    for col in required_columns:
        if col not in players_df.columns:
            # Try to map from other column names
            players_df = map_column_names(players_df, col)
    
    # Clean and validate data
    players_df = clean_player_data(players_df)
    
    return players_df

def process_rankings_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process data from rankings CSV file"""
    # This file should already be in the correct format
    return df

def process_cheat_sheet_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process data from draft cheat sheet"""
    # Map columns to standard format
    column_mapping = {
        'Player': 'UNNAMED:_0_LEVEL_0_PLAYER',
        'Position': 'POSITION',
        'Custom_VBD': 'FANTASY_PTS'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
    
    return df

def process_csg_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process data from CSG Fantasy Football sheet"""
    # This is likely in a different format, map as needed
    column_mapping = {
        'Player': 'UNNAMED:_0_LEVEL_0_PLAYER',
        'Position': 'POSITION',
        'VBD': 'FANTASY_PTS',
        'ECR Prj': 'FANTASY_PTS'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
    
    return df

def map_column_names(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Try to map alternative column names to required columns"""
    
    column_mappings = {
        'UNNAMED:_0_LEVEL_0_PLAYER': ['Player', 'player', 'Name', 'PlayerName'],
        'POSITION': ['Position', 'position', 'Pos', 'pos'],
        'FANTASY_PTS': ['FANTASY_PTS', 'Fantasy_Pts', 'Proj', 'Projection', 'Points', 'VBD', 'Custom_VBD']
    }
    
    if target_col in column_mappings:
        for alt_name in column_mappings[target_col]:
            if alt_name in df.columns and target_col not in df.columns:
                df[target_col] = df[alt_name]
                break
    
    return df

def clean_player_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate player data"""
    
    # Remove rows with missing essential data
    essential_cols = ['UNNAMED:_0_LEVEL_0_PLAYER', 'POSITION']
    df = df.dropna(subset=essential_cols)
    
    # Fill missing fantasy points with 0
    if 'FANTASY_PTS' in df.columns:
        df['FANTASY_PTS'] = pd.to_numeric(df['FANTASY_PTS'], errors='coerce').fillna(0)
    else:
        df['FANTASY_PTS'] = 0
    
    # Standardize position names
    position_mapping = {
        'DEF': 'DST',
        'D/ST': 'DST',
        'Defense': 'DST'
    }
    
    if 'POSITION' in df.columns:
        df['POSITION'] = df['POSITION'].replace(position_mapping)
    
    # Remove duplicate players (keep highest scoring)
    if 'UNNAMED:_0_LEVEL_0_PLAYER' in df.columns:
        df = df.sort_values('FANTASY_PTS', ascending=False)
        df = df.drop_duplicates(subset=['UNNAMED:_0_LEVEL_0_PLAYER', 'POSITION'], keep='first')
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Add Custom_VBD if not present (use FANTASY_PTS as proxy)
    if 'Custom_VBD' not in df.columns:
        df['Custom_VBD'] = df['FANTASY_PTS']
    
    return df

def get_player_stats_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get summary statistics about the player data"""
    summary = {
        'total_players': len(df),
        'positions': df['POSITION'].value_counts().to_dict() if 'POSITION' in df.columns else {},
        'columns': list(df.columns),
        'fantasy_pts_range': {
            'min': df['FANTASY_PTS'].min() if 'FANTASY_PTS' in df.columns else None,
            'max': df['FANTASY_PTS'].max() if 'FANTASY_PTS' in df.columns else None,
            'mean': df['FANTASY_PTS'].mean() if 'FANTASY_PTS' in df.columns else None
        }
    }
    
    return summary

def validate_data_integrity(df: pd.DataFrame) -> List[str]:
    """Validate data integrity and return list of issues"""
    issues = []
    
    # Check for required columns
    required_cols = ['UNNAMED:_0_LEVEL_0_PLAYER', 'POSITION', 'FANTASY_PTS']
    for col in required_cols:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")
    
    # Check for empty data
    if len(df) == 0:
        issues.append("No player data found")
    
    # Check for valid positions
    valid_positions = {'QB', 'RB', 'WR', 'TE', 'K', 'DST'}
    if 'POSITION' in df.columns:
        invalid_positions = set(df['POSITION'].unique()) - valid_positions
        if invalid_positions:
            issues.append(f"Invalid positions found: {invalid_positions}")
    
    # Check for reasonable fantasy point values
    if 'FANTASY_PTS' in df.columns:
        if df['FANTASY_PTS'].max() > 500:
            issues.append("Unusually high fantasy point values detected")
        if df['FANTASY_PTS'].min() < 0:
            issues.append("Negative fantasy point values detected")
    
    return issues

if __name__ == "__main__":
    # Test the data loading
    try:
        config = load_league_config()
        print("League config loaded successfully")
        print(f"League: {config.get('league_name', 'Unknown')}")
        print(f"Teams: {config.get('basic_settings', {}).get('teams', 'Unknown')}")
        
        players_df = load_player_data()
        print(f"\nPlayer data loaded successfully")
        
        summary = get_player_stats_summary(players_df)
        print(f"Total players: {summary['total_players']}")
        print(f"Positions: {summary['positions']}")
        
        issues = validate_data_integrity(players_df)
        if issues:
            print(f"\nData issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\nData validation passed!")
            
        print(f"\nSample data:")
        print(players_df[['UNNAMED:_0_LEVEL_0_PLAYER', 'POSITION', 'FANTASY_PTS']].head())
        
    except Exception as e:
        print(f"Error: {e}")