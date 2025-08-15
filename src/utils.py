"""
Utility functions for fantasy football analysis
"""
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('fantasy_analysis.log')
        ]
    )

def load_latest_projections(data_path="data/raw"):
    """
    Load the most recent projection data
    
    Args:
        data_path (str): Path to data directory
        
    Returns:
        pd.DataFrame: Latest projection data
    """
    try:
        # Look for most recent all_positions file
        files = [f for f in os.listdir(data_path) if f.startswith('projections_all_positions_')]
        if not files:
            raise FileNotFoundError("No projection files found")
        
        latest_file = sorted(files)[-1]
        filepath = os.path.join(data_path, latest_file)
        
        df = pd.read_csv(filepath)
        logging.info(f"Loaded {len(df)} players from {latest_file}")
        
        return df
        
    except Exception as e:
        logging.error(f"Failed to load projections: {str(e)}")
        return pd.DataFrame()

def save_rankings(df, filename_prefix="rankings", output_path="data/output"):
    """
    Save rankings to CSV with timestamp
    
    Args:
        df (pd.DataFrame): Rankings dataframe
        filename_prefix (str): Prefix for filename
        output_path (str): Output directory
        
    Returns:
        str: Saved filename
    """
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f"{filename_prefix}_{timestamp}.csv"
    filepath = os.path.join(output_path, filename)
    
    df.to_csv(filepath, index=False)
    logging.info(f"Saved {len(df)} players to {filename}")
    
    return filename

def clean_player_names(df, player_col='PLAYER'):
    """
    Clean player names and extract team information
    
    Args:
        df (pd.DataFrame): Dataframe with player column
        player_col (str): Name of player column
        
    Returns:
        pd.DataFrame: Dataframe with cleaned names and team info
    """
    df = df.copy()
    
    if player_col in df.columns:
        # Extract team from player name (usually in format "Player Name TEAM")
        df['TEAM'] = df[player_col].str.extract(r'([A-Z]{2,3})$')
        df['PLAYER_CLEAN'] = df[player_col].str.replace(r'\s+[A-Z]{2,3}$', '', regex=True)
    
    return df

def compare_rankings(df, methods=['VBD_VOLS', 'VBD_VORP', 'VBD_BEER', 'VBD_BLENDED'], top_n=50):
    """
    Compare top players across different VBD methods
    
    Args:
        df (pd.DataFrame): Dataframe with VBD calculations
        methods (list): VBD methods to compare
        top_n (int): Number of top players to show
        
    Returns:
        pd.DataFrame: Comparison table
    """
    comparison_data = []
    
    for method in methods:
        top_players = df.nlargest(top_n, method)[['PLAYER', 'POSITION', method]].reset_index(drop=True)
        top_players['Rank'] = range(1, len(top_players) + 1)
        top_players['Method'] = method
        comparison_data.append(top_players)
    
    return pd.concat(comparison_data, ignore_index=True)

def validate_data_quality(df):
    """
    Perform basic data quality checks
    
    Args:
        df (pd.DataFrame): Dataframe to validate
        
    Returns:
        dict: Validation results
    """
    results = {
        'total_players': len(df),
        'missing_fantasy_pts': df['FANTASY_PTS'].isna().sum() if 'FANTASY_PTS' in df.columns else 0,
        'zero_fantasy_pts': (df['FANTASY_PTS'] == 0).sum() if 'FANTASY_PTS' in df.columns else 0,
        'position_counts': df['POSITION'].value_counts().to_dict() if 'POSITION' in df.columns else {},
        'duplicate_players': df.duplicated(subset=['PLAYER']).sum() if 'PLAYER' in df.columns else 0
    }
    
    return results

def create_draft_board(df, positions_per_round=12, rounds=20):
    """
    Create a draft board view with players arranged by round
    
    Args:
        df (pd.DataFrame): Ranked players dataframe
        positions_per_round (int): Number of picks per round
        rounds (int): Number of rounds to show
        
    Returns:
        pd.DataFrame: Draft board format
    """
    total_picks = positions_per_round * rounds
    top_players = df.head(total_picks).copy()
    
    top_players['Round'] = ((top_players.index) // positions_per_round) + 1
    top_players['Pick_in_Round'] = (top_players.index % positions_per_round) + 1
    top_players['Overall_Pick'] = top_players.index + 1
    
    return top_players