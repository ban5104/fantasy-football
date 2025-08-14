"""
Fantasy football scoring utilities
"""
import pandas as pd
import numpy as np
import yaml
import logging

def load_league_config(config_path="config/league-config.yaml"):
    """Load league configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Failed to load config: {str(e)}")
        return {}

def prepare_stats_columns(df):
    """
    Convert stat columns to numeric and fill NaN values with 0
    
    Args:
        df (pd.DataFrame): Player projections dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe with numeric stats
    """
    # Define stat columns that should be numeric
    stat_columns = [
        'PASSING_ATT', 'PASSING_CMP', 'PASSING_YDS', 'PASSING_TD', 'PASSING_INT',
        'RUSHING_ATT', 'RUSHING_YDS', 'RUSHING_TD', 'RECEIVING_REC', 'RECEIVING_YDS', 
        'RECEIVING_TD', 'FUMBLES', 'KICKING_FG', 'KICKING_PAT', 'DST_SACK', 
        'DST_INT', 'DST_FR', 'DST_TD', 'DST_SAFETY', 'DST_PA', 'DST_YA'
    ]
    
    # Convert to numeric and fill NaN with 0
    for col in stat_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

def calculate_fantasy_points_vectorized(df, scoring_config):
    """
    Calculate fantasy points using vectorized operations
    
    Args:
        df (pd.DataFrame): Player projections with numeric stats
        scoring_config (dict): League scoring configuration
        
    Returns:
        pd.DataFrame: Dataframe with FANTASY_PTS column added
    """
    df = df.copy()
    df['FANTASY_PTS'] = 0.0
    
    # Passing scoring
    passing = scoring_config.get('passing', {})
    if 'PASSING_YDS' in df.columns:
        df['FANTASY_PTS'] += df['PASSING_YDS'] * passing.get('yards', 0)
    if 'PASSING_TD' in df.columns:
        df['FANTASY_PTS'] += df['PASSING_TD'] * passing.get('td', 0)
    if 'PASSING_INT' in df.columns:
        df['FANTASY_PTS'] += df['PASSING_INT'] * passing.get('int', 0)
    
    # Rushing scoring
    rushing = scoring_config.get('rushing', {})
    if 'RUSHING_YDS' in df.columns:
        df['FANTASY_PTS'] += df['RUSHING_YDS'] * rushing.get('yards', 0)
    if 'RUSHING_TD' in df.columns:
        df['FANTASY_PTS'] += df['RUSHING_TD'] * rushing.get('td', 0)
    
    # Receiving scoring
    receiving = scoring_config.get('receiving', {})
    if 'RECEIVING_REC' in df.columns:
        df['FANTASY_PTS'] += df['RECEIVING_REC'] * receiving.get('rec', 0)
    if 'RECEIVING_YDS' in df.columns:
        df['FANTASY_PTS'] += df['RECEIVING_YDS'] * receiving.get('yards', 0)
    if 'RECEIVING_TD' in df.columns:
        df['FANTASY_PTS'] += df['RECEIVING_TD'] * receiving.get('td', 0)
    
    # Fumbles
    if 'FUMBLES' in df.columns:
        fumble_pts = scoring_config.get('fumbles', {}).get('lost', 0)
        df['FANTASY_PTS'] += df['FUMBLES'] * fumble_pts
    
    # Kicking scoring
    kicking = scoring_config.get('kicking', {})
    if 'KICKING_FG' in df.columns:
        df['FANTASY_PTS'] += df['KICKING_FG'] * kicking.get('fg', 0)
    if 'KICKING_PAT' in df.columns:
        df['FANTASY_PTS'] += df['KICKING_PAT'] * kicking.get('pat', 0)
    
    # Defense scoring
    defense = scoring_config.get('defense', {})
    if 'DST_SACK' in df.columns:
        df['FANTASY_PTS'] += df['DST_SACK'] * defense.get('sack', 0)
    if 'DST_INT' in df.columns:
        df['FANTASY_PTS'] += df['DST_INT'] * defense.get('int', 0)
    if 'DST_FR' in df.columns:
        df['FANTASY_PTS'] += df['DST_FR'] * defense.get('fumble_recovery', 0)
    if 'DST_TD' in df.columns:
        df['FANTASY_PTS'] += df['DST_TD'] * defense.get('td', 0)
    if 'DST_SAFETY' in df.columns:
        df['FANTASY_PTS'] += df['DST_SAFETY'] * defense.get('safety', 0)
    
    # Round to 2 decimal places
    df['FANTASY_PTS'] = df['FANTASY_PTS'].round(2)
    
    return df

def rank_players_by_position(df):
    """
    Add position-specific ranks to dataframe
    
    Args:
        df (pd.DataFrame): Dataframe with FANTASY_PTS column
        
    Returns:
        pd.DataFrame: Dataframe with POSITION_RANK column added
    """
    df = df.copy()
    df['POSITION_RANK'] = df.groupby('Position')['FANTASY_PTS'].rank(
        method='dense', ascending=False
    ).astype(int)
    
    return df