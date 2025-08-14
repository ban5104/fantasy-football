"""
Value Based Drafting (VBD) calculations
"""
import pandas as pd
import logging

def calculate_position_baselines(df, config):
    """
    Calculate baseline ranks for each VBD method by position
    
    Args:
        df (pd.DataFrame): Player projections with fantasy points
        config (dict): League configuration
        
    Returns:
        dict: Baseline ranks by position and method
    """
    teams = config.get('basic_settings', {}).get('teams', 12)
    roster_slots = config.get('roster', {}).get('roster_slots', {})
    
    baselines = {}
    
    for position in df['Position'].unique():
        # Get starter requirements for this position
        starters = roster_slots.get(position, 1)
        
        # Count available players at this position
        pos_players = len(df[df['Position'] == position])
        
        # Calculate baseline ranks (0-indexed)
        baselines[position] = {
            'VOLS': min(teams * starters, pos_players) - 1,
            'VORP': min(teams * (starters + 1), pos_players) - 1, 
            'BEER': min(int(teams * (starters + 0.5)), pos_players) - 1
        }
        
        # Ensure baselines are not negative
        for method in baselines[position]:
            baselines[position][method] = max(0, baselines[position][method])
    
    return baselines

def calculate_vbd_for_position(pos_df, baselines):
    """
    Calculate VBD for a single position
    
    Args:
        pos_df (pd.DataFrame): Players for single position, sorted by fantasy points
        baselines (dict): Baseline ranks for this position
        
    Returns:
        pd.DataFrame: Position dataframe with VBD columns added
    """
    pos_df = pos_df.copy()
    
    for method, baseline_idx in baselines.items():
        if baseline_idx < len(pos_df):
            baseline_points = pos_df.iloc[baseline_idx]['FANTASY_PTS']
        else:
            baseline_points = 0
            
        pos_df[f'VBD_{method}'] = pos_df['FANTASY_PTS'] - baseline_points
    
    return pos_df

def calculate_all_vbd_methods(df, config):
    """
    Calculate VBD using all standard methods (VOLS, VORP, BEER, Blended)
    
    Args:
        df (pd.DataFrame): Player projections with fantasy points
        config (dict): League configuration
        
    Returns:
        pd.DataFrame: Dataframe with all VBD columns added
    """
    df = df.copy()
    
    # Calculate baselines for each position
    baselines = calculate_position_baselines(df, config)
    
    # Initialize VBD columns
    for method in ['VOLS', 'VORP', 'BEER']:
        df[f'VBD_{method}'] = 0.0
    
    # Calculate VBD for each position
    position_dfs = []
    
    for position in df['Position'].unique():
        pos_df = df[df['Position'] == position].copy()
        pos_df = pos_df.sort_values('FANTASY_PTS', ascending=False).reset_index(drop=True)
        
        pos_baselines = baselines.get(position, {})
        pos_df = calculate_vbd_for_position(pos_df, pos_baselines)
        
        position_dfs.append(pos_df)
        
        # Log baseline information
        logging.info(f"{position} baselines - VOLS: {pos_baselines.get('VOLS', 0)+1}, "
                    f"VORP: {pos_baselines.get('VORP', 0)+1}, "
                    f"BEER: {pos_baselines.get('BEER', 0)+1}")
    
    # Combine all positions
    df_with_vbd = pd.concat(position_dfs, ignore_index=True)
    
    # Calculate blended VBD (weighted average)
    df_with_vbd['VBD_BLENDED'] = (
        0.5 * df_with_vbd['VBD_BEER'] + 
        0.25 * df_with_vbd['VBD_VORP'] + 
        0.25 * df_with_vbd['VBD_VOLS']
    ).round(2)
    
    # Round VBD columns
    for col in ['VBD_VOLS', 'VBD_VORP', 'VBD_BEER']:
        df_with_vbd[col] = df_with_vbd[col].round(2)
    
    return df_with_vbd

def get_top_players_by_vbd(df, method='VBD_BLENDED', top_n=300):
    """
    Get top N players sorted by specified VBD method
    
    Args:
        df (pd.DataFrame): Dataframe with VBD calculations
        method (str): VBD method to sort by
        top_n (int): Number of top players to return
        
    Returns:
        pd.DataFrame: Top players sorted by VBD
    """
    return df.nlargest(top_n, method).reset_index(drop=True)

def analyze_vbd_distribution(df, top_n=300):
    """
    Analyze position distribution in top N players by VBD
    
    Args:
        df (pd.DataFrame): Dataframe with VBD calculations
        top_n (int): Number of top players to analyze
        
    Returns:
        pd.DataFrame: Position distribution summary
    """
    top_players = get_top_players_by_vbd(df, top_n=top_n)
    
    distribution = top_players['Position'].value_counts().to_frame('Count')
    distribution['Percentage'] = (distribution['Count'] / top_n * 100).round(1)
    
    return distribution