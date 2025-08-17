"""
Value Based Drafting (VBD) calculations
"""
import pandas as pd
import logging
from typing import Dict, Optional, Union, Any

def calculate_position_baselines(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """
    Calculate baseline ranks for each VBD method by position
    
    Args:
        df (pd.DataFrame): Player projections with fantasy points
        config (dict): League configuration
        
    Returns:
        dict: Baseline ranks by position and method
        
    Raises:
        ValueError: If DataFrame is empty or missing required columns
        KeyError: If required config sections are missing
    """
    try:
        # Input validation
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        if 'POSITION' not in df.columns:
            raise ValueError("DataFrame missing required 'POSITION' column")
        
        if 'FANTASY_PTS' not in df.columns:
            raise ValueError("DataFrame missing required 'FANTASY_PTS' column")
        
        # Validate config structure
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")
        
        if 'basic_settings' not in config:
            raise KeyError("Config missing required 'basic_settings' section")
        
        if 'roster' not in config:
            raise KeyError("Config missing required 'roster' section")
        
        if 'roster_slots' not in config['roster']:
            raise KeyError("Config missing required 'roster.roster_slots' section")
    
        teams = config.get('basic_settings', {}).get('teams', 12)
        roster_slots = config.get('roster', {}).get('roster_slots', {})
        
        if teams <= 0:
            raise ValueError(f"Invalid team count: {teams}. Must be positive integer.")
        
        logging.info(f"Calculating baselines for {teams} teams with roster slots: {roster_slots}")
    
        # Position mapping: data position -> config position
        position_mapping = {
            'DST': 'DEF',  # Defense/Special Teams
            'DEF': 'DEF',
            'QB': 'QB',
            'RB': 'RB', 
            'WR': 'WR',
            'TE': 'TE',
            'K': 'K'
        }
        
        baselines = {}
        positions_in_data = df['POSITION'].unique()
        
        # Validate that all positions have mappings
        unmapped_positions = [pos for pos in positions_in_data if pos not in position_mapping]
        if unmapped_positions:
            logging.warning(f"Unmapped positions found: {unmapped_positions}. Using default mapping.")
        
        for position in positions_in_data:
            try:
                # Map position name to config name
                config_position = position_mapping.get(position, position)
                
                # Get starter requirements for this position
                if config_position not in roster_slots:
                    logging.warning(f"Position {config_position} not found in roster_slots. Using default of 1.")
                    starters = 1
                else:
                    starters = roster_slots[config_position]
                
                if not isinstance(starters, int) or starters <= 0:
                    logging.warning(f"Invalid starter count for {config_position}: {starters}. Using default of 1.")
                    starters = 1
        
                # Count available players at this position
                pos_players = len(df[df['POSITION'] == position])
                
                if pos_players == 0:
                    logging.warning(f"No players found for position {position}")
                    baselines[position] = {'VOLS': 0, 'VORP': 0, 'BEER': 0}
                    continue
                
                # Calculate baseline ranks (0-indexed)
                vols_baseline = min(teams * starters, pos_players) - 1
                vorp_baseline = min(teams * (starters + 1), pos_players) - 1
                beer_baseline = min(int(teams * (starters + 0.5)), pos_players) - 1
                
                baselines[position] = {
                    'VOLS': max(0, min(vols_baseline, pos_players - 1)),
                    'VORP': max(0, min(vorp_baseline, pos_players - 1)),
                    'BEER': max(0, min(beer_baseline, pos_players - 1))
                }
                
                logging.debug(f"{position}: {pos_players} players, baselines = {baselines[position]}")
                
            except Exception as e:
                logging.error(f"Error calculating baselines for position {position}: {str(e)}")
                baselines[position] = {'VOLS': 0, 'VORP': 0, 'BEER': 0}
    
        return baselines
    
    except Exception as e:
        logging.error(f"Error in calculate_position_baselines: {str(e)}")
        raise

def calculate_vbd_for_position(pos_df: pd.DataFrame, baselines: Dict[str, Union[int, float]]) -> pd.DataFrame:
    """
    Calculate VBD for a single position
    
    Args:
        pos_df (pd.DataFrame): Players for single position, sorted by fantasy points
        baselines (dict): Baseline ranks for this position
        
    Returns:
        pd.DataFrame: Position dataframe with VBD columns added
        
    Raises:
        ValueError: If input DataFrame is empty or missing required columns
    """
    try:
        # Input validation
        if pos_df.empty:
            logging.warning("Position DataFrame is empty")
            return pos_df.copy()
        
        if 'FANTASY_PTS' not in pos_df.columns:
            raise ValueError("Position DataFrame missing required 'FANTASY_PTS' column")
        
        pos_df = pos_df.copy()
        
        for method, baseline_value in baselines.items():
            try:
                # Check if this is a dynamic VBD override (dict with 'points' key)
                if isinstance(baseline_value, dict) and 'points' in baseline_value:
                    # Dynamic VBD override - use specified baseline points
                    baseline_points = baseline_value['points']
                    logging.debug(f"Using dynamic VBD override for {method}: {baseline_points:.2f} points")
                else:
                    # Traditional rank-based baseline
                    baseline_idx = int(baseline_value)
                    if baseline_idx < len(pos_df) and baseline_idx >= 0:
                        baseline_points = pos_df.iloc[baseline_idx]['FANTASY_PTS']
                        if pd.isna(baseline_points):
                            logging.warning(f"Baseline player has NaN fantasy points for method {method}")
                            baseline_points = 0
                    else:
                        # Handle insufficient data case - use last player's points as baseline
                        if len(pos_df) > 0:
                            baseline_points = pos_df.iloc[-1]['FANTASY_PTS']
                            logging.warning(f"Insufficient players for {method} baseline (index {baseline_idx}), using last player's points: {baseline_points}")
                        else:
                            baseline_points = 0
                            logging.warning(f"No players available for {method} baseline calculation")
                
                # Calculate VBD, handling potential NaN values
                vbd_values = pos_df['FANTASY_PTS'] - baseline_points
                pos_df[f'VBD_{method}'] = vbd_values.fillna(0)
                
            except Exception as e:
                logging.error(f"Error calculating VBD for method {method}: {str(e)}")
                pos_df[f'VBD_{method}'] = 0
        
        return pos_df
    
    except Exception as e:
        logging.error(f"Error in calculate_vbd_for_position: {str(e)}")
        raise

def _apply_baseline_overrides(baselines: Dict[str, Dict[str, int]], 
                             baseline_overrides: Dict[str, Dict[str, float]], 
                             df: pd.DataFrame) -> Dict[str, Dict[str, Union[int, float]]]:
    """
    Apply baseline overrides from dynamic VBD to calculated baselines
    
    Args:
        baselines: Original baseline ranks by position and method
        baseline_overrides: Override values from dynamic VBD (position -> method -> points)
        df: Player projections DataFrame for context
        
    Returns:
        Updated baselines with overrides applied
    """
    updated_baselines = baselines.copy()
    
    for position, method_overrides in baseline_overrides.items():
        if position in updated_baselines:
            for method, override_points in method_overrides.items():
                if method in updated_baselines[position]:
                    # Store as explicit override dict with points key
                    updated_baselines[position][method] = {'points': float(override_points), 'override': True}
                    logging.debug(f"Applied baseline override: {position} {method} -> {override_points:.2f} points")
    
    return updated_baselines


def calculate_all_vbd_methods(df: pd.DataFrame, config: Dict[str, Any], 
                             baseline_overrides: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
    """
    Calculate VBD using all standard methods (VOLS, VORP, BEER, Blended)
    
    Args:
        df (pd.DataFrame): Player projections with fantasy points
        config (dict): League configuration
        baseline_overrides (dict, optional): Dynamic VBD baseline overrides
                                           Format: {position: {method: baseline_points}}
        
    Returns:
        pd.DataFrame: Dataframe with all VBD columns added
        
    Raises:
        ValueError: If input DataFrame is empty or missing required columns
        KeyError: If required config sections are missing
    """
    try:
        # Input validation
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")
        
        df = df.copy()
        
        # Get VBD configuration with defaults
        vbd_config = config.get('vbd', {})
        blend_weights = vbd_config.get('blend_weights', {
            'BEER': 0.5,
            'VORP': 0.25,
            'VOLS': 0.25
        })
        
        # Validate blend weights
        if not all(isinstance(w, (int, float)) for w in blend_weights.values()):
            logging.warning("Invalid blend weights in config, using defaults")
            blend_weights = {'BEER': 0.5, 'VORP': 0.25, 'VOLS': 0.25}
        
        # Normalize weights to sum to 1
        weight_sum = sum(blend_weights.values())
        if weight_sum == 0:
            raise ValueError("Blend weights cannot sum to zero")
        
        blend_weights = {k: v/weight_sum for k, v in blend_weights.items()}
        
        logging.info(f"Using blend weights: {blend_weights}")
        
        # Calculate baselines for each position
        baselines = calculate_position_baselines(df, config)
        
        # Apply baseline overrides if provided
        if baseline_overrides:
            baselines = _apply_baseline_overrides(baselines, baseline_overrides, df)
        
        # Initialize VBD columns
        for method in ['VOLS', 'VORP', 'BEER']:
            df[f'VBD_{method}'] = 0.0
        
        # Check if we have any positions to process
        positions = df['POSITION'].unique()
        if len(positions) == 0:
            logging.warning("No positions found in DataFrame")
            return df
        
        # Calculate VBD for each position
        position_dfs = []
        successful_positions = 0
        
        for position in positions:
            try:
                pos_df = df[df['POSITION'] == position].copy()
                
                if pos_df.empty:
                    logging.warning(f"No players found for position {position}")
                    continue
                
                # Check for minimum players required for meaningful VBD
                min_players = config.get('vbd', {}).get('min_players_per_position', 3)
                if len(pos_df) < min_players:
                    logging.warning(f"Position {position} has only {len(pos_df)} players, minimum {min_players} recommended")
                
                pos_df = pos_df.sort_values('FANTASY_PTS', ascending=False).reset_index(drop=True)
                
                pos_baselines = baselines.get(position, {'VOLS': 0, 'VORP': 0, 'BEER': 0})
                pos_df = calculate_vbd_for_position(pos_df, pos_baselines)
                
                position_dfs.append(pos_df)
                successful_positions += 1
                
                # Log baseline information (handle both rank and override formats)
                def format_baseline(baseline_val):
                    if isinstance(baseline_val, dict) and 'points' in baseline_val:
                        return f"{baseline_val['points']:.1f}pts"
                    else:
                        return f"{int(baseline_val)+1}" if baseline_val is not None else "0"
                
                logging.info(f"{position} baselines - VOLS: {format_baseline(pos_baselines.get('VOLS'))}, "
                            f"VORP: {format_baseline(pos_baselines.get('VORP'))}, "
                            f"BEER: {format_baseline(pos_baselines.get('BEER'))} (players: {len(pos_df)})")
                            
            except Exception as e:
                logging.error(f"Error processing position {position}: {str(e)}")
                # Continue with other positions rather than failing completely
                continue
        
        if successful_positions == 0:
            raise ValueError("No positions could be processed successfully")
        
        # Combine all positions
        df_with_vbd = pd.concat(position_dfs, ignore_index=True)
        
        # Calculate blended VBD (weighted average)
        try:
            df_with_vbd['VBD_BLENDED'] = (
                blend_weights.get('BEER', 0.5) * df_with_vbd['VBD_BEER'] + 
                blend_weights.get('VORP', 0.25) * df_with_vbd['VBD_VORP'] + 
                blend_weights.get('VOLS', 0.25) * df_with_vbd['VBD_VOLS']
            ).round(2)
        except Exception as e:
            logging.error(f"Error calculating blended VBD: {str(e)}")
            df_with_vbd['VBD_BLENDED'] = 0
        
        # Round VBD columns and handle any NaN values
        for col in ['VBD_VOLS', 'VBD_VORP', 'VBD_BEER']:
            if col in df_with_vbd.columns:
                df_with_vbd[col] = df_with_vbd[col].fillna(0).round(2)
        
        logging.info(f"VBD calculation completed for {successful_positions} positions, {len(df_with_vbd)} total players")
        
        return df_with_vbd
    
    except Exception as e:
        logging.error(f"Error in calculate_all_vbd_methods: {str(e)}")
        raise

def get_top_players_by_vbd(df: pd.DataFrame, method: str = 'VBD_BLENDED', top_n: int = 300) -> pd.DataFrame:
    """
    Get top N players sorted by specified VBD method
    
    Args:
        df (pd.DataFrame): Dataframe with VBD calculations
        method (str): VBD method to sort by
        top_n (int): Number of top players to return
        
    Returns:
        pd.DataFrame: Top players sorted by VBD
        
    Raises:
        ValueError: If DataFrame is empty or method column doesn't exist
    """
    try:
        if df.empty:
            logging.warning("Input DataFrame is empty")
            return pd.DataFrame()
        
        if method not in df.columns:
            raise ValueError(f"Method column '{method}' not found in DataFrame. Available columns: {list(df.columns)}")
        
        if top_n <= 0:
            raise ValueError(f"top_n must be positive, got {top_n}")
        
        # Handle NaN values in the method column
        df_clean = df.dropna(subset=[method])
        
        if df_clean.empty:
            logging.warning(f"No valid data found for method {method}")
            return pd.DataFrame()
        
        if len(df_clean) < top_n:
            logging.info(f"Only {len(df_clean)} players available, requested {top_n}")
        
        result = df_clean.nlargest(top_n, method).reset_index(drop=True)
        logging.info(f"Returning top {len(result)} players by {method}")
        
        return result
    
    except Exception as e:
        logging.error(f"Error in get_top_players_by_vbd: {str(e)}")
        raise

def analyze_vbd_distribution(df: pd.DataFrame, top_n: int = 300) -> pd.DataFrame:
    """
    Analyze position distribution in top N players by VBD
    
    Args:
        df (pd.DataFrame): Dataframe with VBD calculations
        top_n (int): Number of top players to analyze
        
    Returns:
        pd.DataFrame: Position distribution summary
        
    Raises:
        ValueError: If DataFrame is empty or missing required columns
    """
    try:
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        if 'POSITION' not in df.columns:
            raise ValueError("DataFrame missing required 'POSITION' column")
        
        top_players = get_top_players_by_vbd(df, top_n=top_n)
        
        if top_players.empty:
            logging.warning("No top players found for analysis")
            return pd.DataFrame(columns=['Count', 'Percentage'])
        
        distribution = top_players['POSITION'].value_counts().to_frame('Count')
        actual_count = len(top_players)
        distribution['Percentage'] = (distribution['Count'] / actual_count * 100).round(1)
        
        logging.info(f"Position distribution analysis completed for {actual_count} players")
        
        return distribution
    
    except Exception as e:
        logging.error(f"Error in analyze_vbd_distribution: {str(e)}")
        raise