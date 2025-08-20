"""
Dynamic Replacement Level Module - Simplified

Calculates replacement level thresholds for each position per simulation.
"""

import numpy as np
import pandas as pd
from typing import Dict
from collections import defaultdict


def calculate_replacement_levels(players_df: pd.DataFrame, 
                                player_values: Dict[int, float],
                                n_teams: int = 14) -> Dict[str, float]:
    """
    Calculate replacement level for each position
    
    Args:
        players_df: DataFrame with player data including 'pos' column
        player_values: Dict mapping player_id to projected value
        n_teams: Number of teams in league
        
    Returns:
        Dict mapping position to replacement level value
    """
    # Standard roster requirements for 14-team league
    STARTERS_PER_POS = {
        'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 
        'FLEX': 1, 'K': 1, 'DST': 1
    }
    
    pos_values = defaultdict(list)
    
    # Group values by position
    for player_id, value in player_values.items():
        if player_id not in players_df.index:
            continue
        pos = players_df.loc[player_id, 'pos']
        pos_values[pos].append(value)
    
    replacement = {}
    
    # Calculate replacement for standard positions
    for pos in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
        if pos not in pos_values or not pos_values[pos]:
            replacement[pos] = 0
            continue
            
        sorted_vals = sorted(pos_values[pos], reverse=True)
        # Replacement level = last starter for the position
        rank = n_teams * STARTERS_PER_POS.get(pos, 1)
        
        # Ensure rank is within bounds
        rank_idx = min(rank - 1, len(sorted_vals) - 1)
        rank_idx = max(0, rank_idx)
        replacement[pos] = sorted_vals[rank_idx]
    
    # FLEX replacement from combined RB/WR/TE pool
    flex_pool = (pos_values.get('RB', []) + 
                pos_values.get('WR', []) + 
                pos_values.get('TE', []))
    
    if flex_pool:
        sorted_flex = sorted(flex_pool, reverse=True)
        flex_rank = n_teams * STARTERS_PER_POS['FLEX']
        flex_idx = min(flex_rank - 1, len(sorted_flex) - 1)
        flex_idx = max(0, flex_idx)
        replacement['FLEX'] = sorted_flex[flex_idx]
    else:
        replacement['FLEX'] = 0
    
    # Sanity checks
    for pos, val in replacement.items():
        if np.isnan(val):
            replacement[pos] = 0
    
    return replacement


def calculate_replacement_quantiles(all_replacement_values):
    """Calculate quantiles from multiple simulations for replacement distribution modeling"""
    if not all_replacement_values:
        return {}
        
    quantiles = {}
    
    # Get all positions from first simulation (assuming consistent structure)
    positions = all_replacement_values[0].keys() if all_replacement_values else []
    
    for pos in positions:
        # Extract values for this position across all simulations
        values = []
        for replacement_dict in all_replacement_values:
            if pos in replacement_dict:
                values.append(replacement_dict[pos])
        
        if values:
            quantiles[pos] = {
                'p25': np.percentile(values, 25),
                'p50': np.percentile(values, 50),  # Median
                'p75': np.percentile(values, 75),
                'mean': np.mean(values),
                'std': np.std(values)
            }
        else:
            quantiles[pos] = {
                'p25': 0, 'p50': 0, 'p75': 0, 'mean': 0, 'std': 0
            }
    
    return quantiles