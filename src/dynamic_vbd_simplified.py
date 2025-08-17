"""
Simplified Dynamic Value Based Drafting (VBD) calculations

This is a streamlined version of the Dynamic VBD system that focuses on 
essential functionality with BEER method only and simple position scarcity.
"""

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, Set, List, Optional, Any


@dataclass
class DraftState:
    """Represents the current state of a fantasy draft"""
    current_pick: int
    drafted_players: Set[str]


@dataclass
class ProbabilityForecast:
    """Position draft probability forecast for a given horizon"""
    horizon_picks: int                    # How many picks ahead
    position_probs: Dict[str, float]      # {'RB': 0.60, 'WR': 0.25, ...}


class SimplifiedDynamicVBD:
    """
    Simplified Dynamic VBD engine using BEER method only
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the simplified transformer with configuration"""
        self.config = config
        
        # Extract dynamic VBD parameters with defaults
        dynamic_config = config.get('dynamic_vbd', {})
        self.enabled = dynamic_config.get('enabled', False)
        self.scale = dynamic_config.get('scale', 2.0)
        
        # Simple cache
        self._cache = {}
        
        logging.info(f"SimplifiedDynamicVBD initialized: enabled={self.enabled}")
    
    def transform(self, df: pd.DataFrame, probabilities: ProbabilityForecast, 
                 draft_state: DraftState) -> pd.DataFrame:
        """
        Transform player DataFrame with simplified dynamic VBD adjustments
        """
        if not self.enabled:
            return df.copy()
        
        try:
            # Simple cache key
            cache_key = f"{draft_state.current_pick}_{len(draft_state.drafted_players)}"
            
            if cache_key in self._cache:
                adjustments = self._cache[cache_key]
            else:
                adjustments = self._compute_simple_adjustments(df, probabilities)
                self._cache[cache_key] = adjustments
            
            # Apply adjustments directly to DataFrame
            df_adjusted = df.copy()
            for position, adjustment in adjustments.items():
                mask = df_adjusted['POSITION'] == position
                if mask.any():
                    # Adjust the primary VBD column
                    vbd_col = self._get_primary_vbd_column(df_adjusted)
                    if vbd_col:
                        df_adjusted.loc[mask, vbd_col] += adjustment
            
            return df_adjusted
            
        except Exception as e:
            logging.error(f"Simplified Dynamic VBD transform failed: {e}")
            return df.copy()
    
    def _compute_simple_adjustments(self, df: pd.DataFrame, 
                                   probabilities: ProbabilityForecast) -> Dict[str, float]:
        """
        Compute simple position adjustments based on scarcity and draft stage
        """
        adjustments = {}
        
        for position in df['POSITION'].unique():
            pos_count = len(df[df['POSITION'] == position])
            total_players = len(df)
            
            if pos_count == 0 or total_players == 0:
                continue
            
            # Basic scarcity calculation
            scarcity = 1.0 - (pos_count / total_players)
            
            # Get position probability from forecast
            position_prob = probabilities.position_probs.get(position, 0.1)
            
            # Combine scarcity and probability for adjustment
            adjustment = scarcity * position_prob * self.scale
            
            adjustments[position] = adjustment
        
        return adjustments
    
    def _get_primary_vbd_column(self, df: pd.DataFrame) -> Optional[str]:
        """Get the primary VBD column to adjust"""
        for col in ['VBD_BEER', 'VBD_BLENDED', 'FANTASY_PTS']:
            if col in df.columns:
                return col
        return None
    
    def clear_cache(self) -> None:
        """Clear the adjustment cache"""
        self._cache.clear()
        logging.info("Simplified Dynamic VBD cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self._cache),
            'cache_keys': list(self._cache.keys())
        }


# Utility functions for external integration
def create_probability_forecast(horizon_picks: int, 
                              position_probs: Dict[str, float]) -> ProbabilityForecast:
    """Create ProbabilityForecast object"""
    return ProbabilityForecast(
        horizon_picks=horizon_picks,
        position_probs=position_probs
    )


def create_draft_state(current_pick: int, 
                      drafted_players: List[str]) -> DraftState:
    """Create DraftState object"""
    return DraftState(
        current_pick=current_pick,
        drafted_players=set(drafted_players)
    )