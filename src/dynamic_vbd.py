"""
Dynamic Value Based Drafting (VBD) calculations

Simplified Dynamic VBD system that adjusts VBD baselines in real-time
based on draft probability forecasts using BEER method only.
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




class DynamicVBDTransformer:
    """
    Simplified Dynamic VBD engine using BEER method only
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the transformer with configuration
        
        Args:
            config: League configuration dictionary
        """
        self.config = config
        
        # Extract dynamic VBD parameters with defaults
        dynamic_config = config.get('dynamic_vbd', {})
        self.enabled = dynamic_config.get('enabled', False)
        
        params = dynamic_config.get('params', {})
        self.scale = params.get('scale', 3.0)
        self.kappa = params.get('kappa', 5.0)
        
        # Simple cache without size limits
        self._cache = {}
        
        logging.info(f"DynamicVBDTransformer initialized: enabled={self.enabled}")
    
    def transform(self, df: pd.DataFrame, probabilities: ProbabilityForecast, 
                 draft_state: DraftState) -> pd.DataFrame:
        """
        Transform player DataFrame with dynamic VBD baseline adjustments
        
        Args:
            df: Available player projections DataFrame
            probabilities: Position draft probability forecast
            draft_state: Current draft state
            
        Returns:
            DataFrame with adjusted VBD scores
        """
        if not self.enabled:
            return df.copy()
        
        try:
            # Simple cache key
            cache_key = f"{draft_state.current_pick}_{len(draft_state.drafted_players)}"
            
            if cache_key in self._cache:
                baseline_overrides = self._cache[cache_key]
            else:
                baseline_overrides = self._compute_adjustments(df, probabilities)
                self._cache[cache_key] = baseline_overrides
            
            # Apply adjustments using existing VBD system
            try:
                from src.vbd import calculate_all_vbd_methods
            except ImportError:
                from vbd import calculate_all_vbd_methods
            
            df_adjusted = calculate_all_vbd_methods(
                df, 
                self.config, 
                baseline_overrides=baseline_overrides
            )
            
            return df_adjusted
            
        except ImportError as e:
            logging.error(f"Dynamic VBD import failed: {e}")
            logging.warning("VBD module not available, falling back to static VBD calculations")
            print("⚠️  Dynamic VBD module import failed during transformation")
            print("    This may indicate a missing dependency or configuration issue")
            # Fallback to static VBD
            try:
                from src.vbd import calculate_all_vbd_methods
                return calculate_all_vbd_methods(df, self.config)
            except ImportError:
                try:
                    from vbd import calculate_all_vbd_methods
                    return calculate_all_vbd_methods(df, self.config)
                except ImportError:
                    logging.error("Static VBD module also unavailable, returning original DataFrame")
                    print("⚠️  Neither Dynamic nor Static VBD modules available")
                    return df.copy()
        except Exception as e:
            logging.error(f"Dynamic VBD transform failed: {e}")
            logging.warning("Falling back to static VBD calculations")
            print(f"⚠️  Dynamic VBD transformation failed: {e}")
            print("    Falling back to static VBD calculations")
            # Fallback to static VBD
            try:
                from src.vbd import calculate_all_vbd_methods
                return calculate_all_vbd_methods(df, self.config)
            except ImportError:
                try:
                    from vbd import calculate_all_vbd_methods
                    return calculate_all_vbd_methods(df, self.config)
                except Exception as fallback_error:
                    logging.error(f"Static VBD fallback also failed: {fallback_error}")
                    print(f"⚠️  Static VBD fallback also failed: {fallback_error}")
                    print("    Returning original DataFrame without VBD calculations")
                    return df.copy()
            except Exception as fallback_error:
                logging.error(f"Static VBD fallback also failed: {fallback_error}")
                print(f"⚠️  Static VBD fallback also failed: {fallback_error}")
                print("    Returning original DataFrame without VBD calculations")
                return df.copy()
    
    def _compute_adjustments(self, df: pd.DataFrame, probabilities: ProbabilityForecast) -> Dict[str, Dict[str, float]]:
        """
        Compute BEER baseline adjustments for each position
        
        Args:
            df: Available player projections
            probabilities: Position draft probabilities
            
        Returns:
            Dictionary of baseline overrides {position: {method: baseline_points}}
        """
        # Check for empty or invalid DataFrame
        if df.empty or 'POSITION' not in df.columns or 'FANTASY_PTS' not in df.columns:
            return {}
        
        adjustments = {}
        
        for position in df['POSITION'].unique():
            pos_df = df[df['POSITION'] == position].copy()
            if pos_df.empty:
                continue
            
            # Sort by fantasy points for baseline calculations
            pos_df = pos_df.sort_values('FANTASY_PTS', ascending=False).reset_index(drop=True)
            
            # Get position probability
            position_prob = probabilities.position_probs.get(position, 0.0)
            
            # Convert probability to expected picks using horizon
            expected_picks = position_prob * probabilities.horizon_picks
            
            # Calculate adjustment magnitude using sigmoid
            adjustment = self.scale * np.tanh(expected_picks / self.kappa)
            
            # Apply adjustment using BEER method only
            try:
                adjusted_baseline = self._apply_beer_interpolation(pos_df, adjustment)
                # Format as expected by VBD system: {position: {method: baseline_points}}
                adjustments[position] = {'BEER': adjusted_baseline}
                
            except Exception as e:
                logging.warning(f"Error adjusting {position}: {e}")
                continue
        
        return adjustments
    
    
    def _apply_beer_interpolation(self, pos_df: pd.DataFrame, adjustment: float) -> float:
        """
        Apply continuous point interpolation for BEER baseline adjustment
        
        Args:
            pos_df: Position DataFrame sorted by fantasy points
            adjustment: Adjustment magnitude
            
        Returns:
            Adjusted baseline points
        """
        # Get BEER baseline index
        baseline_idx = self._get_beer_baseline_index(pos_df)
        
        if baseline_idx >= len(pos_df) - 1:
            return pos_df.iloc[-1]['FANTASY_PTS']
        
        # Get points at baseline rank and next rank
        pts_at_baseline = pos_df.iloc[baseline_idx]['FANTASY_PTS']
        pts_at_next = pos_df.iloc[baseline_idx + 1]['FANTASY_PTS']
        
        # Handle edge cases
        if pd.isna(pts_at_baseline) or pd.isna(pts_at_next):
            return pts_at_baseline if not pd.isna(pts_at_baseline) else 0.0
        
        # Linear interpolation between ranks
        point_diff = pts_at_baseline - pts_at_next
        
        # Adjustment shifts the effective baseline
        adjusted_points = pts_at_baseline - (adjustment * point_diff)
        
        return float(adjusted_points)
    
    def _get_beer_baseline_index(self, pos_df: pd.DataFrame) -> int:
        """
        Get BEER baseline index based on league configuration
        
        Args:
            pos_df: Position DataFrame
            
        Returns:
            Baseline index (0-indexed)
        """
        if pos_df.empty:
            return 0
        
        position = pos_df.iloc[0]['POSITION']
        teams = self.config.get('basic_settings', {}).get('teams', 14)
        roster_slots = self.config.get('roster', {}).get('roster_slots', {})
        
        # Map position names
        position_mapping = {
            'DST': 'DEF', 'DEF': 'DEF', 'QB': 'QB', 'RB': 'RB', 
            'WR': 'WR', 'TE': 'TE', 'K': 'K'
        }
        config_position = position_mapping.get(position, position)
        starters = roster_slots.get(config_position, 1)
        
        # BEER baseline calculation
        baseline = int(teams * (starters + 0.5))
        
        # Convert to 0-indexed and bound to available players
        baseline_idx = max(0, min(baseline - 1, len(pos_df) - 1))
        return baseline_idx
    
    
    
    def clear_cache(self) -> None:
        """Clear the adjustment cache"""
        self._cache.clear()
        logging.info("Dynamic VBD cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self._cache),
            'cache_keys': list(self._cache.keys())
        }




# Utility functions for external integration
def create_probability_forecast(horizon_picks: int, 
                              position_probs: Dict[str, float]) -> ProbabilityForecast:
    """
    Create ProbabilityForecast object
    
    Args:
        horizon_picks: Number of picks to forecast ahead
        position_probs: Position probabilities dictionary
        
    Returns:
        ProbabilityForecast object
    """
    return ProbabilityForecast(
        horizon_picks=horizon_picks,
        position_probs=position_probs
    )


def create_draft_state(current_pick: int, 
                      drafted_players: List[str]) -> DraftState:
    """
    Create DraftState object
    
    Args:
        current_pick: Current draft pick number
        drafted_players: List of already drafted player names
        
    Returns:
        DraftState object
    """
    return DraftState(
        current_pick=current_pick,
        drafted_players=set(drafted_players)
    )