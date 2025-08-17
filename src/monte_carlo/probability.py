"""
Probability Model for Fantasy Football Draft
Handles ESPN/ADP rankings and probability calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set
import os


class ProbabilityModel:
    """Calculate draft pick probabilities from ESPN and ADP data"""
    
    def __init__(self, 
                 espn_weight: float = 0.8,
                 adp_weight: float = 0.2,
                 temperature: float = 5.0):
        """
        Initialize probability model
        
        Args:
            espn_weight: Weight for ESPN rankings (default 0.8)
            adp_weight: Weight for ADP rankings (default 0.2)
            temperature: Softmax temperature for probability distribution
        """
        self.espn_weight = espn_weight
        self.adp_weight = adp_weight
        self.temperature = temperature
        self.players_df = None
        
    def load_data(self, base_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load and merge ESPN rankings, ADP data, and projections
        
        Args:
            base_path: Root directory for data files
            
        Returns:
            DataFrame with player data and calculated probabilities
        """
        if base_path is None:
            base_path = '/Users/ben/projects/fantasy-football-draft-spreadsheet-draft-pick-odds'
            
        # Load data files
        espn_df = self._load_espn_data(base_path)
        adp_df = self._load_adp_data(base_path)
        proj_df = self._load_projections(base_path)
        
        # Merge and clean data
        players_df = self._merge_player_data(espn_df, adp_df, proj_df)
        
        # Calculate probabilities
        players_df = self._calculate_probabilities(players_df)
        
        self.players_df = players_df
        return players_df
        
    def _load_espn_data(self, base_path: str) -> pd.DataFrame:
        """Load ESPN rankings"""
        espn_file = os.path.join(base_path, 'data/espn_projections_20250814.csv')
        
        try:
            if os.path.exists(espn_file):
                espn_df = pd.read_csv(espn_file)
            else:
                print(f"Warning: ESPN file not found at {espn_file}")
                return pd.DataFrame()
                
            espn_df['espn_rank'] = espn_df['overall_rank']
            return espn_df[['player_name', 'position', 'espn_rank', 'team']]
            
        except Exception as e:
            print(f"Error loading ESPN data: {e}")
            return pd.DataFrame()
            
    def _load_adp_data(self, base_path: str) -> pd.DataFrame:
        """Load ADP (Average Draft Position) data"""
        adp_file = os.path.join(base_path, 'data/fantasypros_adp_20250815.csv')
        
        try:
            if os.path.exists(adp_file):
                adp_df = pd.read_csv(adp_file)
            else:
                print(f"Warning: ADP file not found at {adp_file}")
                return pd.DataFrame()
                
            adp_df['adp_rank'] = adp_df['RANK']
            adp_df['player_name'] = adp_df['PLAYER']
            return adp_df[['player_name', 'adp_rank']]
            
        except Exception as e:
            print(f"Error loading ADP data: {e}")
            return pd.DataFrame()
            
    def _load_projections(self, base_path: str) -> pd.DataFrame:
        """Load fantasy point projections"""
        proj_file = os.path.join(base_path, 'data/rankings_top300_20250814.csv')
        
        try:
            if os.path.exists(proj_file):
                proj_df = pd.read_csv(proj_file)
            else:
                print(f"Warning: Projections file not found at {proj_file}")
                return pd.DataFrame()
                
            # Clean player names (remove team abbreviations)
            proj_df['player_name'] = (proj_df['PLAYER']
                                     .str.replace(r'\s+[A-Z]{2,3}$', '', regex=True)
                                     .str.strip())
            proj_df['proj'] = proj_df['FANTASY_PTS'].fillna(100)
            
            return proj_df[['player_name', 'proj', 'POSITION']]
            
        except Exception as e:
            print(f"Error loading projections: {e}")
            return pd.DataFrame()
            
    def _merge_player_data(self, espn_df: pd.DataFrame, 
                          adp_df: pd.DataFrame, 
                          proj_df: pd.DataFrame) -> pd.DataFrame:
        """Merge all data sources into single DataFrame"""
        
        # Start with ESPN as base
        if espn_df.empty:
            merged = pd.DataFrame()
        else:
            merged = espn_df.copy()
            
        # Merge ADP data
        if not adp_df.empty:
            merged = merged.merge(adp_df, on='player_name', how='outer')
        else:
            merged['adp_rank'] = 300
            
        # Merge projections
        if not proj_df.empty:
            merged = merged.merge(proj_df, on='player_name', how='left')
        else:
            merged['proj'] = 50
            merged['POSITION'] = 'FLEX'
            
        # Clean positions
        if 'position' in merged.columns and 'POSITION' in merged.columns:
            merged['pos'] = merged['position'].fillna(merged['POSITION']).fillna('FLEX')
        elif 'position' in merged.columns:
            merged['pos'] = merged['position'].fillna('FLEX')
        else:
            merged['pos'] = 'FLEX'
            
        merged['pos'] = merged['pos'].str.extract(r'([A-Z]+)')[0]
        
        # Fill missing values
        merged['espn_rank'] = merged.get('espn_rank', 300).fillna(300)
        merged['adp_rank'] = merged.get('adp_rank', 300).fillna(300)
        merged['proj'] = merged.get('proj', 50).fillna(50)
        
        # Drop rows without player names
        merged = merged.dropna(subset=['player_name'])
        
        # Add player IDs
        merged['player_id'] = range(len(merged))
        merged = merged.set_index('player_id')
        
        return merged
        
    def _calculate_probabilities(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pick probabilities using softmax"""
        
        # ESPN probabilities
        espn_probs = self.softmax(players_df['espn_rank'].values)
        players_df['espn_prob'] = espn_probs
        
        # ADP probabilities
        adp_probs = self.softmax(players_df['adp_rank'].values)
        players_df['adp_prob'] = adp_probs
        
        # Combined probability (weighted average)
        players_df['pick_prob'] = (
            self.espn_weight * players_df['espn_prob'] + 
            self.adp_weight * players_df['adp_prob']
        )
        
        # Normalize to ensure sum = 1
        total_prob = players_df['pick_prob'].sum()
        if total_prob > 0:
            players_df['pick_prob'] = players_df['pick_prob'] / total_prob
            
        return players_df
        
    def softmax(self, ranks: np.ndarray) -> np.ndarray:
        """
        Convert ranks to probabilities using softmax with temperature
        
        Args:
            ranks: Array of player rankings (lower is better)
            
        Returns:
            Array of probabilities
        """
        if len(ranks) == 0:
            return np.array([])
            
        # Lower rank = better, so negate for softmax
        scores = -np.array(ranks, dtype=np.float64) / self.temperature
        
        # Clip extreme values to prevent overflow
        scores = np.clip(scores, -500, 500)
        
        # Subtract max for numerical stability
        exp_scores = np.exp(scores - np.max(scores))
        total = exp_scores.sum()
        
        if total == 0 or np.isnan(total) or np.isinf(total):
            # Fallback to uniform distribution
            return np.ones(len(ranks)) / len(ranks)
            
        return exp_scores / total
        
    def get_pick_probabilities(self, 
                              available_players: Set[int],
                              already_drafted: Optional[Set[str]] = None) -> Dict[int, float]:
        """
        Get pick probabilities for available players
        
        Args:
            available_players: Set of player IDs still available
            already_drafted: Optional set of player names already drafted
            
        Returns:
            Dictionary mapping player ID to pick probability
        """
        if self.players_df is None:
            raise ValueError("Must load data first with load_data()")
            
        # Filter to available players
        available_df = self.players_df.loc[list(available_players)]
        
        # Filter out already drafted by name if provided
        if already_drafted:
            available_df = available_df[~available_df['player_name'].isin(already_drafted)]
            
        # Get probabilities and renormalize
        probs = available_df['pick_prob'].to_dict()
        total = sum(probs.values())
        
        if total > 0:
            return {pid: p/total for pid, p in probs.items()}
        else:
            # Uniform if no probabilities
            n = len(available_df)
            return {pid: 1.0/n for pid in available_df.index}
            
    def calculate_survival_probability(self, 
                                      player_id: int,
                                      picks_until_next: int,
                                      available_players: Set[int]) -> float:
        """
        Calculate probability a player survives until your next pick
        
        Args:
            player_id: ID of player to check
            picks_until_next: Number of picks before your turn
            available_players: Set of currently available player IDs
            
        Returns:
            Probability player is still available (0-1)
        """
        if player_id not in available_players:
            return 0.0
            
        # Get current pick probabilities
        pick_probs = self.get_pick_probabilities(available_players)
        player_prob = pick_probs.get(player_id, 0)
        
        # Discrete survival: (1 - p)^n
        survival = (1 - player_prob) ** picks_until_next
        
        return survival
        
    def calibrate_temperature(self, draft_file: Optional[str] = None) -> float:
        """
        Calibrate temperature parameter based on actual draft data
        
        Args:
            draft_file: Path to draft results file (optional)
            
        Returns:
            Best temperature value
        """
        if draft_file and os.path.exists(draft_file):
            try:
                # Load actual draft data
                draft_df = pd.read_csv(draft_file)
                actual_picks = draft_df['player_name'].tolist()
                
                # Test different temperatures
                temperatures = [2.0, 3.0, 5.0, 8.0, 10.0]
                best_temp = 5.0
                best_mse = float('inf')
                
                for temp in temperatures:
                    old_temp = self.temperature
                    self.temperature = temp
                    
                    # Recalculate probabilities
                    if self.players_df is not None:
                        self.players_df = self._calculate_probabilities(self.players_df)
                        
                        # Calculate MSE against actual picks
                        mse = self._calculate_prediction_mse(actual_picks)
                        if mse < best_mse:
                            best_mse = mse
                            best_temp = temp
                    
                    # Restore old temperature
                    self.temperature = old_temp
                
                # Set best temperature
                self.temperature = best_temp
                if self.players_df is not None:
                    self.players_df = self._calculate_probabilities(self.players_df)
                
                return best_temp
                
            except Exception as e:
                print(f"Error calibrating temperature: {e}")
                return self.temperature
        else:
            # No draft file provided, return current temperature
            return self.temperature
            
    def _calculate_prediction_mse(self, actual_picks: List[str]) -> float:
        """Calculate MSE between predicted and actual pick probabilities"""
        if self.players_df is None:
            return float('inf')
            
        mse = 0.0
        for i, player_name in enumerate(actual_picks):
            # Find player in dataframe
            mask = self.players_df['player_name'] == player_name
            if mask.any():
                predicted_prob = self.players_df.loc[mask, 'pick_prob'].iloc[0]
                # Actual probability is 1 for picked player, 0 for others
                mse += (1.0 - predicted_prob) ** 2
                
        return mse / len(actual_picks) if actual_picks else float('inf')