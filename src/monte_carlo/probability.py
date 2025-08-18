"""Probability Model for Fantasy Football Draft - Simplified"""

import numpy as np
import pandas as pd
import os
from typing import Dict, Set, Optional, List


class ProbabilityModel:
    """Calculate draft pick probabilities from ESPN and ADP data"""
    
    def __init__(self, espn_weight=0.8, adp_weight=0.2, temperature=5.0):
        self.espn_weight = espn_weight
        self.adp_weight = adp_weight
        self.temperature = temperature
        self.players_df = None
        self.rng = np.random.default_rng(42)
        self.beta_concentration = 4.0
        
    def load_data(self, base_path=None):
        """Load and merge ESPN rankings, ADP data, and projections"""
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
        
    def _load_espn_data(self, base_path):
        """Load ESPN rankings"""
        espn_file = os.path.join(base_path, 'data/espn_projections_20250814.csv')
        
        try:
            if os.path.exists(espn_file):
                espn_df = pd.read_csv(espn_file)
                espn_df['espn_rank'] = espn_df['overall_rank']
                return espn_df[['player_name', 'position', 'espn_rank', 'team']]
            else:
                print(f"Warning: ESPN file not found at {espn_file}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error loading ESPN data: {e}")
            return pd.DataFrame()
            
    def _load_adp_data(self, base_path):
        """Load ADP data"""
        adp_file = os.path.join(base_path, 'data/fantasypros_adp_20250815.csv')
        
        try:
            if os.path.exists(adp_file):
                adp_df = pd.read_csv(adp_file)
                adp_df['adp_rank'] = adp_df['RANK']
                adp_df['player_name'] = adp_df['PLAYER']
                return adp_df[['player_name', 'adp_rank']]
            else:
                print(f"Warning: ADP file not found at {adp_file}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error loading ADP data: {e}")
            return pd.DataFrame()
            
    def _load_projections(self, base_path):
        """Load fantasy point projections with envelope data"""
        proj_file = os.path.join(base_path, 'data/rankings_top300_20250814.csv')
        
        try:
            if os.path.exists(proj_file):
                proj_df = pd.read_csv(proj_file)
                proj_df['player_name'] = (proj_df['PLAYER']
                                         .str.replace(r'\s+[A-Z]{2,3}$', '', regex=True)
                                         .str.strip())
                proj_df['proj'] = proj_df['FANTASY_PTS'].fillna(100)
                
                # Try to load envelope data if available
                envelope_file = os.path.join(base_path, 'data/player_envelopes.csv')
                if os.path.exists(envelope_file):
                    try:
                        env_df = pd.read_csv(envelope_file)
                        # Expecting columns: player_name, low, base, high
                        proj_df = proj_df.merge(env_df, on='player_name', how='left')
                        print(f"Loaded envelope data for {len(env_df)} players")
                    except Exception as e:
                        print(f"Warning: Error loading envelope data: {e}")
                else:
                    # Create envelope data from projections (BASE ± 20%)
                    proj_df['base'] = proj_df['proj']
                    proj_df['low'] = proj_df['proj'] * 0.8
                    proj_df['high'] = proj_df['proj'] * 1.2
                    print("Created envelope data from projections (±20%)")
                
                return proj_df[['player_name', 'proj', 'POSITION', 'low', 'base', 'high']]
            else:
                print(f"Warning: Projections file not found at {proj_file}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error loading projections: {e}")
            return pd.DataFrame()
            
    def _merge_player_data(self, espn_df, adp_df, proj_df):
        """Merge all data sources"""
        if espn_df.empty:
            merged = pd.DataFrame()
        else:
            merged = espn_df.copy()
            
        # Merge ADP and projections
        if not adp_df.empty:
            merged = merged.merge(adp_df, on='player_name', how='outer')
        else:
            merged['adp_rank'] = 300
            
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
        
        # Fix envelope data - fill NaN values and ensure proper bounds
        if 'low' in merged.columns:
            merged['low'] = merged['low'].fillna(merged['proj'] * 0.8)
        else:
            merged['low'] = merged['proj'] * 0.8
            
        if 'base' in merged.columns:
            merged['base'] = merged['base'].fillna(merged['proj'])
        else:
            merged['base'] = merged['proj']
            
        if 'high' in merged.columns:
            merged['high'] = merged['high'].fillna(merged['proj'] * 1.2)
        else:
            merged['high'] = merged['proj'] * 1.2
        
        # Drop rows without player names and add IDs
        merged = merged.dropna(subset=['player_name'])
        merged['player_id'] = range(len(merged))
        merged = merged.set_index('player_id')
        
        return merged
        
    def _calculate_probabilities(self, players_df):
        """Calculate pick probabilities using softmax"""
        # ESPN and ADP probabilities
        espn_probs = self.softmax(players_df['espn_rank'].values)
        adp_probs = self.softmax(players_df['adp_rank'].values)
        
        # Combined weighted probability
        players_df['pick_prob'] = (
            self.espn_weight * espn_probs + 
            self.adp_weight * adp_probs
        )
        
        # Normalize
        total_prob = players_df['pick_prob'].sum()
        if total_prob > 0:
            players_df['pick_prob'] = players_df['pick_prob'] / total_prob
            
        return players_df
        
    def softmax(self, ranks):
        """Convert ranks to probabilities using softmax"""
        if len(ranks) == 0:
            return np.array([])
            
        scores = -np.array(ranks, dtype=np.float64) / self.temperature
        scores = np.clip(scores, -500, 500)  # Prevent overflow
        exp_scores = np.exp(scores - np.max(scores))
        total = exp_scores.sum()
        
        if total == 0 or np.isnan(total):
            return np.ones(len(ranks)) / len(ranks)  # Uniform fallback
            
        return exp_scores / total
        
    def get_pick_probabilities(self, available_players, already_drafted=None):
        """Get pick probabilities for available players"""
        if self.players_df is None:
            raise ValueError("Must load data first")
            
        available_df = self.players_df.loc[list(available_players)]
        
        if already_drafted:
            available_df = available_df[~available_df['player_name'].isin(already_drafted)]
            
        probs = available_df['pick_prob'].to_dict()
        total = sum(probs.values())
        
        if total > 0:
            return {pid: p/total for pid, p in probs.items()}
        else:
            n = len(available_df)
            return {pid: 1.0/n for pid in available_df.index}
            
    def calculate_survival_probability(self, player_id, picks_until_next, available_players):
        """Calculate probability a player survives until your next pick"""
        if player_id not in available_players:
            return 0.0
            
        pick_probs = self.get_pick_probabilities(available_players)
        player_prob = pick_probs.get(player_id, 0)
        
        # Discrete survival: (1 - p)^n
        return (1 - player_prob) ** picks_until_next
    
    def sample_projections(self, player_ids: List[int] = None, sim_seed: int = None) -> Dict[int, float]:
        """Sample player projections using Beta-PERT distribution. Handles both individual and batch sampling."""
        if self.players_df is None:
            return {}
            
        # If no specific players requested, sample all
        if player_ids is None:
            player_ids = list(self.players_df.index)
        elif isinstance(player_ids, int):
            # Single player case
            player_ids = [player_ids]
            
        # Use simulation-specific seed for reproducibility
        if sim_seed is not None:
            rng = np.random.default_rng(sim_seed)
        else:
            rng = self.rng
            
        # Filter to valid player IDs
        valid_ids = [pid for pid in player_ids if pid in self.players_df.index]
        if not valid_ids:
            return {}
            
        # Get data for requested players
        player_data = self.players_df.loc[valid_ids]
        
        # Get envelope data arrays with fallbacks
        base_values = player_data.get('base', player_data.get('proj', 100)).values
        low_values = player_data.get('low', base_values * 0.8).values
        high_values = player_data.get('high', base_values * 1.2).values
        
        # Vectorized Beta-PERT sampling
        ranges = high_values - low_values
        valid_mask = ranges > 1e-6
        
        # Initialize result array
        sampled_values = base_values.copy()
        
        if np.any(valid_mask):
            # Vectorized computation for valid cases only
            valid_base = base_values[valid_mask]
            valid_low = low_values[valid_mask] 
            valid_high = high_values[valid_mask]
            valid_ranges = valid_high - valid_low
            
            # Beta-PERT parameters (vectorized)
            base_norm = (valid_base - valid_low) / valid_ranges
            alpha = 1 + self.beta_concentration * base_norm
            beta = 1 + self.beta_concentration * (1 - base_norm)
            
            # Sample Beta values for all valid players at once
            u_samples = rng.beta(alpha, beta)
            
            # Transform to final values
            sampled_values[valid_mask] = valid_low + u_samples * valid_ranges
            
        # Convert to dict with player IDs
        return dict(zip(valid_ids, sampled_values))
        
    # Legacy methods for backward compatibility
    def sample_player_projection(self, player_id: int, sim_seed: int = None) -> float:
        """Legacy method - use sample_projections instead"""
        result = self.sample_projections([player_id], sim_seed)
        return result.get(player_id, 0.0)
        
    def sample_all_projections(self, sim_seed: int = None) -> Dict[int, float]:
        """Legacy method - use sample_projections instead"""
        return self.sample_projections(None, sim_seed)
    
    def has_envelope_data(self) -> bool:
        """Check if envelope data is available"""
        if self.players_df is None:
            return False
        return any(col in self.players_df.columns for col in ['low', 'base', 'high'])

    def _to_dict(self) -> dict:
        """Serialize model state for multiprocessing"""
        return {
            'espn_weight': self.espn_weight,
            'adp_weight': self.adp_weight,
            'temperature': self.temperature,
            'beta_concentration': self.beta_concentration,
            'players_df': self.players_df.to_dict('records') if self.players_df is not None else None,
            'players_index': self.players_df.index.tolist() if self.players_df is not None else None
        }

    @classmethod
    def _from_dict(cls, data: dict):
        """Deserialize model state from multiprocessing"""
        model = cls(
            espn_weight=data['espn_weight'],
            adp_weight=data['adp_weight'], 
            temperature=data['temperature']
        )
        model.beta_concentration = data['beta_concentration']
        
        if data['players_df'] is not None:
            model.players_df = pd.DataFrame(data['players_df'])
            model.players_df.index = data['players_index']
        
        return model