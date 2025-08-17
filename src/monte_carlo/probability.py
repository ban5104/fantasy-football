"""Probability Model for Fantasy Football Draft - Simplified"""

import numpy as np
import pandas as pd
import os
from typing import Dict, Set, Optional


class ProbabilityModel:
    """Calculate draft pick probabilities from ESPN and ADP data"""
    
    def __init__(self, espn_weight=0.8, adp_weight=0.2, temperature=5.0):
        self.espn_weight = espn_weight
        self.adp_weight = adp_weight
        self.temperature = temperature
        self.players_df = None
        
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
        """Load fantasy point projections"""
        proj_file = os.path.join(base_path, 'data/rankings_top300_20250814.csv')
        
        try:
            if os.path.exists(proj_file):
                proj_df = pd.read_csv(proj_file)
                proj_df['player_name'] = (proj_df['PLAYER']
                                         .str.replace(r'\s+[A-Z]{2,3}$', '', regex=True)
                                         .str.strip())
                proj_df['proj'] = proj_df['FANTASY_PTS'].fillna(100)
                return proj_df[['player_name', 'proj', 'POSITION']]
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