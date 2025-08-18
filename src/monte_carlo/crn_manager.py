"""Common Random Numbers (CRN) Manager for Monte Carlo Simulation"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class CRNManager:
    """Pre-generates and manages random samples for CRN"""
    
    def __init__(self, n_max_sims: int = 5000, seed: int = 42, beta_concentration: float = 4.0):
        """Initialize CRN manager with pre-allocation"""
        self.n_max_sims = n_max_sims
        self.beta_concentration = beta_concentration
        self.rng = np.random.default_rng(seed)
        
        # Memory bounds checking
        estimated_mb = n_max_sims * 300 * 8 / 1_000_000
        if estimated_mb > 200:
            print(f"⚠️ Large memory allocation: ~{estimated_mb:.0f}MB")
        
        # Pre-allocated storage
        self.player_samples = {}  # {player_id: array(n_sims,)}
        self.team_multipliers = None  # array(n_teams, n_sims) for correlation
        self.opponent_seeds = None  # array(n_sims,) for opponent behavior
        
    def generate_all_samples(self, players_df: pd.DataFrame, n_teams: int = 14):
        """One-time generation of all random variates"""
        print(f"🎲 Generating CRN samples for {len(players_df)} players, {self.n_max_sims} sims...")
        
        # Generate Beta-PERT samples for each player
        for player_id in players_df.index:
            player = players_df.loc[player_id]
            
            # Get envelope data, fallback to BASE ± 20% if not available
            base = player.get('base', player.get('proj', 100))
            low = player.get('low', base * 0.8)
            high = player.get('high', base * 1.2)
            
            # Handle degenerate case
            if abs(high - low) < 1e-6:
                self.player_samples[player_id] = np.full(self.n_max_sims, base)
                continue
                
            # Beta-PERT parameters
            alpha = 1 + self.beta_concentration * (base - low) / (high - low)
            beta = 1 + self.beta_concentration * (high - base) / (high - low)
            
            # Generate all samples at once for this player
            u_samples = self.rng.beta(alpha, beta, size=self.n_max_sims)
            self.player_samples[player_id] = low + u_samples * (high - low)
            
        # Generate team correlation multipliers (for opponent behavior modeling)
        self.team_multipliers = self.rng.normal(1.0, 0.05, size=(n_teams, self.n_max_sims))
        
        # Generate opponent behavior seeds
        self.opponent_seeds = self.rng.integers(0, 2**31 - 1, size=self.n_max_sims)
        print(f"✅ CRN samples generated successfully")
        
    def get_projection(self, player_id: int, sim_idx: int, base: float = 0.0) -> float:
        """Retrieve pre-generated sample"""
        if not self.player_samples:
            raise ValueError("Must call generate_all_samples() first")
            
        if player_id not in self.player_samples:
            print(f"Warning: Unknown player {player_id}, using base projection")
            return base
            
        if sim_idx >= self.n_max_sims:
            raise ValueError(f"sim_idx {sim_idx} exceeds max_sims {self.n_max_sims}")
            
        return self.player_samples[player_id][sim_idx]
        
    def get_team_multiplier(self, team_idx: int, sim_idx: int) -> float:
        """Get team-specific correlation multiplier"""
        if not self.player_samples:
            raise ValueError("Must call generate_all_samples() first")
        return self.team_multipliers[team_idx, sim_idx]
        
    def get_opponent_seed(self, sim_idx: int) -> int:
        """Get opponent behavior seed for simulation"""
        if not self.player_samples:
            raise ValueError("Must call generate_all_samples() first")
        return self.opponent_seeds[sim_idx]
        
    def is_ready(self) -> bool:
        """Check if samples are generated and ready"""
        return bool(self.player_samples)