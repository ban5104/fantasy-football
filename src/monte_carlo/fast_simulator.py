"""Ultra-Fast Monte Carlo Simulation Engine - Optimized for Live Draft"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import time
from numba import jit, prange
import concurrent.futures
from functools import lru_cache

from .strategies import ROSTER_REQUIREMENTS, POSITION_LIMITS, get_strategy


class FastMonteCarloSimulator:
    """Optimized Monte Carlo simulator for sub-minute 1000+ simulations"""
    
    def __init__(self, probability_model, opponent_model, n_teams=14, n_rounds=14):
        self.prob_model = probability_model
        self.opponent_model = opponent_model
        self.n_teams = n_teams
        self.n_rounds = n_rounds
        
        # Pre-compute all pick orders
        self.pick_order = self._precompute_pick_order()
        
        # Pre-allocate arrays for performance
        self.n_picks = n_teams * n_rounds
        
        # Cache for frequently accessed data
        self._cache = {}
        
    def _precompute_pick_order(self):
        """Pre-compute snake draft order once"""
        order = np.zeros(self.n_teams * self.n_rounds, dtype=np.int32)
        idx = 0
        for round_num in range(self.n_rounds):
            if round_num % 2 == 0:
                order[idx:idx+self.n_teams] = np.arange(self.n_teams)
            else:
                order[idx:idx+self.n_teams] = np.arange(self.n_teams-1, -1, -1)
            idx += self.n_teams
        return order
    
    def _presample_all_projections(self, n_sims: int):
        """Pre-sample ALL player projections for ALL simulations at once"""
        if not self.prob_model.has_envelope_data():
            # No sampling needed - use static values
            return None
            
        players_df = self.prob_model.players_df
        n_players = len(players_df)
        
        # Pre-allocate array for all sampled values
        sampled_values = np.zeros((n_players, n_sims), dtype=np.float32)
        
        # Vectorized Beta-PERT sampling for all players and simulations
        for idx, (player_id, player) in enumerate(players_df.iterrows()):
            base = player.get('base', player.get('proj', 100))
            low = player.get('low', base * 0.8)
            high = player.get('high', base * 1.2)
            
            if abs(high - low) < 1e-6:
                sampled_values[idx, :] = base
            else:
                # Vectorized Beta-PERT sampling
                alpha = 1 + 4.0 * (base - low) / (high - low)
                beta = 1 + 4.0 * (high - base) / (high - low)
                
                # Sample all simulations at once
                u = np.random.beta(alpha, beta, n_sims)
                sampled_values[idx, :] = low + u * (high - low)
        
        return sampled_values
    
    @lru_cache(maxsize=1024)
    def _get_position_requirements(self, pos: str) -> int:
        """Cached position requirements lookup"""
        return ROSTER_REQUIREMENTS.get(pos, 0)
    
    @lru_cache(maxsize=1024)
    def _get_position_limit(self, pos: str) -> int:
        """Cached position limits lookup"""
        return POSITION_LIMITS.get(pos, 3)
    
    def _create_player_lookup_arrays(self):
        """Create numpy arrays for fast player lookups"""
        players_df = self.prob_model.players_df
        n_players = len(players_df)
        
        # Create lookup arrays
        player_ids = np.arange(n_players, dtype=np.int32)
        positions = np.array([self._encode_position(p) for p in players_df['pos'].values], dtype=np.int8)
        espn_ranks = players_df['espn_rank'].values.astype(np.float32)
        base_projections = players_df['proj'].values.astype(np.float32)
        
        return {
            'ids': player_ids,
            'positions': positions,
            'ranks': espn_ranks,
            'projections': base_projections
        }
    
    def _encode_position(self, pos: str) -> int:
        """Encode position string to integer for faster comparison"""
        pos_map = {'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3, 'K': 4, 'DST': 5, 'FLEX': 6}
        return pos_map.get(pos, 6)
    
    def _calculate_roster_value_vectorized(self, roster_matrix, sampled_values, sim_idx):
        """Vectorized roster value calculation"""
        # roster_matrix: boolean matrix of [n_players x roster_size]
        # sampled_values: pre-sampled projections [n_players x n_sims]
        
        if sampled_values is not None:
            player_values = sampled_values[:, sim_idx]
        else:
            player_values = self.prob_model.players_df['proj'].values
        
        # Get values for rostered players
        roster_values = player_values[roster_matrix]
        
        # Fast starter selection using numpy operations
        # This is simplified - you'd need position-specific logic
        roster_values_sorted = np.sort(roster_values)[::-1]
        
        # Take top 9 as starters (simplified)
        n_starters = min(9, len(roster_values_sorted))
        starter_points = np.sum(roster_values_sorted[:n_starters])
        
        # Bench value
        if len(roster_values_sorted) > n_starters:
            bench_value = np.sum(roster_values_sorted[n_starters:]) * 0.1
        else:
            bench_value = 0
        
        return starter_points + bench_value
    
    def run_simulations_parallel(self, my_team_idx: int, strategy_name: str, 
                                n_sims: int = 1000, n_workers: int = 4,
                                initial_roster=None, already_drafted=None):
        """Run simulations in parallel for massive speedup"""
        
        # Pre-sample all projections upfront
        print(f"⚡ Pre-sampling {n_sims} projections...")
        start = time.time()
        sampled_values = self._presample_all_projections(n_sims)
        print(f"   Sampling done in {time.time() - start:.2f}s")
        
        # Pre-compute lookup arrays
        lookup_arrays = self._create_player_lookup_arrays()
        
        # Split simulations across workers
        sims_per_worker = n_sims // n_workers
        remainder = n_sims % n_workers
        
        # Prepare work packages
        work_packages = []
        sim_start = 0
        for i in range(n_workers):
            worker_sims = sims_per_worker + (1 if i < remainder else 0)
            if worker_sims > 0:
                work_packages.append({
                    'start_idx': sim_start,
                    'n_sims': worker_sims,
                    'my_team_idx': my_team_idx,
                    'strategy_name': strategy_name,
                    'sampled_values': sampled_values[:, sim_start:sim_start+worker_sims] if sampled_values is not None else None,
                    'lookup_arrays': lookup_arrays,
                    'initial_roster': initial_roster,
                    'already_drafted': already_drafted
                })
                sim_start += worker_sims
        
        # Run in parallel
        print(f"⚡ Running {n_sims} simulations on {n_workers} workers...")
        start = time.time()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(self._run_simulation_batch, pkg) for pkg in work_packages]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        print(f"   Simulations done in {time.time() - start:.2f}s")
        
        # Combine results
        all_results = []
        for batch_result in results:
            all_results.extend(batch_result['results'])
        
        # Calculate statistics
        values = np.array([r['roster_value'] for r in all_results])
        
        return {
            'mean_value': np.mean(values),
            'std_value': np.std(values),
            'all_results': all_results
        }
    
    def _run_simulation_batch(self, work_package):
        """Run a batch of simulations (for parallel processing)"""
        results = []
        
        for i in range(work_package['n_sims']):
            # Simplified simulation logic - you'd adapt your existing logic here
            # Key is to use the pre-sampled values and lookup arrays
            result = self._simulate_single_fast(
                work_package['my_team_idx'],
                work_package['strategy_name'],
                work_package['sampled_values'][:, i] if work_package['sampled_values'] is not None else None,
                work_package['lookup_arrays'],
                seed=work_package['start_idx'] + i
            )
            results.append(result)
        
        return {'results': results}
    
    def _simulate_single_fast(self, my_team_idx, strategy_name, sampled_values, lookup_arrays, seed):
        """Fast single simulation using pre-computed data"""
        # This is a simplified version - adapt your existing logic
        # Key optimizations:
        # 1. Use pre-sampled values
        # 2. Use integer position encoding
        # 3. Use numpy arrays instead of DataFrames
        # 4. Minimize string operations
        
        rng = np.random.default_rng(seed)
        
        # Fast simulation logic here...
        # Return same format as your current simulator
        
        return {
            'roster': [],
            'roster_value': rng.uniform(1200, 1500),  # Placeholder
            'starter_points': rng.uniform(1100, 1400),
            'depth_bonus': rng.uniform(50, 100),
            'starters': []
        }
    
    def run_simulations_vectorized(self, my_team_idx: int, strategy_name: str, 
                                   n_sims: int = 1000, initial_roster=None, 
                                   already_drafted=None):
        """Fully vectorized simulation for maximum speed"""
        
        print(f"⚡ Running {n_sims} vectorized simulations...")
        start = time.time()
        
        # Pre-sample all values
        sampled_values = self._presample_all_projections(n_sims)
        
        # Pre-compute all random decisions for all simulations
        n_picks = self.n_teams * self.n_rounds
        n_players = len(self.prob_model.players_df)
        
        # Generate all random numbers needed upfront
        random_matrix = np.random.random((n_sims, n_picks, n_players))
        
        # Run vectorized simulation
        results = self._simulate_vectorized_batch(
            my_team_idx, strategy_name, sampled_values, 
            random_matrix, initial_roster, already_drafted
        )
        
        elapsed = time.time() - start
        print(f"✅ Completed {n_sims} simulations in {elapsed:.2f} seconds")
        print(f"   ({n_sims/elapsed:.0f} sims/second)")
        
        return results
    
    def _simulate_vectorized_batch(self, my_team_idx, strategy_name, 
                                   sampled_values, random_matrix,
                                   initial_roster, already_drafted):
        """Fully vectorized batch simulation"""
        # This would contain the core vectorized logic
        # Key is to process all simulations simultaneously using numpy operations
        
        n_sims = random_matrix.shape[0]
        
        # Placeholder for demonstration
        values = np.random.normal(1400, 50, n_sims)
        
        return {
            'mean_value': np.mean(values),
            'std_value': np.std(values),
            'min_value': np.min(values),
            'max_value': np.max(values),
            'all_results': [{'roster_value': v} for v in values]
        }


# Numba-accelerated helper functions
@jit(nopython=True)
def calculate_roster_value_numba(roster_array, value_array):
    """Numba-compiled roster value calculation for extreme speed"""
    # Sort and select top values
    sorted_values = np.sort(value_array[roster_array])[::-1]
    
    # Calculate starter value (top 9)
    starter_value = np.sum(sorted_values[:min(9, len(sorted_values))])
    
    # Calculate bench value
    if len(sorted_values) > 9:
        bench_value = np.sum(sorted_values[9:]) * 0.1
    else:
        bench_value = 0.0
    
    return starter_value + bench_value