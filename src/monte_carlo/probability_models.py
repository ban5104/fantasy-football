"""
probability_models.py - Probability models and caching for draft simulations

Contains ExpectedBestSimulator and sophisticated ESPN-based probability modeling
with performance optimizations and caching.

Usage:
    from probability_models import ExpectedBestSimulator, create_probability_model
    sim = ExpectedBestSimulator(players, probability_model=create_probability_model())
    stats = sim.expected_best_for_position_at_pick("RB", 29, 41, pool_players)
"""

import numpy as np
import pandas as pd
import os
import hashlib
from collections import defaultdict


# Global cache for ESPN data to avoid reloading on every call
_espn_cache = None


def softmax_weights(values, temp=1.0):
    """Safe softmax returning non-negative weights summing to 1. Avoid overflow."""
    vals = np.array(values, dtype=float)
    vals = vals / max(temp, 1e-9)
    vals -= vals.max()
    ex = np.exp(vals)
    s = ex.sum()
    if s <= 0:
        return np.ones_like(ex) / len(ex)
    return ex / s


def default_pick_probability(pool_players, pick_index=None, temp=0.6):
    """Return a probability vector for each player in pool being taken at a pick.
       Default: softmax over mean projection (aggressive players with higher proj).
       Replace this with your actual pick model (position-aware).
    """
    mean_projs = [np.mean(p["samples"]) for p in pool_players]
    return softmax_weights(mean_projs, temp=temp)


def sophisticated_pick_probability(pool_players, pick_index=None, prob_model=None):
    """Use actual ESPN overall rankings for pick probabilities"""
    global _espn_cache
    
    if prob_model is None:
        return default_pick_probability(pool_players, pick_index)
    
    # Load ESPN rankings data and create lookup (with caching)
    try:
        # Check cache first
        if _espn_cache is None:
            # Get project root path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            espn_file = os.path.join(project_root, 'data', 'espn_projections_20250814.csv')
            
            if not os.path.exists(espn_file):
                print(f"Warning: ESPN file not found at {espn_file}, falling back to naive model")
                return default_pick_probability(pool_players, pick_index)
            
            # Load ESPN data once and cache it
            espn_df = pd.read_csv(espn_file)
            
            # Create name -> overall_rank lookup (clean names for matching)
            espn_lookup = {}
            for idx, row in espn_df.iterrows():
                name = str(row['player_name']).strip()
                # Remove team suffix if present (e.g., "Saquon Barkley, PHI" -> "Saquon Barkley")
                clean_name = name.split(',')[0].strip()
                espn_lookup[clean_name] = row['overall_rank']
            
            _espn_cache = espn_lookup
            print("✅ Using lightweight ESPN-based probability model")
        
        # Use cached data
        espn_lookup = _espn_cache
        
        # Get ESPN ranks for each player in the pool
        espn_ranks = []
        for player in pool_players:
            player_name = str(player.get('name', '')).strip()
            # Remove team suffix if present
            clean_player_name = player_name.split(',')[0].strip()
            
            # Look up in ESPN data
            if clean_player_name in espn_lookup:
                espn_rank = espn_lookup[clean_player_name]
            else:
                # Player not found, assign high rank (low priority)
                espn_rank = 400
                
            espn_ranks.append(espn_rank)
        
        # Convert ranks to probabilities (lower rank = higher probability)
        # Use 1/(rank+1) so rank 1 gets highest probability
        inverted_ranks = [1.0 / (rank + 1) for rank in espn_ranks]
        
        # Normalize to probabilities
        total = sum(inverted_ranks)
        return [p / total for p in inverted_ranks] if total > 0 else [1.0/len(pool_players)] * len(pool_players)
        
    except Exception as e:
        print(f"Error loading ESPN rankings: {e}")
        return default_pick_probability(pool_players, pick_index)


def create_probability_model(base_path=None):
    """Create lightweight ESPN-based probability model for testing."""
    print("✅ Using lightweight ESPN-based probability model")
    return "lightweight_espn_model"  # Flag to enable ESPN-based probabilities


def _create_league_hash(league_config):
    """Create a hash of league configuration for cache key collision prevention."""
    config_str = str(sorted(league_config.items()))
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


class ExpectedBestSimulator:
    def __init__(self, players, league=None, pick_prob_fn=None, probability_model=None, rng=None, my_team_idx=None):
        """
        players: list of player dicts with 'id' and 'samples'
        pick_prob_fn: function(pool_players, pick_index) -> probability array for sampling a taken player
        probability_model: sophisticated ESPN/ADP probability model (optional)
        my_team_idx: team index for context-aware caching (optional)
        """
        if league is None:
            league = {"n_teams": 14, "starters_by_pos": {"RB": 2, "WR": 2, "TE": 1, "QB": 1, "FLEX": 1}}
            
        self.players = players
        self.league = league
        self.probability_model = probability_model
        self.my_team_idx = my_team_idx
        
        # Choose probability function based on available model
        if pick_prob_fn:
            self.pick_prob_fn = pick_prob_fn
        elif probability_model is not None:
            # Use sophisticated model with round-aware adjustments
            def wrapped_sophisticated_fn(pool_players, pick_index=None):
                return sophisticated_pick_probability(pool_players, pick_index, probability_model)
            self.pick_prob_fn = wrapped_sophisticated_fn
        else:
            # Fallback to naive model
            self.pick_prob_fn = default_pick_probability
            
        self.rng = np.random.default_rng() if rng is None else rng
        # Enhanced cache key includes league and team context
        self._cache = {}  # keyed by (top_ids_tuple, position, current_pick, future_pick, league_hash, my_team_idx, scenarios)
        self._cache_hits = 0
        self._cache_misses = 0
        self._league_hash = _create_league_hash(league)
    
    def _pool_signature(self, pool, top_n=40):
        # signature: tuple of top_n player ids sorted - keeps cache small-ish
        sorted_ids = tuple(sorted([p["id"] for p in sorted(pool, key=lambda x: np.mean(x["samples"]), reverse=True)[:top_n]]))
        return sorted_ids

    def expected_best_for_position_at_pick(self, position, current_pick, future_pick, pool_players, n_sims=400, top_n_sig=40):
        """
        Simulate picks from current_pick+1 up to future_pick to find distribution of best-of-position
        in the pool at future_pick (when it's our turn).
        Returns expected_best (mean) and optional percentiles.
        """
        # Enhanced cache key with league hash to prevent collisions
        key = (
            self._pool_signature(pool_players, top_n=top_n_sig), 
            position, 
            current_pick, 
            future_pick, 
            self._league_hash,
            self.my_team_idx,
            n_sims
        )
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]
        
        self._cache_misses += 1

        sims_best = []
        picks_to_sim = max(0, future_pick - current_pick)
        # Precompute initial pool list
        initial_pool = list(pool_players)
        for s in range(n_sims):
            pool = initial_pool.copy()
            # simulate picks_to_sim picks (others in the league)
            for pick_offset in range(picks_to_sim):
                if not pool:
                    break
                probs = self.pick_prob_fn(pool, current_pick + pick_offset + 1)
                # safety: normalize
                probs = np.array(probs, dtype=float)
                if probs.sum() <= 0:
                    probs = np.ones_like(probs) / len(probs)
                else:
                    probs = probs / probs.sum()
                chosen_idx = self.rng.choice(len(pool), p=probs)
                pool.pop(chosen_idx)
            # after simulation, find best-of-position in pool by sampled value for this scenario s
            # CRITICAL FIX: Ensure Beta-PERT sampling scenario coherence with consistent modulo logic
            pos_vals = []
            for p in pool:
                # Ensure scenario coherence: use same scenario index for all players in this simulation
                sample_idx = s % len(p["samples"])
                pos_vals.append(p["samples"][sample_idx] if p["pos"] == position else -1e9)
            if pos_vals:
                max_val = max(pos_vals)
                # Edge case: if no candidates for position, return 0.0
                if max_val < 0:
                    sims_best.append(0.0)
                else:
                    sims_best.append(max_val)
            else:
                # Edge case: empty pool = 0.0 replacement
                sims_best.append(0.0)

        sims_best = np.array(sims_best)
        result = {
            "mean": float(sims_best.mean()),
            "p50": float(np.percentile(sims_best, 50)),  # Keep using median (p50) as decided
            "p10": float(np.percentile(sims_best, 10)),
            "p90": float(np.percentile(sims_best, 90))
        }
        self._cache[key] = result
        
        # Cache management: clear if getting too large (LRU-like behavior)
        if len(self._cache) > 500:  # Threshold for cache size
            # Keep most recent half of entries
            keys_to_keep = list(self._cache.keys())[-250:]
            self._cache = {k: self._cache[k] for k in keys_to_keep}
        
        return result
    
    def clear_cache(self):
        """Clear the cache after picks or when context changes significantly."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_cache_stats(self):
        """Return cache statistics for performance monitoring."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "size": len(self._cache)
        }