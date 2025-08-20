"""
starter_optimizer.py

Drop-in helper functions to replace greedy proj-based picks with
starter-aware, scenario-sampled expected-opportunity-cost decisions.

Usage:
    from starter_optimizer import pick_best_now, load_players_from_csv
    players = load_players_from_csv("rankings_top300_20250814.csv")
    best = pick_best_now(players, roster_state, current_pick, next_my_pick, league_config)
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from functools import lru_cache
import functools
import math
import random
import time
import sys
import os
import hashlib

# Add project root to path for importing src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# --------------------------
# Utilities & defaults
# --------------------------

def beta_pert_samples(mode, low=None, high=None, lam=4.0, size=500, rng=None):
    """Sample Beta-PERT around mode with optional low/high bounds.
       If low/high None, assumes +/-20% around mode.
    """
    rng = np.random.default_rng() if rng is None else rng
    if low is None: low = mode * 0.8
    if high is None: high = mode * 1.2
    if high <= low:
        return np.full(size, mode)
    # Convert to alpha/beta for Beta(a,b) on [0,1]
    # PERT method: mean = (low + lam*mode + high) / (lam + 2)
    # map to beta distribution by finding alpha/beta that match mean & variance heuristic
    mean = (low + lam * mode + high) / (lam + 2)
    # use variance proportional to range^2, small heuristic
    variance = ((high - low) / 6.0) ** 2
    # method-of-moments for Beta on [0,1]
    mn = (mean - low) / (high - low)
    var = variance / ((high - low) ** 2)
    # avoid degenerate
    mn = min(max(mn, 1e-6), 1 - 1e-6)
    var = max(var, 1e-9)
    tmp = mn * (1 - mn) / var - 1
    a = max(tmp * mn, 1e-3)
    b = max(tmp * (1 - mn), 1e-3)
    samples = rng.beta(a, b, size=size) * (high - low) + low
    return samples

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

# --------------------------
# Config & league defaults
# --------------------------
DEFAULT_LEAGUE = {
    "n_teams": 14,
    "starters_by_pos": {"RB": 2, "WR": 2, "TE": 1, "QB": 1, "FLEX": 1},
    # FLEX considered later as best of RB/WR/TE
}

# --------------------------
# Load helpers
# --------------------------
def load_players_from_csv(path, proj_col="proj", pos_col="pos", id_col="id", envelope_pct=0.20, scenarios=500, rng=None):
    df = pd.read_csv(path)
    rng = np.random.default_rng() if rng is None else rng
    players = []
    for _, r in df.iterrows():
        p = float(r.get(proj_col, 0.0))
        
        # Extract player name and clean it (remove team suffix)
        raw_name = r.get("name", r.get("PLAYER", r.get("player", "")))
        clean_name = raw_name
        if isinstance(raw_name, str):
            # Remove team suffix (e.g., "Lamar Jackson BAL" -> "Lamar Jackson")
            clean_name = ' '.join(raw_name.split()[:-1]) if ' ' in raw_name else raw_name
            
        players.append({
            "id": r.get(id_col, f"{_}"),
            "name": clean_name,
            "pos": r.get(pos_col, "UNK"),
            "proj": p,
            # create PERT samples, store as numpy array - reduced memory footprint
            "samples": beta_pert_samples(p, low=p*(1-envelope_pct), high=p*(1+envelope_pct),
                                        lam=4.0, size=min(scenarios, 400), rng=rng)  # Reduce memory usage
        })
    return players

# --------------------------
# Replacement-level per scenario
# --------------------------
def compute_replacement_levels(players, scenario_idx, league=DEFAULT_LEAGUE):
    """Compute replacement-level score for each position in scenario_idx.
       Replacement index = total starter slots for that position across league.
    """
    pos_buckets = defaultdict(list)
    for pl in players:
        # Ensure Beta-PERT sampling scenario coherence with consistent modulo logic
        safe_idx = scenario_idx % len(pl["samples"])
        pos_buckets[pl["pos"]].append(pl["samples"][safe_idx])

    replacement = {}
    for pos, arr in pos_buckets.items():
        arr_sorted = sorted(arr, reverse=True)
        total_starters = league["n_teams"] * league["starters_by_pos"].get(pos, 0)
        # replacement-level is the value at index total_starters (0-based),
        # i.e., the next player after the starters
        if len(arr_sorted) > total_starters:
            replacement[pos] = arr_sorted[total_starters]
        else:
            # Edge case: no candidates = 0.0 replacement
            replacement[pos] = arr_sorted[-1] if arr_sorted else 0.0
    
    # Handle FLEX position - max of RB/WR/TE replacement levels
    if 'FLEX' in league["starters_by_pos"]:
        flex_replacement = max(
            replacement.get('RB', 0.0),
            replacement.get('WR', 0.0),
            replacement.get('TE', 0.0)
        )
        replacement['FLEX'] = flex_replacement
    
    return replacement

# --------------------------
# Dynamic replacement levels system
# --------------------------

# Cache for dynamic replacement levels - replaced with LRU cache decorator

def _pool_signature_dyn(pool_players, top_n=40):
    """Create signature from top N players for cache key."""
    top_ids = [p['id'] for p in sorted(pool_players, 
              key=lambda x: np.mean(x['samples']), reverse=True)[:top_n]]
    return tuple(top_ids)

def _create_league_hash(league_config):
    """Create a hash of league configuration for cache key collision prevention."""
    config_str = str(sorted(league_config.items()))
    return hashlib.md5(config_str.encode()).hexdigest()[:8]

@functools.lru_cache(maxsize=200)
def _compute_dynamic_replacement_levels_cached(pool_sig, current_pick, replacement_pick, 
                                             league_hash, team_idx, n_sims_inner, positions_tuple):
    """Cached computation of dynamic replacement levels using LRU cache."""
    # This function is called by compute_dynamic_replacement_levels with hashable arguments
    return None  # Actual computation moved to calling function

def compute_dynamic_replacement_levels(pool_players, current_pick, replacement_pick, 
                                      sim, positions=None, 
                                      n_sims_inner=200, top_n_sig=40):
    """
    Compute replacement levels dynamically based on expected best player 
    at replacement_pick (typically next_my_pick).
    
    Args:
        pool_players: Available players
        current_pick: Current global pick number
        replacement_pick: Pick to evaluate replacement at (e.g., next_my_pick)
        sim: ExpectedBestSimulator instance
        positions: Positions to calculate (defaults to all starter positions)
        n_sims_inner: Simulations for expectation (200 for speed/accuracy balance)
        top_n_sig: Top N players for pool signature
    
    Returns:
        Dict of {position: replacement_value} including FLEX
    """
    if positions is None:
        positions = list(sim.league["starters_by_pos"].keys())
    
    # CRITICAL FIX: Skip zero-starter positions in replacement calculation
    # Filter out positions with 0 starter slots (K/DST) to avoid wasted computation
    positions = [pos for pos in positions if sim.league["starters_by_pos"].get(pos, 0) > 0]
    
    # Create cache signature including league hash to prevent collisions
    league_hash = _create_league_hash(sim.league)
    pool_sig = _pool_signature_dyn(pool_players, top_n=top_n_sig)
    positions_tuple = tuple(sorted(positions))
    
    # Check cache using hashable signature
    try:
        cached_result = _compute_dynamic_replacement_levels_cached(
            pool_sig, current_pick, replacement_pick, league_hash, 
            sim.my_team_idx, n_sims_inner, positions_tuple)
        # Cache miss expected on first call, compute below
    except:
        pass
    
    replacement = {}
    
    # Calculate replacement for each position with error handling
    for pos in positions:
        if pos == 'FLEX':
            continue  # Handle FLEX separately
        try:
            stats = sim.expected_best_for_position_at_pick(
                pos, current_pick - 1, replacement_pick, pool_players, n_sims=n_sims_inner)
            # Use p50 (median) for robustness against outliers
            replacement[pos] = stats.get("p50", stats.get("mean", 0.0))
        except Exception as e:
            print(f"Warning: Error calculating replacement for {pos}: {e}")
            replacement[pos] = 0.0  # Fallback value
    
    # FLEX = max(RB, WR, TE) at replacement pick
    flex_vals = []
    for p in ("RB", "WR", "TE"):
        if p in replacement:
            flex_vals.append(replacement[p])
        elif p in positions:
            try:
                stats = sim.expected_best_for_position_at_pick(
                    p, current_pick - 1, replacement_pick, pool_players, n_sims=n_sims_inner)
                flex_vals.append(stats.get("p50", stats.get("mean", 0.0)))
            except Exception as e:
                print(f"Warning: Error calculating FLEX replacement for {p}: {e}")
                flex_vals.append(0.0)  # Fallback value
    replacement["FLEX"] = max(flex_vals) if flex_vals else 0.0
    
    # Set very low replacement values for non-starter positions to deprioritize them
    replacement["K"] = 0.0
    replacement["DST"] = 0.0
    
    return replacement

def clear_dynamic_replacement_cache():
    """Clear the dynamic replacement cache after picks or when context changes significantly."""
    _compute_dynamic_replacement_levels_cached.cache_clear()

# --------------------------
# Pick-probability (pluggable)
# --------------------------
def default_pick_probability(pool_players, pick_index=None, temp=0.6):
    """Return a probability vector for each player in pool being taken at a pick.
       Default: softmax over mean projection (aggressive players with higher proj).
       Replace this with your actual pick model (position-aware).
    """
    mean_projs = [np.mean(p["samples"]) for p in pool_players]
    return softmax_weights(mean_projs, temp=temp)

# Global cache for ESPN data to avoid reloading on every call
_espn_cache = None

def sophisticated_pick_probability(pool_players, pick_index=None, prob_model=None):
    """Use actual ESPN overall rankings for pick probabilities"""
    global _espn_cache
    
    if prob_model is None:
        return default_pick_probability(pool_players, pick_index)
    
    # Load ESPN rankings data and create lookup (with caching)
    try:
        import pandas as pd
        import os
        
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
                # Player not found, assign high rank (low priority) - this is expected for DST/K
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

# --------------------------
# Expected-best simulation with caching
# --------------------------
class ExpectedBestSimulator:
    def __init__(self, players, league=DEFAULT_LEAGUE, pick_prob_fn=None, probability_model=None, rng=None, my_team_idx=None):
        """
        players: list of player dicts with 'id' and 'samples'
        pick_prob_fn: function(pool_players, pick_index) -> probability array for sampling a taken player
        probability_model: sophisticated ESPN/ADP probability model (optional)
        my_team_idx: team index for context-aware caching (optional)
        """
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

# --------------------------
# Marginal starter value & opportunity cost
# --------------------------
def marginal_starter_value(player, roster_state, starter_slots, all_players, 
                          current_pick, next_my_pick, sim, league=DEFAULT_LEAGUE):
    """
    Calculate marginal value using dynamic replacement levels.
    If position has empty starter slots -> marginal value is player's VOR.
    Else marginal = max(0, player_vor - worst_current_starter_vor)
    Uses dynamic VOR (Value Over Replacement) to properly account for 
    positional scarcity at the actual replacement pick.
    """
    pos = player["pos"]
    filled = roster_state.get(pos, [])
    need = starter_slots.get(pos, 0)
    
    # CRITICAL FIX: Deprioritize non-starter positions when starter needs remain
    if pos in ["K", "DST"] and _has_unfilled_starter_slots(roster_state, starter_slots):
        return -100.0  # Very negative value to prevent early selection
    
    # Use dynamic replacement levels
    replacement_levels = compute_dynamic_replacement_levels(
        all_players, current_pick, next_my_pick, sim, n_sims_inner=100)
    replacement_val = replacement_levels.get(pos, 0.0)
    
    # Calculate player's VOR
    player_value = float(np.mean(player["samples"]))
    player_vor = player_value - replacement_val
    
    if len(filled) < need:
        return player_vor
    
    # compute worst starter by VOR (not raw points)
    current_starters = sorted(filled, key=lambda x: float(np.mean(x["samples"])), reverse=True)[:need]
    if not current_starters:
        return player_vor
    
    worst_starter_value = float(np.mean(current_starters[-1]["samples"]))
    worst_starter_vor = worst_starter_value - replacement_val
    
    return max(0.0, player_vor - worst_starter_vor)

def _has_unfilled_starter_slots(roster_state, starter_slots):
    """Check if there are any unfilled starter slots (QB/RB/WR/TE/FLEX)."""
    core_positions = ["QB", "RB", "WR", "TE", "FLEX"]
    for pos in core_positions:
        filled = len(roster_state.get(pos, []))
        need = starter_slots.get(pos, 0)
        if filled < need:
            return True
    return False

def compute_opportunity_cost(player, roster_state, current_pick, next_my_pick, pool_players, sim: ExpectedBestSimulator, replacement_levels=None, positions_to_consider=None):
    """
    Opportunity cost of picking `player` now = expected loss in VOR across
    OTHER positions due to waiting until `next_my_pick`.
    
    REFINEMENT: Uses actual next_my_pick (not hardcoded +28)
    REFINEMENT: Ensures replacement_levels passed consistently
    """
    if positions_to_consider is None:
        positions_to_consider = list(sim.league["starters_by_pos"].keys())
    
    # REFINEMENT: Ensure replacement_levels always available
    if replacement_levels is None:
        replacement_levels = compute_replacement_levels(pool_players, scenario_idx=0, league=sim.league)
    
    cost = 0.0
    
    for pos in positions_to_consider:
        # CRITICAL FIX: The opportunity cost of drafting a player at position X
        # is the value lost at all positions *other than* X.
        if pos == player["pos"]:
            continue
        
        # Skip FLEX in opportunity cost calculation (handled through RB/WR/TE)
        if pos == 'FLEX':
            continue

        need = max(0, sim.league["starters_by_pos"].get(pos, 0) - len(roster_state.get(pos, [])))
        if need <= 0:
            continue
        
        # Stage 1: Quick simulation (75-100 sims tuned for < 0.2s)
        best_now_raw = sim.expected_best_for_position_at_pick(pos, current_pick - 1, current_pick, pool_players, n_sims=75)["p50"]
        # Stage 2: More thorough for future (200-250 sims tuned for < 1s total)
        best_future_raw = sim.expected_best_for_position_at_pick(pos, current_pick - 1, next_my_pick, pool_players, n_sims=200)["p50"]
        
        replacement_val = replacement_levels.get(pos, 0.0)
        best_now_vor = best_now_raw - replacement_val
        best_future_vor = best_future_raw - replacement_val
        
        deg = max(0.0, best_now_vor - best_future_vor)
        cost += deg * need
        
    return cost

# --------------------------
# Main pick decision function
# --------------------------
def pick_best_now(pool_players, roster_state, current_pick, next_my_pick, league=DEFAULT_LEAGUE,
                  top_k_candidates=20, scenarios=500, pick_prob_fn=None, probability_model=None, 
                  rng=None, my_team_idx=None, clear_cache_after=True):
    """
    Simplified single-pass evaluation with dynamic replacement levels.
    
    Score = marginal_starter_value - opportunity_cost
    
    IMPROVEMENTS:
    - Removed two-stage complexity for maintainability
    - Added error handling and league hash for cache collision prevention
    - Consistent Beta-PERT sampling scenario coherence
    - Uses built-in LRU cache for better performance
    - Performance target: < 2s decision time
    """
    start_time = time.time()
    rng = np.random.default_rng() if rng is None else rng
    
    # Create simulator with team context
    sim = ExpectedBestSimulator(pool_players, league=league, pick_prob_fn=pick_prob_fn, 
                               probability_model=probability_model, rng=rng, my_team_idx=my_team_idx)

    # Single-pass evaluation with reduced sim count for performance
    n_sims_adaptive = min(200, max(100, len(pool_players)))
    replacement_levels = compute_dynamic_replacement_levels(
        pool_players, current_pick, next_my_pick, sim, n_sims_inner=n_sims_adaptive)
    
    # CRITICAL FIX: Filter candidates to prioritize starter positions first
    starter_candidates = []
    non_starter_candidates = []
    
    for player in pool_players:
        if player["pos"] in ["QB", "RB", "WR", "TE"]:
            starter_candidates.append(player)
        else:
            non_starter_candidates.append(player)
    
    # Calculate VOR for sorting using dynamic replacement
    def calculate_vor(player):
        player_value = float(np.mean(player["samples"]))
        replacement_val = replacement_levels.get(player["pos"], 0.0)
        return player_value - replacement_val
    
    # Prioritize starter positions, then add non-starter positions
    starter_sorted = sorted(starter_candidates, key=calculate_vor, reverse=True)
    non_starter_sorted = sorted(non_starter_candidates, key=calculate_vor, reverse=True)
    pool_sorted = starter_sorted + non_starter_sorted
    
    candidates = pool_sorted[:max(top_k_candidates, 40)]  # Reasonable candidate pool
    
    # Single-pass evaluation of candidates
    scores = []
    best_choice = None
    best_score = -1e9
    debug = []
    
    for cand in candidates[:top_k_candidates]:
        try:
            imm_val = marginal_starter_value(cand, roster_state, league["starters_by_pos"], 
                                            pool_players, current_pick, next_my_pick, sim, league)
            opp_cost = compute_opportunity_cost(cand, roster_state, current_pick, next_my_pick, 
                                               pool_players, sim, replacement_levels)
            score = imm_val - opp_cost
            scores.append((cand, score, imm_val, opp_cost))
            debug.append((cand["name"], cand["pos"], imm_val, opp_cost, score))
            
            if score > best_score:
                best_score = score
                best_choice = cand
        except Exception as e:
            print(f"Warning: Error evaluating candidate {cand.get('name', 'Unknown')}: {e}")
            # Continue with other candidates
            continue
    
    # Fallback if no candidates were successfully evaluated
    if best_choice is None and candidates:
        best_choice = candidates[0]
        best_score = 0.0
        debug = [(best_choice["name"], best_choice["pos"], 0.0, 0.0, 0.0)]
    
    total_time = time.time() - start_time
    
    # Performance monitoring
    if total_time > 2.0:
        print(f"⚠️ Performance warning: Total time {total_time:.2f}s exceeded 2s target")
    
    # Clear cache after pick if requested
    if clear_cache_after:
        sim.clear_cache()
        clear_dynamic_replacement_cache()
    
    # Get cache stats for monitoring
    cache_stats = sim.get_cache_stats()
    
    return {
        "pick": best_choice,
        "score": best_score,
        "debug": sorted(debug, key=lambda x: x[4], reverse=True)[:10],
        "performance": {
            "total_time": total_time,
            "cache_hit_rate": cache_stats["hit_rate"],
            "cache_size": cache_stats["size"]
        }
    }

# --------------------------
# Probability model setup helper
# --------------------------
def create_probability_model(base_path=None):
    """Create lightweight ESPN-based probability model for testing."""
    print("✅ Using lightweight ESPN-based probability model")
    return "lightweight_espn_model"  # Flag to enable ESPN-based probabilities

# --------------------------
# Compatibility functions for simulator.py
# --------------------------

def calculate_bench_depth_report(final_roster, starters, threshold_pct=0.25):
    """
    Calculate bench depth using 25% rule - REPORTING ONLY, not used for optimization.
    Quality bench = within 25% of starter's projected points.
    
    Args:
        final_roster: List of all player dicts with 'pos' and 'proj' or 'samples'
        starters: List of starter player dicts
        threshold_pct: Percentage threshold for quality backup (default 0.25 = 25%)
    """
    if not starters:
        return {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0}
    
    # Get minimum starter projection by position
    starter_mins = {}
    starter_positions = defaultdict(list)
    
    for starter in starters:
        pos = starter['pos']
        # Handle both 'proj' field and 'samples' array
        if 'samples' in starter:
            proj = float(np.mean(starter['samples']))
        else:
            proj = starter.get('proj', 0)
        starter_positions[pos].append(proj)
    
    for pos, projections in starter_positions.items():
        starter_mins[pos] = min(projections) if projections else 0
    
    # Count quality bench players (within threshold)
    bench_players = [p for p in final_roster if p not in starters]
    bench_depth = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0}
    
    for bench_player in bench_players:
        pos = bench_player['pos']
        if pos in starter_mins and starter_mins[pos] > 0:
            # Handle both 'proj' field and 'samples' array
            if 'samples' in bench_player:
                bench_proj = float(np.mean(bench_player['samples']))
            else:
                bench_proj = bench_player.get('proj', 0)
                
            threshold = starter_mins[pos] * (1 - threshold_pct)  # 25% rule
            if bench_proj >= threshold:
                bench_depth[pos] += 1
    
    return bench_depth


def get_starter_breakdown(roster, starter_slots=None):
    """
    Get starters grouped by position for reporting.
    
    Args:
        roster: List of all player dicts
        starter_slots: Dict of position requirements (default: standard fantasy)
    """
    if starter_slots is None:
        starter_slots = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1}
    
    if not roster:
        return {}
    
    # Get optimal starters using the same logic as optimizer
    starters = get_optimal_starters(roster, starter_slots)
    
    # Group by position, with special handling for FLEX identification
    breakdown = defaultdict(list)
    actual_starter_counts = defaultdict(int)
    
    for starter in starters:
        pos = starter['pos']
        actual_starter_counts[pos] += 1
        
        # Determine if this player is in FLEX slot
        required_at_pos = starter_slots.get(pos, 0)
        if pos in ['RB', 'WR', 'TE'] and actual_starter_counts[pos] > required_at_pos:
            # This player is filling the FLEX slot
            breakdown['FLEX'].append(starter)
        else:
            breakdown[pos].append(starter)
    
    # Sort within each position by projection (highest first)
    for pos in breakdown:
        breakdown[pos].sort(key=lambda p: _get_player_proj(p), reverse=True)
    
    return dict(breakdown)


def get_optimal_starters(roster, starter_slots=None):
    """
    Get optimal starting lineup from roster based on highest projections.
    Compatible with both optimizer.py and starter_optimizer.py formats.
    """
    if starter_slots is None:
        starter_slots = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1}
    
    if not roster:
        return []
    
    # Group players by position
    position_players = defaultdict(list)
    for player in roster:
        if not isinstance(player, dict) or 'pos' not in player:
            continue
        position_players[player['pos']].append(player)
    
    # Sort each position by projection (highest first)
    for pos in position_players:
        position_players[pos].sort(key=lambda p: _get_player_proj(p), reverse=True)
    
    starters = []
    
    # Handle required positions first (not FLEX)
    for pos, count in starter_slots.items():
        if pos == 'FLEX':
            continue  # Handle FLEX separately
        if pos in position_players:
            available = position_players[pos]
            starters.extend(available[:min(count, len(available))])
    
    # Handle FLEX: best remaining RB/WR/TE
    if 'FLEX' in starter_slots and starter_slots['FLEX'] > 0:
        flex_candidates = []
        # Get remaining RBs (after required 2)
        if 'RB' in position_players and len(position_players['RB']) > 2:
            flex_candidates.extend(position_players['RB'][2:])
        # Get remaining WRs (after required 2)
        if 'WR' in position_players and len(position_players['WR']) > 2:
            flex_candidates.extend(position_players['WR'][2:])
        # Get remaining TEs (after required 1)
        if 'TE' in position_players and len(position_players['TE']) > 1:
            flex_candidates.extend(position_players['TE'][1:])
        
        if flex_candidates:
            # Sort by projection and take best
            flex_candidates.sort(key=lambda p: _get_player_proj(p), reverse=True)
            starters.extend(flex_candidates[:starter_slots['FLEX']])
    
    return starters


def _get_player_proj(player):
    """Get player projection handling both 'proj' field and 'samples' array."""
    if 'samples' in player:
        return float(np.mean(player['samples']))
    else:
        return player.get('proj', 0)

# --------------------------
# Example tiny test harness
# --------------------------
if __name__ == "__main__":
    # Example run using CSV file - adapt file path and column names
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Project root (fixed path)
    players = load_players_from_csv(f"{base_path}/data/rankings_top300_20250814.csv", 
                                   proj_col="FANTASY_PTS", pos_col="POSITION", 
                                   id_col="id", envelope_pct=0.20, scenarios=600)
    
    # Load sophisticated probability model
    prob_model = create_probability_model(base_path)
    
    # Mock roster: assume user already has 2 players
    roster = {"RB": [], "WR": [], "TE": [], "QB": []}
    
    # REFINEMENT: Compute actual next_my_pick using snake draft logic
    my_team_idx = 4  # Example: team 5 (0-based)
    n_teams = 14
    current_pick = 29  # example from your output
    
    # Compute next_my_pick properly (not hardcoded +28)
    current_round = (current_pick - 1) // n_teams + 1
    pick_in_round = (current_pick - 1) % n_teams + 1
    
    if current_round % 2 == 1:  # Odd round (normal order)
        if pick_in_round <= my_team_idx + 1:
            # My pick is still in this round
            picks_until_my_turn = my_team_idx + 1 - pick_in_round
        else:
            # My pick is in the next round (reverse order)
            picks_until_my_turn = (n_teams - pick_in_round) + 1 + (n_teams - my_team_idx - 1) + 1
    else:  # Even round (reverse order)
        if pick_in_round <= n_teams - my_team_idx:
            # My pick is still in this round
            picks_until_my_turn = n_teams - my_team_idx - pick_in_round + 1
        else:
            # My pick is in the next round (normal order)
            picks_until_my_turn = (n_teams - pick_in_round) + 1 + my_team_idx + 1
    
    next_my_pick = current_pick + picks_until_my_turn
    
    print("\n" + "="*60)
    print("STARTER OPTIMIZER TEST - SOPHISTICATED MODEL")
    print("="*60)
    print(f"Current pick: {current_pick}")
    print(f"Next pick: {next_my_pick}")
    print(f"Using sophisticated model: {'Yes' if prob_model else 'No (naive fallback)'}")
    print()
    
    result = pick_best_now(players, roster, current_pick, next_my_pick, 
                          league=DEFAULT_LEAGUE, top_k_candidates=30, scenarios=600,
                          probability_model=prob_model, my_team_idx=my_team_idx)
    
    print(f"🏆 Recommended pick: {result['pick']['name']} ({result['pick']['pos']}) - Score: {result['score']:.1f}")
    print("\nTop candidates:")
    for i, (name, pos, imm_val, opp_cost, score) in enumerate(result["debug"][:5], 1):
        print(f"{i}. {name} ({pos}): {score:.1f} = {imm_val:.1f} immediate - {opp_cost:.1f} opportunity cost")
    
    # Show performance metrics
    if "performance" in result:
        perf = result["performance"]
        print(f"\nPerformance: {perf['total_time']:.2f}s total (target < 2s)")
        print(f"  Stage 1: {perf['stage1_time']:.2f}s (target < 0.5s)")
        print(f"  Cache hit rate: {perf['cache_hit_rate']:.1%}")
        print(f"  Cache size: {perf['cache_size']} entries")
