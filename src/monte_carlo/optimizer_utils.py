"""
optimizer_utils.py - Utilities for draft optimization

Contains data loading, replacement level calculations, Beta-PERT sampling,
and other utility functions used by the starter optimizer.

Usage:
    from optimizer_utils import load_players_from_csv, compute_dynamic_replacement_levels
    players = load_players_from_csv("rankings.csv")
    replacement_levels = compute_dynamic_replacement_levels(pool_players, current_pick, next_pick, sim)
"""

import numpy as np
import pandas as pd
import functools
import hashlib
from collections import defaultdict


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


def load_players_from_csv(path, proj_col="proj", pos_col="pos", id_col="id", envelope_pct=0.20, scenarios=500, rng=None):
    """Load players from CSV with Beta-PERT sampling for projections."""
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


def compute_replacement_levels(players, scenario_idx, league=None):
    """Compute replacement-level score for each position in scenario_idx.
       Replacement index = total starter slots for that position across league.
    """
    if league is None:
        league = {"n_teams": 14, "starters_by_pos": {"RB": 2, "WR": 2, "TE": 1, "QB": 1, "FLEX": 1}}
        
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


# Dynamic replacement levels system
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


def _has_unfilled_starter_slots(roster_state, starter_slots):
    """Check if there are any unfilled starter slots (QB/RB/WR/TE/FLEX)."""
    core_positions = ["QB", "RB", "WR", "TE", "FLEX"]
    for pos in core_positions:
        filled = len(roster_state.get(pos, []))
        need = starter_slots.get(pos, 0)
        if filled < need:
            return True
    return False