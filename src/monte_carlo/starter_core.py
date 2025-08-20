"""
starter_core.py - Core Starter Sum optimization functions

NORTH STAR PRINCIPLE: "Does this increase expected starter lineup points under uncertainty, relative to waiting?"

Core functions for Starter Sum (SS) optimization:
- compute_starter_sum(): Calculate total points from optimal starting lineup
- marginal_starter_gain(): SS improvement when adding a candidate
- compute_opportunity_cost_ss(): SS difference between picking now vs waiting

Usage:
    from starter_core import compute_starter_sum, marginal_starter_gain
    ss = compute_starter_sum(roster_players, scenario_idx, starter_slots)
    msg = marginal_starter_gain(candidate, roster_players, scenario_idx, starter_slots)
"""

import numpy as np
from collections import defaultdict


def compute_starter_sum(roster_players, scenario_idx, starter_slots, league=None):
    """
    Compute total points from optimal starting lineup in a specific scenario.
    This is the core "Starter Sum" function per the North Star principle.
    
    Args:
        roster_players: List of player dicts with 'samples' arrays
        scenario_idx: Which scenario/draw to use for player values
        starter_slots: Dict of position requirements (e.g., {'QB': 1, 'RB': 2, ...})
        
    Returns:
        float: Total points from optimal starting lineup
    """
    if not roster_players:
        return 0.0
    
    # Group players by position with scenario-specific values
    pos_players = defaultdict(list)
    for player in roster_players:
        pos = player["pos"]
        # Ensure scenario coherence
        safe_idx = scenario_idx % len(player["samples"])
        value = player["samples"][safe_idx]
        pos_players[pos].append((value, player))
    
    # Sort each position by value (highest first)
    for pos in pos_players:
        pos_players[pos].sort(key=lambda x: x[0], reverse=True)
    
    total_points = 0.0
    used_players = set()
    
    # Fill required positions first (non-FLEX)
    for pos, count in starter_slots.items():
        if pos == 'FLEX':
            continue
        available = [(val, p) for val, p in pos_players.get(pos, []) if id(p) not in used_players]
        for i, (value, player) in enumerate(available[:count]):
            total_points += value
            used_players.add(id(player))
    
    # Handle FLEX: best remaining RB/WR/TE
    flex_count = starter_slots.get('FLEX', 0)
    if flex_count > 0:
        flex_candidates = []
        for pos in ['RB', 'WR', 'TE']:
            available = [(val, p) for val, p in pos_players.get(pos, []) if id(p) not in used_players]
            flex_candidates.extend(available)
        
        # Sort by value and take best
        flex_candidates.sort(key=lambda x: x[0], reverse=True)
        for i, (value, player) in enumerate(flex_candidates[:flex_count]):
            total_points += value
            used_players.add(id(player))
    
    return total_points


def marginal_starter_gain(candidate, roster_players, scenario_idx, starter_slots, league=None):
    """
    Calculate Marginal Starter Gain (MSG): improvement in starter sum if we add this candidate.
    This replaces VOR-based marginal value with North Star aligned starter sum optimization.
    
    Returns: SS(with_candidate) - SS(without_candidate)
    """
    # Baseline starter sum without candidate
    baseline_ss = compute_starter_sum(roster_players, scenario_idx, starter_slots, league)
    
    # Starter sum with candidate added
    roster_with_candidate = roster_players + [candidate]
    enhanced_ss = compute_starter_sum(roster_with_candidate, scenario_idx, starter_slots, league)
    
    return enhanced_ss - baseline_ss


def compute_opportunity_cost_ss(roster_players, current_pool, current_pick, next_my_pick, 
                               sim, starter_slots, n_scenarios=100, league=None):
    """
    Compute Opportunity Cost using Starter Sum framework: SS(now) - SS(next_pick).
    This replaces position-by-position degradation with holistic starter lineup comparison.
    
    Returns: Expected difference in starter sum between picking now vs waiting
    """
    opportunity_costs = []
    
    for scenario_idx in range(min(n_scenarios, 200)):  # Cap for performance
        try:
            # Current starter sum with available pool
            current_ss = 0.0
            if current_pool:
                # Find best available player for this roster right now
                best_current = None
                best_gain = -1e9
                for candidate in current_pool[:20]:  # Top candidates only
                    gain = marginal_starter_gain(candidate, roster_players, scenario_idx, starter_slots, league)
                    if gain > best_gain:
                        best_gain = gain
                        best_current = candidate
                
                if best_current:
                    current_ss = compute_starter_sum(roster_players + [best_current], scenario_idx, starter_slots, league)
                else:
                    current_ss = compute_starter_sum(roster_players, scenario_idx, starter_slots, league)
            
            # Future starter sum after opponent picks
            future_pool = list(current_pool)  # Copy for simulation
            picks_to_simulate = max(0, next_my_pick - current_pick)
            
            # Simulate opponent picks
            for _ in range(picks_to_simulate):
                if not future_pool:
                    break
                probs = sim.pick_prob_fn(future_pool, current_pick)
                if len(probs) != len(future_pool) or sum(probs) <= 0:
                    probs = [1.0/len(future_pool)] * len(future_pool)
                else:
                    probs = np.array(probs) / sum(probs)
                
                chosen_idx = sim.rng.choice(len(future_pool), p=probs)
                future_pool.pop(chosen_idx)
            
            # Best available at future pick
            future_ss = 0.0
            if future_pool:
                best_future = None
                best_gain = -1e9
                for candidate in future_pool[:20]:
                    gain = marginal_starter_gain(candidate, roster_players, scenario_idx, starter_slots, league)
                    if gain > best_gain:
                        best_gain = gain
                        best_future = candidate
                
                if best_future:
                    future_ss = compute_starter_sum(roster_players + [best_future], scenario_idx, starter_slots, league)
                else:
                    future_ss = compute_starter_sum(roster_players, scenario_idx, starter_slots, league)
            
            # Opportunity cost for this scenario
            oc = max(0.0, current_ss - future_ss)
            opportunity_costs.append(oc)
            
        except Exception:
            # Fallback for errors
            opportunity_costs.append(0.0)
    
    return float(np.mean(opportunity_costs)) if opportunity_costs else 0.0


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


def _get_player_proj(player):
    """Get player projection handling both 'proj' field and 'samples' array."""
    if 'samples' in player:
        return float(np.mean(player['samples']))
    else:
        return player.get('proj', 0)