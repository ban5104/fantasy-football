"""
Starter-Aware Draft Optimizer
ONLY maximizes 7 starter projected points throughout all 14 rounds.
NO bench optimization - bench depth is just informational reporting after draft.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from functools import lru_cache


def _player_to_hashable(player: dict) -> tuple:
    """Convert player dict to hashable tuple for caching"""
    if not isinstance(player, dict):
        return ()
    # Sort items to ensure consistent hash for identical players
    return tuple(sorted(player.items()))

def _hashable_to_player(player_tuple: tuple) -> dict:
    """Convert hashable tuple back to player dict"""
    return dict(player_tuple)

@lru_cache(maxsize=128)
def _get_optimal_starters_cached(roster_tuple: tuple, starter_slots_tuple: tuple) -> tuple:
    """Cached version of get_optimal_starters for performance"""
    try:
        # Convert tuple back to dict/list for processing
        roster = [_hashable_to_player(player_tuple) for player_tuple in roster_tuple]
        starter_slots = dict(starter_slots_tuple)
        
        if not roster:
            return tuple()
        
        # Group players by position
        position_players = defaultdict(list)
        for player in roster:
            if not isinstance(player, dict) or 'pos' not in player or 'proj' not in player:
                continue  # Skip malformed player data
            position_players[player['pos']].append(player)
        
        # Sort each position by projection (highest first)
        for pos in position_players:
            position_players[pos].sort(key=lambda p: p.get('proj', 0), reverse=True)
        
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
                flex_candidates.sort(key=lambda p: p.get('proj', 0), reverse=True)
                starters.extend(flex_candidates[:starter_slots['FLEX']])
        
        # Convert starters back to hashable tuples for return
        return tuple(_player_to_hashable(player) for player in starters)
    
    except (TypeError, ValueError, AttributeError):
        # If conversion fails, return empty tuple
        return tuple()

def get_optimal_starters(roster: List[dict], starter_slots: Dict[str, int] = None) -> List[dict]:
    """Get optimal starting lineup from roster based on highest projections"""
    if starter_slots is None:
        starter_slots = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1}  # 7 starters total
    
    if not roster:
        return []
    
    # Validate roster structure
    valid_roster = []
    for player in roster:
        if isinstance(player, dict) and 'pos' in player and 'proj' in player:
            valid_roster.append(player)
    
    if not valid_roster:
        return []
    
    try:
        # Convert players to hashable tuples for caching
        roster_tuple = tuple(_player_to_hashable(player) for player in valid_roster)
        starter_slots_tuple = tuple(sorted(starter_slots.items()))
        
        # Use cached function
        result_tuple = _get_optimal_starters_cached(roster_tuple, starter_slots_tuple)
        
        # Convert result back to list of dicts
        return [_hashable_to_player(player_tuple) for player_tuple in result_tuple]
    
    except (TypeError, ValueError, AttributeError):
        # Fallback to non-cached computation if caching fails
        print("Warning: Starter caching failed, using fallback computation")
        return _get_optimal_starters_fallback(valid_roster, starter_slots)

def _get_optimal_starters_fallback(roster: List[dict], starter_slots: Dict[str, int]) -> List[dict]:
    """Non-cached fallback computation for get_optimal_starters"""
    if not roster:
        return []
    
    # Group players by position
    position_players = defaultdict(list)
    for player in roster:
        if not isinstance(player, dict) or 'pos' not in player or 'proj' not in player:
            continue  # Skip malformed player data
        position_players[player['pos']].append(player)
    
    # Sort each position by projection (highest first)
    for pos in position_players:
        position_players[pos].sort(key=lambda p: p.get('proj', 0), reverse=True)
    
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
            flex_candidates.sort(key=lambda p: p.get('proj', 0), reverse=True)
            starters.extend(flex_candidates[:starter_slots['FLEX']])
    
    return starters

def calculate_marginal_starter_value(roster: List[dict], player: dict, 
                                   starter_slots: Dict[str, int] = None) -> float:
    """
    How much does this player improve my starting lineup?
    This is the ONLY metric that matters for optimization.
    """
    if starter_slots is None:
        starter_slots = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1}
    
    # Validate inputs
    if not isinstance(player, dict) or 'proj' not in player:
        return 0.0
    
    if not isinstance(roster, list):
        return 0.0
    
    try:
        # Current starting lineup value
        current_starters = get_optimal_starters(roster, starter_slots)
        current_total = float(sum(p.get('proj', 0) for p in current_starters))
        
        # New starting lineup value with this player added
        roster_with_player = roster + [player]
        new_starters = get_optimal_starters(roster_with_player, starter_slots)
        new_total = float(sum(p.get('proj', 0) for p in new_starters))
        
        return new_total - current_total
    except (TypeError, KeyError):
        return 0.0


def get_best_available(position: str, available_players: set, player_cache: dict) -> Optional[dict]:
    """Get best available player at position by projection"""
    # Validate inputs
    if not position or not available_players or not player_cache:
        return None
    
    # Validate player_cache structure
    required_keys = ['pos', 'proj', 'player_name']
    if not all(key in player_cache for key in required_keys):
        return None
    
    best_player = None
    best_projection = 0
    
    try:
        for player_id in available_players:
            if (player_id in player_cache['pos'] and 
                player_cache['pos'][player_id] == position):
                projection = player_cache['proj'].get(player_id, 0)
                if projection > best_projection:
                    best_projection = projection
                    best_player = {
                        'id': player_id,
                        'pos': position,
                        'proj': projection,
                        'name': player_cache['player_name'].get(player_id, 'Unknown')
                    }
    except (TypeError, KeyError):
        return None
    
    return best_player


def estimate_starter_opportunity_cost(available_players: set, player_position: str, 
                                    picks_until_next: int, player_cache: dict,
                                    roster: List[dict] = None,
                                    starter_slots: Dict[str, int] = None) -> float:
    """
    What do we lose at OTHER positions by taking this position now?
    Properly accounts for FLEX needs - we need 3 RBs/WRs total (2 for position + 1 potential FLEX).
    """
    if starter_slots is None:
        starter_slots = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1}
    
    # Validate inputs
    if not available_players or not player_position:
        return 0.0
    
    if not player_cache or not isinstance(picks_until_next, (int, float)):
        return 0.0
    
    if picks_until_next < 0:
        return 0.0
    
    # Count current roster by position
    current_roster_counts = defaultdict(int)
    if roster:
        for player in roster:
            if isinstance(player, dict) and 'pos' in player:
                current_roster_counts[player['pos']] += 1
    
    cost = 0.0
    
    try:
        # Calculate opportunity cost for each position
        for other_pos in ['QB', 'RB', 'WR', 'TE']:
            if other_pos == player_position:
                continue  # Skip same position
            
            # Calculate total needed for position-specific slots only
            # NOTE: FLEX opportunity cost was causing QB delays (Rd2->Rd5) and 81-point drops
            # Original FLEX logic made system too aggressive for RB/WR depth, overriding elite QB value
            # FLEX is handled automatically in lineup optimization, not draft strategy selection
            total_needed = starter_slots.get(other_pos, 0)
            
            # How many more do we still need?
            current_count = current_roster_counts[other_pos]
            remaining_needed = max(0, total_needed - current_count)
            
            if remaining_needed <= 0:
                continue  # We have enough at this position
            
            # Get best available now vs expected after picks_until_next
            best_now = get_best_available(other_pos, available_players, player_cache)
            
            if best_now and 'proj' in best_now:
                # Higher degradation for scarce positions (RB/WR), lower for QB/TE
                degradation_rate = 0.05 if other_pos in ['RB', 'WR'] else 0.03
                expected_degradation = best_now['proj'] * degradation_rate * picks_until_next
                
                # Weight by how urgently we need this position
                urgency_multiplier = 1.0
                if other_pos in ['RB', 'WR'] and remaining_needed >= 2:
                    urgency_multiplier = 1.5  # Extra penalty for being behind on key positions
                
                cost += expected_degradation * remaining_needed * urgency_multiplier
                
    except (TypeError, KeyError):
        return 0.0
    
    return cost


def optimize_pick(roster: List[dict], available_players: set, current_pick: int, 
                 next_pick: int, player_cache: dict, risk_aversion: float = 0.5) -> Optional[int]:
    """
    Main optimization logic: maximize marginal starter value.
    
    Returns:
        player_id of optimal pick, or None if no valid players
    """
    # Validate inputs
    if not available_players:
        return None
    
    if not isinstance(roster, list):
        return None
    
    if not player_cache or not isinstance(player_cache, dict):
        return None
    
    # Validate player_cache structure
    required_keys = ['pos', 'proj', 'player_name']
    if not all(key in player_cache for key in required_keys):
        return None
    
    # Validate risk_aversion parameter
    if not isinstance(risk_aversion, (int, float)) or risk_aversion < 0.0 or risk_aversion > 1.0:
        risk_aversion = 0.5  # Default fallback
    
    # Validate pick numbers
    if not isinstance(current_pick, (int, float)) or not isinstance(next_pick, (int, float)):
        return None
    
    picks_until_next = max(0, next_pick - current_pick)  # Ensure non-negative
    best_player_id = None
    best_net_value = float('-inf')
    
    try:
        # Evaluate each available player
        for player_id in available_players:
            if player_id not in player_cache['pos']:
                continue
                
            # Create player dict for evaluation with safe lookups
            player = {
                'id': player_id,
                'pos': player_cache['pos'].get(player_id, ''),
                'proj': player_cache['proj'].get(player_id, 0),
                'name': player_cache['player_name'].get(player_id, 'Unknown')
            }
            
            # Skip if missing essential data
            if not player['pos'] or player['proj'] <= 0:
                continue
            
            # Calculate marginal starter value (the PRIMARY metric)
            marginal_value = calculate_marginal_starter_value(roster, player)
            
            # Calculate opportunity cost at other positions
            opportunity_cost = estimate_starter_opportunity_cost(
                available_players, player['pos'], picks_until_next, player_cache, roster
            )
            
            # Net value with risk adjustment
            # risk_aversion=0.0 (aggressive): fully weight opportunity cost (flexible, diverse)
            # risk_aversion=1.0 (conservative): ignore opportunity cost (rigid, best player available)
            net_value = marginal_value - ((1.0 - risk_aversion) * opportunity_cost)
            
            if net_value > best_net_value:
                best_net_value = net_value
                best_player_id = player_id
        
        # Debug output removed for clean operation
    
    except (TypeError, KeyError, ValueError):
        pass  # Continue to fallback
    
    # If no player found through optimization, fallback to highest marginal value
    if best_player_id is None:
        try:
            fallback_best_value = float('-inf')
            for player_id in available_players:
                if player_id not in player_cache['pos']:
                    continue
                    
                player = {
                    'id': player_id,
                    'pos': player_cache['pos'].get(player_id, ''),
                    'proj': player_cache['proj'].get(player_id, 0),
                    'name': player_cache['player_name'].get(player_id, 'Unknown')
                }
                
                if not player['pos'] or player['proj'] <= 0:
                    continue
                
                # Just use marginal value without opportunity cost
                marginal_value = calculate_marginal_starter_value(roster, player)
                
                if marginal_value > fallback_best_value:
                    fallback_best_value = marginal_value
                    best_player_id = player_id
        except:
            pass
    
    # Final fallback: highest projection
    if best_player_id is None and available_players:
        try:
            best_proj = 0
            for player_id in available_players:
                if player_id in player_cache['proj']:
                    proj = player_cache['proj'][player_id]
                    if proj > best_proj:
                        best_proj = proj
                        best_player_id = player_id
        except:
            pass
    
    return best_player_id


def calculate_bench_depth_report(final_roster: List[dict], starters: List[dict],
                               threshold_pct: float = 0.25) -> Dict[str, int]:
    """
    Calculate bench depth using 25% rule - REPORTING ONLY, not used for optimization.
    Quality bench = within 25% of starter's projected points.
    """
    if not starters:
        return {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0}
    
    # Get minimum starter projection by position
    starter_mins = {}
    starter_positions = defaultdict(list)
    
    for starter in starters:
        starter_positions[starter['pos']].append(starter['proj'])
    
    for pos, projections in starter_positions.items():
        starter_mins[pos] = min(projections) if projections else 0
    
    # Count quality bench players (within 25% threshold)
    bench_players = [p for p in final_roster if p not in starters]
    bench_depth = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0}
    
    for bench_player in bench_players:
        pos = bench_player['pos']
        if pos in starter_mins and starter_mins[pos] > 0:
            threshold = starter_mins[pos] * (1 - threshold_pct)  # 25% rule
            if bench_player['proj'] >= threshold:
                bench_depth[pos] += 1
    
    return bench_depth


def calculate_total_starter_points(roster: List[dict], starter_slots: Dict[str, int] = None) -> float:
    """Calculate total projected points from optimal starting lineup"""
    if starter_slots is None:
        starter_slots = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1}
    
    starters = get_optimal_starters(roster, starter_slots)
    return sum(p['proj'] for p in starters)


def get_starter_breakdown(roster: List[dict], starter_slots: Dict[str, int] = None) -> Dict[str, List[dict]]:
    """Get starters grouped by position for reporting"""
    if starter_slots is None:
        starter_slots = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1}
    
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
        breakdown[pos].sort(key=lambda p: p['proj'], reverse=True)
    
    return dict(breakdown)