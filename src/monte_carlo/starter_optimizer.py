"""
starter_optimizer.py - North Star Aligned Fantasy Draft Optimizer

NORTH STAR PRINCIPLE: "Does this increase expected starter lineup points under uncertainty, relative to waiting?"

Simplified and modularized MSG-OC decision framework:
- MSG (Marginal Starter Gain) = SS(with_candidate) - SS(without_candidate) 
- OC (Opportunity Cost) = SS(now) - SS(next_pick)
- Decision Score = E[MSG] - E[OC]

Usage:
    from starter_optimizer import pick_best_now
    result = pick_best_now(players, roster_state, current_pick, next_my_pick, league_config)
"""

import numpy as np
import time
import sys
import os

# Add project root to path for importing src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from .starter_core import compute_starter_sum, marginal_starter_gain, compute_opportunity_cost_ss
from .probability_models import ExpectedBestSimulator, create_probability_model
from .optimizer_utils import (
    load_players_from_csv, 
    compute_dynamic_replacement_levels,
    clear_dynamic_replacement_cache
)

# --------------------------
# Config & league defaults
# --------------------------
DEFAULT_LEAGUE = {
    "n_teams": 14,
    "starters_by_pos": {"RB": 2, "WR": 2, "TE": 1, "QB": 1, "FLEX": 1},
}

def pick_best_now(pool_players, roster_state, current_pick, next_my_pick, league=DEFAULT_LEAGUE,
                  top_k_candidates=20, scenarios=500, pick_prob_fn=None, probability_model=None, 
                  rng=None, my_team_idx=None, clear_cache_after=True):
    """
    North Star aligned decision function: MSG - OC framework.
    
    Score = E[Marginal Starter Gain] - E[Opportunity Cost]
    
    This directly answers: "Does this increase expected starter lineup points 
    under uncertainty, relative to waiting?"
    """
    start_time = time.time()
    rng = np.random.default_rng() if rng is None else rng
    
    # Create simulator with team context
    sim = ExpectedBestSimulator(pool_players, league=league, pick_prob_fn=pick_prob_fn, 
                               probability_model=probability_model, rng=rng, my_team_idx=my_team_idx)

    # Convert roster_state to player list
    roster_players = []
    for pos_list in roster_state.values():
        roster_players.extend(pos_list)
    
    # Performance optimization: adaptive scenario count
    n_scenarios = min(scenarios, max(100, len(pool_players)))
    starter_slots = league["starters_by_pos"]
    
    # Smart candidate filtering: prioritize core positions
    candidates = _filter_and_prioritize_candidates(
        pool_players, roster_players, starter_slots, league, top_k_candidates)
    
    # Pre-compute opportunity cost once (shared across all candidates)
    opportunity_cost = compute_opportunity_cost_ss(
        roster_players, pool_players, current_pick, next_my_pick, 
        sim, starter_slots, n_scenarios//2, league)
    
    # Evaluate candidates: MSG - OC
    best_choice, best_score, debug = _evaluate_candidates(
        candidates, roster_players, starter_slots, league, opportunity_cost, n_scenarios)
    
    # Performance monitoring and cleanup
    total_time = time.time() - start_time
    if total_time > 2.0:
        print(f"⚠️ Performance warning: {total_time:.2f}s exceeded 2s target")
    
    if clear_cache_after:
        sim.clear_cache()
        clear_dynamic_replacement_cache()
    
    return {
        "pick": best_choice,
        "score": best_score,
        "debug": sorted(debug, key=lambda x: x[4], reverse=True)[:10],
        "performance": {
            "total_time": total_time,
            "cache_hit_rate": sim.get_cache_stats()["hit_rate"],
            "cache_size": sim.get_cache_stats()["size"]
        },
        "north_star_metrics": {
            "expected_msg": debug[0][2] if debug else 0.0,
            "opportunity_cost": opportunity_cost,
            "decision_score": best_score
        }
    }


def _filter_and_prioritize_candidates(pool_players, roster_players, starter_slots, 
                                    league, top_k_candidates):
    """Filter and prioritize candidates using quick starter impact estimation."""
    core_candidates = []
    other_candidates = []
    
    for player in pool_players:
        if player["pos"] in ["QB", "RB", "WR", "TE"]:
            core_candidates.append(player)
        else:
            other_candidates.append(player)
    
    def quick_starter_impact(player):
        """Quick estimate of player's starter lineup impact"""
        impacts = []
        for i in range(min(20, len(player["samples"]))):
            impact = marginal_starter_gain(player, roster_players, i, starter_slots, league)
            impacts.append(impact)
        return np.mean(impacts) if impacts else 0.0
    
    core_sorted = sorted(core_candidates, key=quick_starter_impact, reverse=True)
    other_sorted = sorted(other_candidates, key=quick_starter_impact, reverse=True)
    
    return core_sorted[:top_k_candidates] + other_sorted[:max(5, top_k_candidates//4)]


def _evaluate_candidates(candidates, roster_players, starter_slots, league, 
                        opportunity_cost, n_scenarios):
    """Evaluate candidates using MSG - OC framework."""
    best_choice = None
    best_score = -1e9
    debug = []
    
    for cand in candidates:
        try:
            # Calculate MSG: expected improvement in starter sum
            msg_values = [
                marginal_starter_gain(cand, roster_players, i, starter_slots, league)
                for i in range(min(n_scenarios, 150))
            ]
            expected_msg = float(np.mean(msg_values)) if msg_values else 0.0
            
            # North Star decision score: MSG - OC
            score = expected_msg - opportunity_cost
            
            debug.append((cand["name"], cand["pos"], expected_msg, opportunity_cost, score))
            
            if score > best_score:
                best_score = score
                best_choice = cand
                
        except Exception as e:
            print(f"Warning: Error evaluating {cand.get('name', 'Unknown')}: {e}")
            continue
    
    # Fallback if no candidates were successfully evaluated
    if best_choice is None and candidates:
        best_choice = candidates[0]
        best_score = 0.0
        debug = [(best_choice["name"], best_choice["pos"], 0.0, 0.0, 0.0)]
    
    return best_choice, best_score, debug

# --------------------------
# Example test harness
# --------------------------
if __name__ == "__main__":
    # Example run using CSV file
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    players = load_players_from_csv(f"{base_path}/data/rankings_top300_20250814.csv", 
                                   proj_col="FANTASY_PTS", pos_col="POSITION", 
                                   id_col="id", envelope_pct=0.20, scenarios=600)
    
    prob_model = create_probability_model(base_path)
    roster = {"RB": [], "WR": [], "TE": [], "QB": []}
    
    # Snake draft logic for next_my_pick calculation
    my_team_idx = 4
    n_teams = 14
    current_pick = 29
    
    current_round = (current_pick - 1) // n_teams + 1
    pick_in_round = (current_pick - 1) % n_teams + 1
    
    if current_round % 2 == 1:  # Odd round
        if pick_in_round <= my_team_idx + 1:
            picks_until_my_turn = my_team_idx + 1 - pick_in_round
        else:
            picks_until_my_turn = (n_teams - pick_in_round) + 1 + (n_teams - my_team_idx - 1) + 1
    else:  # Even round
        if pick_in_round <= n_teams - my_team_idx:
            picks_until_my_turn = n_teams - my_team_idx - pick_in_round + 1
        else:
            picks_until_my_turn = (n_teams - pick_in_round) + 1 + my_team_idx + 1
    
    next_my_pick = current_pick + picks_until_my_turn
    
    print("\n" + "="*60)
    print("STARTER OPTIMIZER TEST - SIMPLIFIED MODULAR VERSION")
    print("="*60)
    print(f"Current pick: {current_pick}")
    print(f"Next pick: {next_my_pick}")
    print(f"Using probability model: {'Yes' if prob_model else 'No'}")
    print()
    
    result = pick_best_now(players, roster, current_pick, next_my_pick, 
                          league=DEFAULT_LEAGUE, top_k_candidates=30, scenarios=600,
                          probability_model=prob_model, my_team_idx=my_team_idx)
    
    print(f"🏆 Recommended: {result['pick']['name']} ({result['pick']['pos']}) - Score: {result['score']:.1f}")
    print("\nTop candidates:")
    for i, (name, pos, msg, oc, score) in enumerate(result["debug"][:5], 1):
        print(f"{i}. {name} ({pos}): {score:.1f} = {msg:.1f} MSG - {oc:.1f} OC")
    
    perf = result["performance"]
    print(f"\nPerformance: {perf['total_time']:.2f}s (target < 2s)")
    print(f"Cache hit rate: {perf['cache_hit_rate']:.1%}")
