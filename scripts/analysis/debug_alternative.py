#!/usr/bin/env python3
"""Debug what alternative is being selected in VONA comparison"""

from src.monte_carlo import DraftSimulator
from src.monte_carlo.strategies import VOR_POLICIES
import numpy as np

# Create simulator
sim = DraftSimulator(n_rounds=14)
params = VOR_POLICIES['balanced']['params']

# Monkey-patch to see what's being selected
original_select = sim.simulator.select_best_player

def debug_select_best_player(available, my_roster, strategy_params, round_num, recent_picks, replacement_levels):
    result = original_select(available, my_roster, strategy_params, round_num, recent_picks, replacement_levels)
    if result and round_num == 3:  # Only log Round 3
        player_name = sim.simulator.prob_model.players_df.loc[result, 'player_name']
        print(f"    Selected: {player_name}")
    return result

sim.simulator.select_best_player = debug_select_best_player

# Test scenario
roster = ['Justin Jefferson', 'Travis Etienne Jr.']
drafted = set(['CeeDee Lamb', 'Christian McCaffrey', 'Tyreek Hill', 'Ja\'Marr Chase'])

print("Testing VONA branches...")
print()

print("Branch A: Forced pick = Saquon Barkley")
np.random.seed(42)
value_a = sim.simulator.simulate_from_pick(
    4, params, 32, forced_pick='Saquon Barkley',
    seed=42, initial_roster=roster.copy(), already_drafted=drafted.copy()
)

print(f"\nBranch B: Best alternative (no forced pick)")
np.random.seed(42)
value_b = sim.simulator.simulate_from_pick(
    4, params, 32, forced_pick=None,
    seed=42, initial_roster=roster.copy(), already_drafted=drafted.copy()
)

print(f"\nVONA = {value_a:.1f} - {value_b:.1f} = {value_a - value_b:.1f}")