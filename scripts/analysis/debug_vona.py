#!/usr/bin/env python3
"""Debug script to understand VONA calculations"""

from src.monte_carlo import DraftSimulator
from src.monte_carlo.strategies import VOR_POLICIES

# Create simulator
sim = DraftSimulator(n_rounds=14)
params = VOR_POLICIES['balanced']['params']

# Test scenario: Round 3, Pick 33
roster = ['Justin Jefferson', 'Travis Etienne Jr.']
drafted = set(['CeeDee Lamb', 'Christian McCaffrey', 'Tyreek Hill', 'Ja\'Marr Chase'])

print("Testing VONA comparison method...")
print(f"Initial roster: {roster}")
print(f"Already drafted: {len(drafted)} players")
print()

# Test the two branches separately
print("Branch A: With target player (Breece Hall)")
value_with_target = sim.simulator.simulate_from_pick(
    4, params, 32, forced_pick='Breece Hall',
    seed=42, initial_roster=roster, already_drafted=drafted
)
print(f"  Value with Breece Hall: {value_with_target:.1f}")

print("\nBranch B: Without forced pick (best available)")
value_with_alternative = sim.simulator.simulate_from_pick(
    4, params, 32, forced_pick=None,
    seed=43, initial_roster=roster, already_drafted=drafted
)
print(f"  Value with best alternative: {value_with_alternative:.1f}")

print(f"\nVONA (difference): {value_with_target - value_with_alternative:.1f}")
print()

# Now test without any roster context for comparison
print("Testing WITHOUT roster context (wrong way):")
value_no_context = sim.simulator.simulate_from_pick(
    4, params, 32, forced_pick='Breece Hall',
    seed=42, initial_roster=None, already_drafted=None
)
print(f"  Value with Breece (no context): {value_no_context:.1f}")

# Test regular draft simulation to understand scale
print("\nReference: Complete draft from start:")
result = sim.simulator.simulate_single_draft(4, params, seed=42)
print(f"  Full draft value: {result['roster_value']:.1f}")
print(f"  Number of players: {result['num_players']}")