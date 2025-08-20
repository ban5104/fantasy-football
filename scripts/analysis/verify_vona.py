#!/usr/bin/env python3
"""Verify VONA calculations are working correctly"""

from src.monte_carlo import DraftSimulator
from src.monte_carlo.strategies import VOR_POLICIES

# Create simulator
sim = DraftSimulator(n_rounds=14)
params = VOR_POLICIES['balanced']['params']

# Test scenario: Round 3, Pick 33
roster = ['Justin Jefferson', 'Travis Etienne Jr.']
drafted = set(['CeeDee Lamb', 'Christian McCaffrey', 'Tyreek Hill', 'Ja\'Marr Chase'])

print("=" * 60)
print("VONA VERIFICATION - Round 3, Pick 33")
print("=" * 60)
print(f"Current roster: {roster}")
print()

# Get top available players
candidates = sim.simulator.get_top_available_players(roster, drafted, k=5)
print("Top 5 available players by VOR:")
for i, c in enumerate(candidates, 1):
    print(f"  {i}. {c['name']} ({c['pos']}): VOR={c['vor']:.1f}, Proj={c['proj']:.1f}")
print()

# Test VONA for each
print("VONA Analysis (14 rounds total):")
for candidate in candidates[:3]:
    vona = sim.simulator.simulate_vona_comparison(
        4, params, 32, candidate['name'],
        seed=42, initial_roster=roster, already_drafted=drafted
    )
    print(f"  {candidate['name']:20s}: VONA={vona:+7.1f} points")

print()
print("Interpretation:")
print("  Positive VONA = Better than best alternative over full draft")
print("  Negative VONA = Worse than best alternative over full draft")
print("  Large values normal for 14-round simulation (cumulative effect)")
print()

# Now test Round 1 for comparison
print("=" * 60)
print("VONA VERIFICATION - Round 1, Pick 5 (no roster)")
print("=" * 60)

candidates_r1 = sim.simulator.get_top_available_players([], [], k=5)
print("Top 5 available players by VOR:")
for i, c in enumerate(candidates_r1, 1):
    print(f"  {i}. {c['name']} ({c['pos']}): VOR={c['vor']:.1f}")

print("\nVONA Analysis (14 rounds total):")
for candidate in candidates_r1[:3]:
    vona = sim.simulator.simulate_vona_comparison(
        4, params, 4, candidate['name'],
        seed=42, initial_roster=[], already_drafted=[]
    )
    print(f"  {candidate['name']:20s}: VONA={vona:+7.1f} points")