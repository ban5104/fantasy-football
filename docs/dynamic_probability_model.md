# Dynamic Probability Model for Fantasy Draft Simulation

## Overview
Real drafters don't follow static probabilities. Their choices dynamically adapt based on roster composition, position runs, and evolving draft context.

## Core Dynamic Factors

### 1. Roster Saturation Penalty
Teams become less likely to draft positions they've already filled.
```python
def roster_saturation_penalty(team_roster, player_position):
    position_count = count_position(team_roster, player_position)
    if position_count >= 2:
        return 0.3  # 70% less likely to draft 3rd at position
    elif position_count == 1:
        return 0.9  # Slightly less likely for 2nd
    else:
        return 1.0
```

### 2. Position Run Momentum (FOMO Effect)
When multiple players at a position go quickly, others panic-draft that position.
```python
def position_run_multiplier(recent_picks, player_position):
    position_run_count = count_position(recent_picks[-5:], player_position)
    if position_run_count >= 3:
        return 1.5  # FOMO effect kicks in
    return 1.0
```

### 3. Scarcity Premium
As top players at a position disappear, remaining ones become more valuable.
```python
def scarcity_multiplier(player_position, available_players):
    top_10_remaining = count_top_remaining(player_position, available_players)
    if top_10_remaining <= 3:
        return 1.8  # "Last elite player available!"
    elif top_10_remaining <= 5:
        return 1.3
    return 1.0
```

### 4. Round-Based Strategy Evolution
Draft strategy changes as rounds progress.
```python
def round_strategy_weight(current_round):
    if current_round <= 3:
        return 0.2  # Early: mostly best available
    elif current_round <= 8:
        return 0.6  # Middle: balance value and need
    else:
        return 0.8  # Late: fill remaining needs
```

### 5. ESPN Interface Bias
Players visible at top of queue get selection boost.
```python
def visibility_boost(player_espn_rank):
    if player_espn_rank <= 5:
        return 1.2  # Top of queue visibility
    return 1.0
```

## Full Dynamic Probability Model

```python
def dynamic_pick_probability(player, team_roster, all_rosters, current_round, available_players):
    """
    Calculate dynamic probability of a player being selected.
    
    Args:
        player: Player object with position, rankings
        team_roster: Current roster of selecting team
        all_rosters: All team rosters (for position run detection)
        current_round: Current draft round
        available_players: Remaining player pool
    
    Returns:
        float: Adjusted probability of selection
    """
    # Base probability (80% ESPN + 20% ADP)
    base_prob = 0.8 * espn_probability(player) + 0.2 * adp_probability(player)
    
    # Apply all dynamic factors
    final_prob = (base_prob * 
                  roster_saturation_penalty(team_roster, player.position) *
                  position_run_multiplier(all_rosters, player.position) *
                  scarcity_multiplier(player.position, available_players) *
                  round_need_adjustment(team_roster, player.position, current_round) *
                  visibility_boost(player.espn_rank))
    
    return normalize_probability(final_prob)
```

## Additional Behavioral Factors

### Tier Break Awareness
```python
# Players spike in value when they're last in their tier
if is_last_in_tier(player):
    tier_break_multiplier = 2.0
```

### Positional Queue Depletion
```python
# As positions get drafted, remaining players become more valuable
depletion_rate = positions_drafted / total_startable
if depletion_rate > 0.5:
    depletion_multiplier = 1 + depletion_rate
```

### Handcuff Correlation
```python
# Backup RBs more likely if team has starter
if has_starter_rb(team_roster, player.team):
    handcuff_multiplier = 1.5
```

### Stack Building
```python
# WRs more likely if team has their QB
if has_qb_from_team(team_roster, player.team):
    stack_multiplier = 1.3
```

## Calibration Against Historical Data

The model parameters can be learned from actual draft data:

```python
historical_params = optimize_against_real_drafts({
    'saturation_penalty': [0.2, 0.3, 0.4],
    'run_multiplier': [1.3, 1.5, 1.8],
    'scarcity_multiplier': [1.5, 1.8, 2.2],
    'need_weight_by_round': {
        'early': [0.1, 0.2, 0.3],
        'middle': [0.5, 0.6, 0.7],
        'late': [0.7, 0.8, 0.9]
    }
})
```

## Implementation Priority

### MVP (Basic Static Model)
- 80% ESPN + 20% ADP weighting
- Simple softmax probability
- No dynamic adjustments

### Version 2 (Roster-Aware)
- Add roster saturation penalty
- Basic position need calculation
- Round-based strategy shifts

### Version 3 (Market-Aware)
- Position run detection
- Scarcity premiums
- Tier break reactions

### Version 4 (Full Behavioral)
- Handcuff correlations
- Stack building
- Historical calibration

## Key Insight
The difference between beating a simple algorithm and beating actual human drafters is modeling how humans actually behave - considering needs, reacting to runs, panicking at tier breaks, and evolving strategy throughout the draft.