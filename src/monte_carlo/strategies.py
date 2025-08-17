"""
Draft Strategy Definitions for Fantasy Football
Pure data - no logic, just strategy configurations
"""

# Position value multipliers (based on scarcity and importance)
POSITION_VALUES = {
    'QB': 1.0,   # QBs score more but are replaceable
    'RB': 1.2,   # RB scarcity premium
    'WR': 1.1,   # WR depth but need quality
    'TE': 1.15,  # Elite TE advantage
    'K': 0.5,    # Kickers don't matter in first 7 rounds
    'DST': 0.6,  # DST don't matter in first 7 rounds
}

# Draft strategy definitions
# Each strategy has position-specific multipliers
STRATEGIES = {
    'balanced': {
        'name': 'Balanced',
        'description': 'Equal weight to all positions based on value',
        'multipliers': {
            'RB': 1.0,
            'WR': 1.0,
            'TE': 1.0,
            'QB': 1.0,
            'K': 1.0,
            'DST': 1.0
        }
    },
    
    'zero_rb': {
        'name': 'Zero RB',
        'description': 'Prioritize WR/TE early, get RBs late',
        'multipliers': {
            'RB': 0.4,
            'WR': 1.4,
            'TE': 1.2,
            'QB': 1.1,
            'K': 1.0,
            'DST': 1.0
        }
    },
    
    'rb_heavy': {
        'name': 'RB Heavy',
        'description': 'Load up on RBs early for scarcity',
        'multipliers': {
            'RB': 1.6,
            'WR': 0.8,
            'TE': 0.9,
            'QB': 0.9,
            'K': 1.0,
            'DST': 1.0
        }
    },
    
    'hero_rb': {
        'name': 'Hero RB',
        'description': 'One elite RB, then WR/TE focus',
        'multipliers': {
            'RB': 1.3,
            'WR': 1.1,
            'TE': 1.0,
            'QB': 0.8,
            'K': 1.0,
            'DST': 1.0
        }
    },
    
    'elite_qb': {
        'name': 'Elite QB',
        'description': 'Prioritize top-tier QB early',
        'multipliers': {
            'RB': 0.9,
            'WR': 0.9,
            'TE': 0.9,
            'QB': 1.5,
            'K': 1.0,
            'DST': 1.0
        }
    }
}

# Roster requirements for value calculation
ROSTER_REQUIREMENTS = {
    'QB': 1,
    'RB': 2,
    'WR': 2,
    'TE': 1,
    'FLEX': 1,  # Best remaining RB/WR/TE
    'K': 1,
    'DST': 1
}

# Position limits (don't draft too many)
POSITION_LIMITS = {
    'QB': 2,
    'RB': 5,
    'WR': 5,
    'TE': 2,
    'K': 1,
    'DST': 1
}

# Round-based position validity
# Which positions are reasonable to draft in each round
ROUND_POSITION_VALIDITY = {
    1: ['RB', 'WR'],                       # Only RB/WR in round 1
    2: ['RB', 'WR', 'TE'],                # Can add elite TE
    3: ['RB', 'WR', 'TE', 'QB'],          # Can add elite QB
    4: ['RB', 'WR', 'TE', 'QB'],
    5: ['RB', 'WR', 'TE', 'QB'],
    6: ['RB', 'WR', 'TE', 'QB'],
    7: ['RB', 'WR', 'TE', 'QB', 'K', 'DST']  # K/DST only in last round
}

def get_strategy(name: str) -> dict:
    """Get strategy configuration by name"""
    return STRATEGIES.get(name, STRATEGIES['balanced'])

def list_strategies() -> list:
    """List all available strategy names"""
    return list(STRATEGIES.keys())