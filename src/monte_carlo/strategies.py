"""
Draft Strategy Definitions for Fantasy Football
Pure data - no logic, just strategy configurations
"""

def generate_scenario_configs(base_strategy='balanced'):
    """
    Generate multiple scenario configurations by varying key parameters
    
    Returns list of strategy configs with different trade-off parameters
    """
    scenarios = []
    
    # Base parameters from selected strategy
    if base_strategy in VOR_POLICIES:
        base_params = VOR_POLICIES[base_strategy]['params'].copy()
        is_vor = True
    else:
        base_params = STRATEGIES[base_strategy]['multipliers'].copy()
        is_vor = False
    
    # Define parameter variations for scenarios
    variations = {
        'scenario_1_max_starters': {
            'name': 'Max Starters',
            'risk_aversion': 0.0,
            'bench_value_decay': 0.1,  # Bench worth 10% of starters
            'tier_cliff_penalty': 1.0,
            'position_importance': {'RB': 1.3, 'WR': 1.2, 'TE': 1.0, 'QB': 0.8},
            'bench_phase': {
                'handcuff_weight': 0.1,
                'quality_weight': 0.2,
                'upside_weight': 0.1,
                'coverage_weight': 0.1
            }
        },
        'scenario_2_balanced': {
            'name': 'Balanced',
            'risk_aversion': 0.5,
            'bench_value_decay': 0.3,  # Bench worth 30% of starters
            'tier_cliff_penalty': 1.5,
            'position_importance': {'RB': 1.2, 'WR': 1.1, 'TE': 1.0, 'QB': 0.9},
            'bench_phase': {
                'handcuff_weight': 0.3,
                'quality_weight': 0.5,
                'upside_weight': 0.2,
                'coverage_weight': 0.3
            }
        },
        'scenario_3_conservative': {
            'name': 'Conservative Depth',
            'risk_aversion': 0.8,
            'bench_value_decay': 0.5,  # Bench worth 50% of starters
            'tier_cliff_penalty': 2.0,
            'position_importance': {'RB': 1.1, 'WR': 1.1, 'TE': 1.1, 'QB': 1.0},
            'bench_phase': {
                'handcuff_weight': 0.5,
                'quality_weight': 0.7,
                'upside_weight': 0.1,
                'coverage_weight': 0.5
            }
        },
        'scenario_4_rb_heavy': {
            'name': 'RB Priority',
            'risk_aversion': 0.3,
            'bench_value_decay': 0.2,
            'tier_cliff_penalty': 2.5,  # Urgent when RB cliff approaches
            'position_importance': {'RB': 1.5, 'WR': 1.0, 'TE': 0.9, 'QB': 0.8},
            'bench_phase': {
                'handcuff_weight': 0.6,  # High handcuff priority
                'quality_weight': 0.4,
                'upside_weight': 0.2,
                'coverage_weight': 0.3
            }
        },
        'scenario_5_wr_heavy': {
            'name': 'WR Priority',
            'risk_aversion': 0.3,
            'bench_value_decay': 0.2,
            'tier_cliff_penalty': 2.5,
            'position_importance': {'RB': 1.0, 'WR': 1.5, 'TE': 1.0, 'QB': 0.8},
            'bench_phase': {
                'handcuff_weight': 0.2,
                'quality_weight': 0.4,
                'upside_weight': 0.4,  # WRs have more upside
                'coverage_weight': 0.3
            }
        },
        'scenario_6_upside': {
            'name': 'Upside Chaser',
            'risk_aversion': 0.1,
            'bench_value_decay': 0.2,
            'tier_cliff_penalty': 1.2,
            'position_importance': {'RB': 1.2, 'WR': 1.2, 'TE': 1.1, 'QB': 0.9},
            'bench_phase': {
                'handcuff_weight': 0.2,
                'quality_weight': 0.3,
                'upside_weight': 0.6,  # High upside focus
                'coverage_weight': 0.2
            }
        },
        'scenario_7_safe': {
            'name': 'Safe Floor',
            'risk_aversion': 0.9,
            'bench_value_decay': 0.4,
            'tier_cliff_penalty': 3.0,  # Very urgent at tier breaks
            'position_importance': {'RB': 1.1, 'WR': 1.1, 'TE': 1.0, 'QB': 1.1},
            'bench_phase': {
                'handcuff_weight': 0.4,
                'quality_weight': 0.8,  # Focus on quality depth
                'upside_weight': 0.0,
                'coverage_weight': 0.6
            }
        },
        'scenario_8_anti_fragile': {
            'name': 'Anti-Fragile',
            'risk_aversion': 0.6,
            'bench_value_decay': 0.6,  # High bench value
            'tier_cliff_penalty': 1.8,
            'position_importance': {'RB': 1.2, 'WR': 1.2, 'TE': 1.0, 'QB': 1.0},
            'bench_phase': {
                'handcuff_weight': 0.5,
                'quality_weight': 0.6,
                'upside_weight': 0.3,
                'coverage_weight': 0.7  # Maximum coverage
            }
        },
        'scenario_9_tier_focused': {
            'name': 'Tier Focused',
            'risk_aversion': 0.4,
            'bench_value_decay': 0.25,
            'tier_cliff_penalty': 4.0,  # Extreme tier urgency
            'position_importance': {'RB': 1.3, 'WR': 1.2, 'TE': 0.9, 'QB': 0.7},
            'bench_phase': {
                'handcuff_weight': 0.3,
                'quality_weight': 0.6,
                'upside_weight': 0.2,
                'coverage_weight': 0.3
            }
        },
        'scenario_10_adaptive': {
            'name': 'Adaptive',
            'risk_aversion': 0.5,
            'bench_value_decay': 0.35,
            'tier_cliff_penalty': 1.5 + (0.5 * base_params.get('alpha', 10) / 10),  # Adaptive to base
            'position_importance': {'RB': 1.25, 'WR': 1.15, 'TE': 1.05, 'QB': 0.95},
            'bench_phase': {
                'handcuff_weight': 0.4,
                'quality_weight': 0.5,
                'upside_weight': 0.3,
                'coverage_weight': 0.4
            }
        }
    }
    
    # Generate scenario configurations
    for scenario_id, params in variations.items():
        scenario_config = {
            'scenario_id': scenario_id,
            'name': params['name'],
            'risk_aversion': params['risk_aversion'],
            'bench_value_decay': params['bench_value_decay'],
            'tier_cliff_penalty': params['tier_cliff_penalty'],
            'position_importance': params['position_importance'],
            'bench_phase': params['bench_phase']
        }
        
        # Merge with base strategy parameters
        if is_vor:
            # Adjust VOR parameters based on scenario
            scenario_config['params'] = base_params.copy()
            scenario_config['params']['alpha'] *= (1 + params['risk_aversion'])
            scenario_config['params']['gamma'] = 0.5 + (0.5 * params['risk_aversion'])
            scenario_config['params']['bench_phase'] = params['bench_phase']
        else:
            # Adjust multipliers based on scenario
            adjusted_multipliers = {}
            for pos, mult in base_params.items():
                importance = params['position_importance'].get(pos, 1.0)
                adjusted_multipliers[pos] = mult * importance
            scenario_config['multipliers'] = adjusted_multipliers
            scenario_config['bench_phase'] = params['bench_phase']
        
        scenarios.append(scenario_config)
    
    return scenarios

# Position value multipliers (based on scarcity and importance)
POSITION_VALUES = {
    'QB': 1.0,   # QBs score more but are replaceable
    'RB': 1.2,   # RB scarcity premium
    'WR': 1.1,   # WR depth but need quality
    'TE': 1.15,  # Elite TE advantage
    'K': 0.5,    # Kickers don't matter in first 7 rounds
    'DST': 0.6,  # DST don't matter in first 7 rounds
}

# VOR Policy Definitions - New VOR-driven approach
VOR_POLICIES = {
    "conservative": {
        'name': 'Conservative VOR',
        'description': 'Risk-averse approach with high survival weight',
        'params': {
            "alpha": 10,     # Strong scarcity bonus (was 4)
            "lambda": 0,     # Neutral risk preference (was 2)
            "gamma": 0.95,   # Very high survival probability weight (was 0.6)
            "r_te": 10,      # Very late TE punt round (was 9)
            "delta_qb": 0    # No QB threshold
        }
    },
    "shadow_conservative": {
        'name': 'Shadow Conservative VOR',
        'description': 'Conservative VOR with RB shadow pricing in early rounds',
        'params': {
            "alpha": 10,
            "lambda": 0,
            "gamma": 0.95,
            "r_te": 10,
            "delta_qb": 0,
            "rb_shadow": 15,    # Shadow price bonus for RBs in early rounds
            "shadow_decay_round": 3  # Stop shadow pricing after this round
        }
    },
    "balanced": {
        'name': 'Balanced VOR',
        'description': 'Moderate approach balancing risk and opportunity',
        'params': {
            "alpha": 12,     # Reduced scarcity bonus to avoid overvaluing RB/WR depth
            "lambda": -1,    # Moderate upside seeking 
            "gamma": 0.85,   # Reduced survival weight to allow earlier QB
            "r_te": 9,       # Later TE punt round
            "delta_qb": -2   # Small QB bonus for elite QBs (negative = bonus)
        }
    },
    "shadow_balanced": {
        'name': 'Shadow Balanced VOR',
        'description': 'Balanced VOR with RB shadow pricing to enforce early RB timing',
        'params': {
            "alpha": 18,
            "lambda": -2,
            "gamma": 0.92,
            "r_te": 9,
            "delta_qb": 0,
            "rb_shadow": 20,    # Shadow price bonus for RBs in early rounds
            "shadow_decay_round": 3  # Stop shadow pricing after this round
        }
    },
    "constraint_balanced": {
        'name': 'Constraint Balanced VOR',
        'description': 'Balanced VOR with chance constraint for 2+ RBs by Round 2',
        'params': {
            "alpha": 18,
            "lambda": -2,
            "gamma": 0.92,
            "r_te": 9,
            "delta_qb": 0,
            "constraint_threshold": 0.75,  # P(≥2 RBs by R2) ≥ 75%
            "constraint_target": {"RB": 2, "round": 2}  # 2 RBs by Round 2
        }
    },
    "aggressive": {
        'name': 'Aggressive VOR', 
        'description': 'Ultra-aggressive tuning to beat RB Heavy baseline',
        'params': {
            "alpha": 85,     # Unprecedented scarcity bonus (was 50)
            "lambda": -9,    # Extreme variance seeking (was -5)
            "gamma": 1.0,    # Maximum survival weight
            "r_te": 17,      # TE doesn't exist (was 13)
            "delta_qb": -8   # Strong QB avoidance (was 0)
        }
    },
    "rb_focused": {
        'name': 'RB Focused VOR',
        'description': 'Maximum RB scarcity bonus to exceed RB Heavy performance',
        'params': {
            "alpha": 70,     # EXTREME scarcity bonus (was 45)
            "lambda": -7,    # Maximum upside seeking (was -4)
            "gamma": 1.0,    # Complete survival focus (was 0.99)
            "r_te": 15,      # Never consider TE (was 12)
            "delta_qb": -5   # Actively avoid QB early (was 0)
        }
    },
    "early_value": {
        'name': 'Early Value VOR',
        'description': 'Balanced optimization targeting high-end outcomes',
        'params': {
            "alpha": 32,     # Higher scarcity bonus (was 22)
            "lambda": -2,    # More upside seeking (was -1)
            "gamma": 0.94,   # Higher survival weighting (was 0.90)
            "r_te": 10,      # Very late TE punt
            "delta_qb": 0    # No QB threshold
        }
    },
    "nuclear": {
        'name': 'Nuclear VOR',
        'description': 'Absolute maximum parameters to beat RB Heavy at any cost',
        'params': {
            "alpha": 100,    # Maximum possible scarcity
            "lambda": -10,   # Maximum variance chasing
            "gamma": 1.0,    # Complete survival focus
            "r_te": 20,      # TE banned completely
            "delta_qb": -15  # Extreme QB avoidance
        }
    },
    "rb_heavy_vor": {
        'name': 'RB Heavy VOR',
        'description': 'VOR system tuned to mimic RB Heavy behavior exactly',
        'params': {
            "alpha": 35,     # Strong RB scarcity focus
            "lambda": -2,    # Moderate upside seeking
            "gamma": 0.95,   # High survival probability
            "r_te": 8,       # Allow some TE consideration like RB Heavy
            "delta_qb": 0    # Ignore QB early like RB Heavy
        }
    }
}

# Legacy strategy definitions (maintained for backward compatibility)
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
        },
        'bench_phase': {  # Rounds 8-14 parameters
            'handcuff_weight': 0.3,   # Bonus for RB handcuffs
            'quality_weight': 0.5,    # Bonus for quality backups (within 25 pts)
            'upside_weight': 0.2,     # Bonus for high-variance players
            'coverage_weight': 0.3    # Bonus for bye week coverage
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
    },
    
    'starter_max': {
        'name': 'Starter Maximization',
        'description': 'Pure starter points maximization (no VOR calculations)',
        'multipliers': {
            'RB': 1.0,
            'WR': 1.0,
            'TE': 1.0,
            'QB': 1.0,
            'K': 0.8,  # Slight penalty for late-round positions
            'DST': 0.8
        },
        'bench_phase': {  # Simple bench strategy when depth is needed
            'handcuff_weight': 0.2,
            'quality_weight': 0.6,  # Focus on quality over everything else
            'upside_weight': 0.3,
            'coverage_weight': 0.2
        }
    },
    
    'starter_optimize': {
        'name': 'Starter Optimizer',
        'description': 'Marginal value optimization using starter-aware scenario sampling',
        'use_starter_optimizer': True,
        'bench_phase': {  # Simple bench strategy when depth is needed
            'handcuff_weight': 0.3,
            'quality_weight': 0.5,
            'upside_weight': 0.3,
            'coverage_weight': 0.3
        }
    }
}

# Roster requirements for value calculation
# NOTE: FLEX is NOT a draft position - it's filled automatically by best remaining RB/WR/TE
ROSTER_REQUIREMENTS = {
    'QB': 1,
    'RB': 2,
    'WR': 2,
    'TE': 1,
    'K': 1,
    'DST': 1
    # FLEX is handled automatically in lineup optimization, NOT draft strategy
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
    7: ['RB', 'WR', 'TE', 'QB'],          # Continue normal positions
    8: ['RB', 'WR', 'TE', 'QB'],
    9: ['RB', 'WR', 'TE', 'QB'],
    10: ['RB', 'WR', 'TE', 'QB'],
    11: ['RB', 'WR', 'TE', 'QB'],
    12: ['RB', 'WR', 'TE', 'QB'],
    13: ['RB', 'WR', 'TE', 'QB'],
    14: ['RB', 'WR', 'TE', 'QB']          # All positions valid through end
}

def get_strategy(name: str) -> dict:
    """Get strategy configuration by name"""
    return STRATEGIES.get(name, STRATEGIES['balanced'])

def get_vor_policy(name: str) -> dict:
    """Get VOR policy configuration by name"""
    return VOR_POLICIES.get(name, VOR_POLICIES['balanced'])

def list_strategies() -> list:
    """List all available strategy names"""
    return list(STRATEGIES.keys())

def list_vor_policies() -> list:
    """List all available VOR policy names"""
    return list(VOR_POLICIES.keys())

def is_vor_policy(name: str) -> bool:
    """Check if name refers to a VOR policy"""
    return name in VOR_POLICIES