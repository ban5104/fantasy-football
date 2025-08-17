"""
Depth Evaluation Module for Fantasy Football Draft Analysis
Evaluates bench quality and calculates depth value beyond starting lineup
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class DepthEvaluator:
    """Evaluates roster depth quality beyond starting lineup"""
    
    def __init__(self, projections_df: pd.DataFrame):
        """
        Initialize depth evaluator
        
        Args:
            projections_df: DataFrame with player projections and metadata
        """
        self.projections = projections_df
        self.thresholds = {
            'excellent': 0.15,  # ≤15% gap from starter
            'useful': 0.25,     # 15-25% gap
            'depth_only': 0.40  # >25% gap
        }
        
        # Expected injury rates by position
        self.injury_rates = {
            'RB': 0.20,  # Higher injury rate
            'WR': 0.15,  # Moderate injury rate
            'TE': 0.12,  # Lower injury rate
            'QB': 0.10,  # Lowest injury rate
        }
        
    def evaluate_depth(self, roster_players: List[dict], starters: List[dict]) -> Dict:
        """
        Calculate bench quality metrics
        
        Args:
            roster_players: List of all roster players with pos/proj
            starters: List of starting lineup players
            
        Returns:
            dict: excellent_backups, useful_backups, depth_value, 
                  injury_resilience, flex_options
        """
        # Get bench players (all roster minus starters)
        starter_names = {p['name'] for p in starters}
        bench = [p for p in roster_players if p['name'] not in starter_names]
        
        if not bench:
            return {
                'excellent_backups': 0,
                'useful_backups': 0,
                'depth_value': 0.0,
                'injury_resilience': 0.0,
                'flex_options': 0
            }
        
        # Group starters by position for comparison
        starter_projections = defaultdict(list)
        for player in starters:
            starter_projections[player['pos']].append(player['proj'])
            
        # Sort starter projections (highest first)
        for pos in starter_projections:
            starter_projections[pos].sort(reverse=True)
            
        # Calculate quality tiers
        excellent, useful = self._count_by_thresholds(bench, starter_projections)
        
        # Calculate total depth value
        # Weighted by position-specific injury rates
        depth_value = self._calculate_weighted_depth_value(bench, starter_projections)
        
        # Calculate injury resilience (ability to replace injured starters)
        injury_resilience = self._calculate_replacement_value(bench, starter_projections)
        
        # Count flex-worthy players (RB/WR/TE that could start)
        flex_options = self._count_flex_worthy(bench)
        
        return {
            'excellent_backups': excellent,
            'useful_backups': useful,
            'depth_value': depth_value,
            'injury_resilience': injury_resilience,
            'flex_options': flex_options
        }
        
    def _count_by_thresholds(self, bench: List[dict], 
                           starter_projections: Dict[str, List[float]]) -> Tuple[int, int]:
        """Count bench players by quality thresholds"""
        excellent = 0
        useful = 0
        
        for player in bench:
            pos = player['pos']
            proj = player['proj']
            
            # Skip positions we don't start (K, DST for now)
            if pos not in starter_projections or not starter_projections[pos]:
                continue
                
            # Compare to weakest starter at position
            weakest_starter = min(starter_projections[pos])
            
            if weakest_starter > 0:
                gap = (weakest_starter - proj) / weakest_starter
                
                if gap <= self.thresholds['excellent']:
                    excellent += 1
                elif gap <= self.thresholds['useful']:
                    useful += 1
                    
        return excellent, useful
        
    def _calculate_weighted_depth_value(self, bench: List[dict],
                                      starter_projections: Dict[str, List[float]]) -> float:
        """Calculate depth value weighted by injury risk and replacement value"""
        total_value = 0.0
        
        # Group bench by position
        bench_by_pos = defaultdict(list)
        for player in bench:
            bench_by_pos[player['pos']].append(player['proj'])
            
        # Sort bench projections (highest first)
        for pos in bench_by_pos:
            bench_by_pos[pos].sort(reverse=True)
            
        # Calculate value for each position
        for pos in ['RB', 'WR', 'TE', 'QB']:
            if pos not in starter_projections or not starter_projections[pos]:
                continue
                
            bench_players = bench_by_pos.get(pos, [])
            if not bench_players:
                continue
                
            # Get injury rate for position
            injury_rate = self.injury_rates.get(pos, 0.15)
            
            # Value = (bench_strength - replacement_level) * injury_probability
            starter_avg = np.mean(starter_projections[pos])
            bench_avg = np.mean(bench_players[:2])  # Top 2 bench players
            
            # Replacement level is 60% of starter level
            replacement_level = starter_avg * 0.6
            
            if bench_avg > replacement_level:
                position_value = (bench_avg - replacement_level) * injury_rate
                total_value += position_value
                
        return total_value
        
    def _calculate_replacement_value(self, bench: List[dict],
                                   starter_projections: Dict[str, List[float]]) -> float:
        """Calculate ability to replace injured starters"""
        if not bench:
            return 0.0
            
        replacement_scores = []
        
        # For each starting position, check best bench replacement
        for pos in ['RB', 'WR', 'TE', 'QB']:
            if pos not in starter_projections or not starter_projections[pos]:
                continue
                
            # Find best bench player at position
            best_bench = 0.0
            for player in bench:
                if player['pos'] == pos:
                    best_bench = max(best_bench, player['proj'])
                    
            if best_bench > 0:
                # Compare to average starter at position
                starter_avg = np.mean(starter_projections[pos])
                if starter_avg > 0:
                    replacement_ratio = best_bench / starter_avg
                    replacement_scores.append(replacement_ratio)
                    
        return np.mean(replacement_scores) if replacement_scores else 0.0
        
    def _count_flex_worthy(self, bench: List[dict]) -> int:
        """Count bench players who could reasonably start at FLEX"""
        flex_worthy = 0
        
        # Flexible positions that can play FLEX
        flex_positions = {'RB', 'WR', 'TE'}
        
        for player in bench:
            if player['pos'] in flex_positions:
                # Simple threshold: above 60% of position average
                # This would need more sophisticated logic in production
                if player['proj'] > 8.0:  # Rough flex-worthy threshold
                    flex_worthy += 1
                    
        return flex_worthy
        
    def analyze_depth_strategy(self, roster_players: List[dict]) -> Dict:
        """
        Analyze whether a roster follows depth-focused strategy
        
        Args:
            roster_players: Complete roster
            
        Returns:
            Analysis of depth vs. top-heavy strategy
        """
        # Group by position
        pos_groups = defaultdict(list)
        for player in roster_players:
            pos_groups[player['pos']].append(player['proj'])
            
        # Sort each position
        for pos in pos_groups:
            pos_groups[pos].sort(reverse=True)
            
        depth_indicators = {}
        
        # Analyze each position's depth
        for pos in ['RB', 'WR', 'TE']:
            players = pos_groups.get(pos, [])
            if len(players) < 2:
                depth_indicators[pos] = 'insufficient'
                continue
                
            # Calculate depth ratio (bench strength vs starter strength)
            if len(players) >= 3:
                starter_avg = np.mean(players[:2])  # Top 2
                bench_avg = np.mean(players[2:])    # Rest
                
                if starter_avg > 0:
                    depth_ratio = bench_avg / starter_avg
                    
                    if depth_ratio > 0.8:
                        depth_indicators[pos] = 'excellent_depth'
                    elif depth_ratio > 0.6:
                        depth_indicators[pos] = 'good_depth'
                    elif depth_ratio > 0.4:
                        depth_indicators[pos] = 'moderate_depth'
                    else:
                        depth_indicators[pos] = 'top_heavy'
                else:
                    depth_indicators[pos] = 'weak_overall'
            else:
                depth_indicators[pos] = 'minimal_depth'
                
        # Overall strategy classification
        depth_scores = []
        for pos, indicator in depth_indicators.items():
            if indicator == 'excellent_depth':
                depth_scores.append(3)
            elif indicator == 'good_depth':
                depth_scores.append(2)
            elif indicator == 'moderate_depth':
                depth_scores.append(1)
            else:
                depth_scores.append(0)
                
        avg_depth = np.mean(depth_scores) if depth_scores else 0
        
        if avg_depth >= 2.0:
            strategy_type = 'depth_focused'
        elif avg_depth >= 1.0:
            strategy_type = 'balanced_depth'
        else:
            strategy_type = 'top_heavy'
            
        return {
            'strategy_type': strategy_type,
            'avg_depth_score': avg_depth,
            'position_analysis': depth_indicators,
            'total_players': len(roster_players)
        }