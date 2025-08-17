"""
Pattern Detection Module for Fantasy Football Draft Analysis
Discovers emergent strategies from simulation results without predetermined patterns
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter


class PatternDetector:
    """Detects natural draft patterns without predetermined strategies"""
    
    def __init__(self):
        """Initialize pattern detector"""
        self.pattern_definitions = {
            # Early position patterns (rounds 1-4)
            'rb_heavy_start': {
                'rb_in_first_3': 2,
                'description': 'Prioritizes RB early for scarcity'
            },
            'zero_rb_start': {
                'rb_in_first_4': 0,
                'description': 'Avoids RB in early rounds'
            },
            'early_qb': {
                'qb_by_round': 5,
                'description': 'Takes QB earlier than typical'
            },
            
            # Mid-draft patterns (rounds 5-8)
            'depth_loading': {
                'rb_wr_in_middle': 3,
                'description': 'Loads up on RB/WR depth in middle rounds'
            },
            'fill_starters': {
                'unique_positions': 6,
                'description': 'Prioritizes filling starting positions'
            },
            
            # Late position patterns (rounds 9+)
            'punt_te': {
                'te_after_round': 10,
                'description': 'Waits very late for TE'
            },
            'punt_qb': {
                'qb_after_round': 8,
                'description': 'Waits late for QB'
            },
            
            # Overall patterns
            'balanced': {
                'description': 'No clear positional bias - even distribution'
            }
        }
        
    def detect_pattern(self, draft_history: List[Tuple[int, str, str]]) -> str:
        """
        Analyze draft history to classify emergent strategy
        
        Args:
            draft_history: List of (round, player_name, position) tuples
            
        Returns:
            str: Pattern name (e.g., 'depth_loading', 'fill_starters', 'balanced')
        """
        if not draft_history:
            return 'unknown'
            
        # Extract positions by round
        positions_by_round = {}
        position_counts = defaultdict(int)
        
        for round_num, player_name, position in draft_history:
            positions_by_round[round_num] = position
            position_counts[position] += 1
            
        max_round = max(positions_by_round.keys()) if positions_by_round else 0
        
        # Analyze early rounds (1-4)
        early_pattern = self._analyze_early_rounds(positions_by_round)
        if early_pattern != 'balanced':
            return early_pattern
            
        # Analyze middle rounds (5-8) if we have enough rounds
        if max_round >= 6:
            middle_pattern = self._analyze_middle_rounds(positions_by_round)
            if middle_pattern != 'balanced':
                return middle_pattern
                
        # Analyze late round patterns (9+) if we have enough rounds
        if max_round >= 9:
            late_pattern = self._analyze_late_rounds(positions_by_round)
            if late_pattern != 'balanced':
                return late_pattern
                
        # Check overall position distribution
        return self._analyze_overall_pattern(position_counts, max_round)
        
    def _analyze_early_rounds(self, positions_by_round: Dict[int, str]) -> str:
        """Analyze rounds 1-4 for early draft patterns"""
        early_rounds = [positions_by_round.get(r, '') for r in range(1, 5)]
        
        # Count RBs in first 3 rounds
        rb_count_early = sum(1 for pos in early_rounds[:3] if pos == 'RB')
        
        if rb_count_early >= 2:
            return 'rb_heavy_start'
            
        # Check for zero-RB approach
        rb_count_first_4 = sum(1 for pos in early_rounds if pos == 'RB')
        if rb_count_first_4 == 0:
            return 'zero_rb_start'
            
        # Check for early QB
        qb_round = self._get_position_round(positions_by_round, 'QB')
        if qb_round and qb_round <= 5:
            return 'early_qb'
            
        return 'balanced'
        
    def _analyze_middle_rounds(self, positions_by_round: Dict[int, str]) -> str:
        """Analyze rounds 5-8 for middle draft patterns"""
        middle_rounds = [positions_by_round.get(r, '') for r in range(5, 9)]
        
        # Count RB/WR in middle rounds
        rb_wr_count = sum(1 for pos in middle_rounds if pos in ['RB', 'WR'])
        
        if rb_wr_count >= 3:
            return 'depth_loading'
            
        # Count unique positions through round 8
        all_positions = set()
        for r in range(1, 9):
            pos = positions_by_round.get(r, '')
            if pos:
                all_positions.add(pos)
                
        if len(all_positions) >= 6:
            return 'fill_starters'
            
        return 'balanced'
        
    def _analyze_late_rounds(self, positions_by_round: Dict[int, str]) -> str:
        """Analyze rounds 9+ for late draft patterns"""
        # Check for position punting
        te_round = self._get_position_round(positions_by_round, 'TE')
        if te_round and te_round >= 10:
            return 'punt_te'
            
        qb_round = self._get_position_round(positions_by_round, 'QB')
        if qb_round and qb_round >= 9:
            return 'punt_qb'
            
        return 'balanced'
        
    def _analyze_overall_pattern(self, position_counts: Dict[str, int], 
                               total_rounds: int) -> str:
        """Analyze overall position distribution"""
        if total_rounds < 4:
            return 'balanced'  # Too few rounds to determine pattern
            
        # Check for position concentration
        skill_positions = ['RB', 'WR', 'TE']
        skill_total = sum(position_counts.get(pos, 0) for pos in skill_positions)
        
        if skill_total >= total_rounds * 0.8:  # 80%+ skill positions
            return 'skill_heavy'
            
        return 'balanced'
        
    def _get_position_round(self, positions_by_round: Dict[int, str], 
                           target_position: str) -> Optional[int]:
        """Get the round when a position was first drafted"""
        for round_num in sorted(positions_by_round.keys()):
            if positions_by_round[round_num] == target_position:
                return round_num
        return None
        
    def analyze_patterns(self, simulation_results: List[Dict]) -> Dict:
        """
        Group and analyze emergent patterns from multiple simulations
        
        Args:
            simulation_results: List of simulation result dictionaries
            
        Returns:
            Pattern analysis with metrics for each discovered pattern
        """
        pattern_groups = defaultdict(list)
        
        # Classify each simulation result
        for result in simulation_results:
            # Extract draft history from roster and position sequence
            draft_history = []
            if 'position_sequence' in result:
                for round_num, position in enumerate(result['position_sequence'], 1):
                    # Create dummy history entry
                    draft_history.append((round_num, f"Player_{round_num}", position))
                    
            pattern = self.detect_pattern(draft_history)
            pattern_groups[pattern].append(result)
            
        # Calculate metrics for each pattern
        return self._calculate_pattern_metrics(pattern_groups)
        
    def _calculate_pattern_metrics(self, pattern_groups: Dict[str, List[Dict]]) -> Dict:
        """Calculate performance metrics for each discovered pattern"""
        pattern_analysis = {}
        
        total_simulations = sum(len(results) for results in pattern_groups.values())
        
        for pattern, results in pattern_groups.items():
            if not results:
                continue
                
            # Extract values
            values = [r['roster_value'] for r in results if 'roster_value' in r]
            
            if not values:
                continue
                
            # Calculate statistics
            pattern_analysis[pattern] = {
                'frequency': len(results),
                'frequency_pct': len(results) / total_simulations * 100,
                'mean_value': np.mean(values),
                'std_value': np.std(values),
                'max_value': np.max(values),
                'min_value': np.min(values),
                'description': self.pattern_definitions.get(pattern, {}).get('description', 'Unknown pattern'),
                'example_sequence': self._get_example_sequence(results[0]) if results else []
            }
            
        # Sort by mean value
        sorted_patterns = sorted(
            pattern_analysis.items(),
            key=lambda x: x[1]['mean_value'],
            reverse=True
        )
        
        return {
            'patterns': dict(sorted_patterns),
            'total_simulations': total_simulations,
            'unique_patterns': len(pattern_groups),
            'most_common': max(pattern_groups.keys(), key=lambda k: len(pattern_groups[k])) if pattern_groups else 'none'
        }
        
    def _get_example_sequence(self, result: Dict) -> List[str]:
        """Extract position sequence as example"""
        if 'position_sequence' in result:
            return result['position_sequence'][:8]  # First 8 rounds
        return []
        
    def display_pattern_analysis(self, analysis: Dict, title: str = "Emergent Pattern Analysis") -> None:
        """
        Display pattern analysis in formatted output
        
        Args:
            analysis: Pattern analysis dictionary
            title: Title for the analysis
        """
        print("=" * 70)
        print(f"🧠 {title.upper()}")
        print("=" * 70)
        print(f"Total simulations: {analysis['total_simulations']}")
        print(f"Unique patterns discovered: {analysis['unique_patterns']}")
        print(f"Most common pattern: {analysis['most_common']}")
        print("")
        
        if not analysis['patterns']:
            print("No patterns detected.")
            return
            
        print("Discovered Patterns:")
        print("-" * 70)
        print(f"{'Pattern':<15} {'Value':<8} {'Freq':<6} {'±SD':<6} {'Description'}")
        print("-" * 70)
        
        for pattern, metrics in analysis['patterns'].items():
            freq_pct = metrics['frequency_pct']
            mean_val = metrics['mean_value']
            std_val = metrics['std_value']
            desc = metrics['description'][:30] + "..." if len(metrics['description']) > 30 else metrics['description']
            
            print(f"{pattern:<15} {mean_val:<8.1f} {freq_pct:<6.1f}% {std_val:<6.1f} {desc}")
            
            # Show example sequence for top patterns
            if freq_pct >= 10.0 and metrics['example_sequence']:
                seq_str = "-".join(metrics['example_sequence'])
                print(f"{'':>15} Example: {seq_str}")
                
        print("")
        
        # Show trade-off analysis for top patterns
        if len(analysis['patterns']) >= 2:
            patterns_list = list(analysis['patterns'].items())
            best_pattern = patterns_list[0]
            
            print("Trade-off Analysis:")
            print("-" * 40)
            print(f"Best pattern: {best_pattern[0]} ({best_pattern[1]['mean_value']:.1f} points)")
            
            for pattern, metrics in patterns_list[1:3]:  # Show top 2 alternatives
                diff = metrics['mean_value'] - best_pattern[1]['mean_value']
                freq_diff = metrics['frequency_pct'] - best_pattern[1]['frequency_pct']
                
                print(f"  vs {pattern}: {diff:+.1f} points ({freq_diff:+.1f}% frequency)")
                
        print("")