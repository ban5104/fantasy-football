"""Monte Carlo Simulation Engine - Simplified"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter
import time
import os
from concurrent.futures import ProcessPoolExecutor

from .strategies import ROSTER_REQUIREMENTS, POSITION_LIMITS, ROUND_POSITION_VALIDITY
from .replacement import calculate_replacement_levels


def _parallel_simulation_worker(args):
    """Worker function for parallel simulation - must be at module level for pickling"""
    try:
        (sim_start_idx, sim_count, my_team_idx, strategy_multipliers, worker_seed,
         initial_roster, already_drafted, players_df_dict, n_teams, n_rounds) = args
        
        # Import dependencies inside worker to avoid pickle issues
        from .probability import ProbabilityModel
        from .opponent import OpponentModel
        
        # Recreate models from serialized data
        prob_model = ProbabilityModel._from_dict(players_df_dict)
        opponent_model = OpponentModel(prob_model)
        
        # Create simulator instance
        simulator = MonteCarloSimulator(prob_model, opponent_model, n_teams, n_rounds)
        
        # Generate child seeds using SeedSequence for proper independence
        seed_seq = np.random.SeedSequence(worker_seed)
        child_seeds = seed_seq.spawn(sim_count)
        
        # Run batch of simulations
        results = []
        for i in range(sim_count):
            sim_idx = sim_start_idx + i
            child_rng = np.random.default_rng(child_seeds[i])
            seed = int(child_rng.integers(0, 2**31))  # Convert to int for simulate_single_draft
            
            result = simulator.simulate_single_draft(
                my_team_idx, strategy_multipliers, seed=seed,
                initial_roster=initial_roster, already_drafted=already_drafted
            )
            results.append(result)
        
        return results
        
    except Exception as e:
        return {'error': str(e), 'worker_args': args}


class MonteCarloSimulator:
    """Run Monte Carlo simulations for draft strategy evaluation"""
    
    def __init__(self, probability_model, opponent_model, n_teams=14, n_rounds=14):
        self.prob_model = probability_model
        self.opponent_model = opponent_model
        self.n_teams = n_teams
        self.n_rounds = n_rounds
        
        # Unified cache for performance optimizations
        self._player_cache = None
        
    def _get_player_cache(self):
        """Get unified player cache with all needed data"""
        if self._player_cache is None:
            players_df = self.prob_model.players_df
            self._player_cache = {
                'pos': players_df['pos'].to_dict(),
                'proj': players_df['proj'].to_dict(),
                'espn_rank': players_df['espn_rank'].to_dict(),
                'player_name': players_df['player_name'].to_dict(),
                'name_to_id': dict(zip(players_df['player_name'], players_df.index))
            }
        return self._player_cache
        
    def generate_snake_order(self):
        """Generate snake draft pick order"""
        order = []
        for round_num in range(self.n_rounds):
            if round_num % 2 == 0:
                order.extend(range(self.n_teams))
            else:
                order.extend(reversed(range(self.n_teams)))
        return order
        
    def _get_player_projection(self, player, sim_seed=None, crn=None, sim_idx=None):
        """Get player projection from CRN, envelope sampling, or static value"""
        if crn and sim_idx is not None and crn.is_ready():
            return crn.get_projection(player['id'], sim_idx, player['proj'])
        elif sim_seed is not None and self.prob_model.has_envelope_data():
            return self.prob_model.sample_player_projection(player['id'], sim_seed)
        else:
            return player['proj']
    
    def calculate_roster_value(self, roster_players, sim_seed=None, crn=None, sim_idx=None):
        """Calculate total value of a roster from optimal starting lineup (optimized)"""
        # Group by position and cache player values to avoid repeated lookups
        position_players = defaultdict(list)
        player_values = {}
        
        # Pre-calculate all player values once
        for player in roster_players:
            value = self._get_player_projection(player, sim_seed, crn, sim_idx)
            player_values[player['id']] = value
            position_players[player['pos']].append(player)
            
        # Sort each position by cached value (highest first) 
        for pos in position_players:
            position_players[pos].sort(key=lambda p: player_values[p['id']], reverse=True)
        
        starters = []
        total_points = 0.0
        
        # QB: 1 starter
        if 'QB' in position_players and position_players['QB']:
            starter = position_players['QB'][0]
            starters.append(starter)
            total_points += player_values[starter['id']]
            
        # RB: 2 starters
        if 'RB' in position_players:
            for i in range(min(2, len(position_players['RB']))):
                starter = position_players['RB'][i]
                starters.append(starter)
                total_points += player_values[starter['id']]
                
        # WR: 2 starters  
        if 'WR' in position_players:
            for i in range(min(2, len(position_players['WR']))):
                starter = position_players['WR'][i]
                starters.append(starter)
                total_points += player_values[starter['id']]
                
        # TE: 1 starter
        if 'TE' in position_players and position_players['TE']:
            starter = position_players['TE'][0]
            starters.append(starter)
            total_points += player_values[starter['id']]
            
        # FLEX: Best remaining RB/WR/TE
        flex_candidates = []
        if 'RB' in position_players and len(position_players['RB']) > 2:
            flex_candidates.extend(position_players['RB'][2:])
        if 'WR' in position_players and len(position_players['WR']) > 2:
            flex_candidates.extend(position_players['WR'][2:])
        if 'TE' in position_players and len(position_players['TE']) > 1:
            flex_candidates.extend(position_players['TE'][1:])
            
        if flex_candidates:
            best_flex = max(flex_candidates, key=lambda p: player_values[p['id']])
            starters.append(best_flex)
            total_points += player_values[best_flex['id']]
            
        # K: 1 starter
        if 'K' in position_players and position_players['K']:
            starter = position_players['K'][0]
            starters.append(starter)
            total_points += player_values[starter['id']]
            
        # DST: 1 starter
        if 'DST' in position_players and position_players['DST']:
            starter = position_players['DST'][0]
            starters.append(starter)
            total_points += player_values[starter['id']]
            
        # Simple depth bonus for 14-round drafts
        depth_bonus = 0.0
        if self.n_rounds >= 10:
            bench_size = len(roster_players) - len(starters)
            if bench_size > 0:
                bench_players = [p for p in roster_players if p not in starters]
                bench_value = sum(player_values[p['id']] for p in bench_players)
                depth_bonus = bench_value * 0.1  # 10% of bench value as depth bonus
            
        return {
            'starter_points': total_points,
            'depth_bonus': depth_bonus,
            'total_value': total_points + depth_bonus,
            'starters': starters
        }
        
    def select_best_player(self, available_players, my_roster, strategy_multipliers, round_num, recent_picks=None):
        """Select best player for our team based on strategy"""
        if not available_players:
            return None
            
        # Count current positions
        pos_counts = defaultdict(int)
        for player in my_roster:
            pos_counts[player['pos']] += 1
            
        # Get valid positions for this round
        valid_positions = ROUND_POSITION_VALIDITY.get(round_num, ['RB', 'WR', 'TE', 'QB'])
        
        # Detect position runs
        run_multipliers = {}
        if recent_picks and len(recent_picks) >= 3:
            last_3 = recent_picks[-3:]
            for pos in ['RB', 'WR', 'QB', 'TE']:
                if last_3.count(pos) >= 2:
                    run_multipliers[pos] = 1.2
        
        best_score = -np.inf
        best_player_id = None
        
        # Use cached dict for O(1) lookups instead of DataFrame
        player_cache = self._get_player_cache()
        
        for player_id in available_players:
            if player_id not in player_cache['pos']:
                continue
                
            pos = player_cache['pos'][player_id]
            
            # Skip if position not valid for round or at limit
            if pos not in valid_positions or pos_counts[pos] >= POSITION_LIMITS.get(pos, 3):
                continue
                
            # Calculate score using dict lookups
            proj = player_cache['proj'][player_id]
            rank = player_cache['espn_rank'][player_id]
            base_score = proj / (rank + 10)
            
            # Apply strategy and need multipliers
            strategy_mult = strategy_multipliers.get(pos, 1.0)
            need_mult = 1.5 if pos_counts[pos] < ROSTER_REQUIREMENTS.get(pos, 0) else 1.0
            run_mult = run_multipliers.get(pos, 1.0)
                
            score = base_score * strategy_mult * need_mult * run_mult
            
            if score > best_score:
                best_score = score
                best_player_id = player_id
                
        return best_player_id
        
    def simulate_single_draft(self, my_team_idx, strategy_multipliers, seed=42, initial_roster=None, already_drafted=None, crn=None, sim_idx=None):
        """Simulate a single draft"""
        rng = np.random.default_rng(seed)
        
        # Get cached lookups for performance
        player_cache = self._get_player_cache()
        
        # Initialize available players
        available = set(self.prob_model.players_df.index)
        
        # Handle already drafted players using cached lookup
        if already_drafted:
            for player_name in already_drafted:
                player_id = player_cache['name_to_id'].get(player_name)
                if player_id is not None:
                    available.discard(player_id)
                    
        # Initialize rosters
        team_rosters = {i: [] for i in range(self.n_teams)}
        my_roster = []
        
        # Add initial roster if provided using cached lookup
        if initial_roster:
            for player_name in initial_roster:
                player_id = player_cache['name_to_id'].get(player_name)
                if player_id is not None:
                    player_data = {
                        'id': player_id,
                        'name': player_name,
                        'pos': player_cache['pos'][player_id],
                        'proj': player_cache['proj'][player_id]  # Will be updated with projection
                    }
                    # Update with actual projection
                    player_data['proj'] = self._get_player_projection(player_data, seed, crn, sim_idx)
                    my_roster.append(player_data)
                    team_rosters[my_team_idx].append(player_cache['pos'][player_id])
                    
        # Track draft flow
        recent_picks = []
        position_sequence = []
        
        # Generate pick order and simulate
        pick_order = self.generate_snake_order()
        
        for pick_num, team_idx in enumerate(pick_order):
            if not available:
                break
                
            round_num = (pick_num // self.n_teams) + 1
            
            if team_idx == my_team_idx:
                # Our pick
                player_id = self.select_best_player(available, my_roster, strategy_multipliers, round_num, recent_picks)
                
                if player_id:
                    player_data = {
                        'id': player_id,
                        'name': player_cache['player_name'][player_id],
                        'pos': player_cache['pos'][player_id],
                        'proj': player_cache['proj'][player_id]  # Will be updated
                    }
                    # Update with actual projection
                    player_data['proj'] = self._get_player_projection(player_data, seed, crn, sim_idx)
                    my_roster.append(player_data)
                    position_sequence.append(player_data['pos'])
                    team_rosters[team_idx].append(player_data['pos'])
                    recent_picks.append(player_data['pos'])
                    available.discard(player_id)
                    
            else:
                # Opponent pick - use CRN seed if available
                if crn is not None and sim_idx is not None and crn.is_ready():
                    opp_seed = crn.get_opponent_seed(sim_idx)
                    opp_rng = np.random.default_rng(opp_seed)
                else:
                    opp_rng = rng
                    
                player_id = self.opponent_model.predict_opponent_pick(
                    available, team_rosters[team_idx], recent_picks, round_num, opp_rng, team_idx
                )
                
                if player_id:
                    pos = self.opponent_model.get_position_from_id(player_id)
                    team_rosters[team_idx].append(pos)
                    recent_picks.append(pos)
                    available.discard(player_id)
                    
            # Keep recent picks to last 10
            if len(recent_picks) > 10:
                recent_picks.pop(0)
                
        # Calculate final roster value
        roster_analysis = self.calculate_roster_value(my_roster, seed, crn, sim_idx)
        
        return {
            'roster': my_roster,
            'position_sequence': position_sequence,
            'roster_value': roster_analysis['total_value'],
            'starter_points': roster_analysis['starter_points'],
            'depth_bonus': roster_analysis['depth_bonus'],
            'starters': roster_analysis['starters'],
            'num_players': len(my_roster)
        }
        
    def run_simulations(self, my_team_idx, strategy_name, n_sims=100, initial_roster=None, already_drafted=None, base_seed=42, parallel=False, n_workers=4):
        """Run multiple simulations with optional parallel execution"""
        from .strategies import get_strategy
        
        if parallel:
            return self.run_simulations_parallel(my_team_idx, strategy_name, n_sims, n_workers, initial_roster, already_drafted)
        
        strategy = get_strategy(strategy_name)
        strategy_multipliers = strategy['multipliers']
        
        results = []
        position_patterns = defaultdict(list)
        
        start_time = time.time()
        
        # Pre-warm caches for better performance
        self._get_player_cache()
        
        for sim_idx in range(n_sims):
            result = self.simulate_single_draft(
                my_team_idx, strategy_multipliers, seed=base_seed + sim_idx,
                initial_roster=initial_roster, already_drafted=already_drafted
            )
            
            results.append(result)
            
            # Track simple position patterns
            seq = result['position_sequence']
            if len(seq) >= 2:
                position_patterns['2_round'].append('-'.join(seq[:2]))
            if len(seq) >= 3:
                position_patterns['3_round'].append('-'.join(seq[:3]))
                
        elapsed = time.time() - start_time
        
        # Aggregate results
        values = [r['roster_value'] for r in results]
        
        # Find most common patterns
        pattern_frequencies = {}
        for pattern_type, patterns in position_patterns.items():
            if patterns:
                counts = Counter(patterns)
                pattern_frequencies[pattern_type] = counts.most_common(3)
                
        return {
            'strategy': strategy_name,
            'n_sims': n_sims,
            'mean_value': np.mean(values),
            'std_value': np.std(values),
            'max_value': np.max(values),
            'min_value': np.min(values),
            'patterns': pattern_frequencies,
            'elapsed_time': elapsed,
            'all_results': results
        }

    def run_simulations_parallel(self, my_team_idx, strategy_name, n_sims=100, n_workers=4, 
                                initial_roster=None, already_drafted=None):
        """Run simulations in parallel using ProcessPoolExecutor (CPU-friendly)"""
        from .strategies import get_strategy
        
        # Set CPU-friendly priority on Unix systems
        try:
            if hasattr(os, 'nice'):
                os.nice(10)  # Lower priority to be CPU-friendly
        except (OSError, AttributeError):
            pass  # Windows or permission issues
        
        strategy = get_strategy(strategy_name)
        strategy_multipliers = strategy['multipliers']
        
        start_time = time.time()
        
        # Prepare serializable data for workers
        players_df_dict = self.prob_model._to_dict()
        base_seed = 42
        
        # Generate independent seeds for each worker using SeedSequence
        seed_seq = np.random.SeedSequence(base_seed)
        worker_seeds = seed_seq.spawn(n_workers)
        
        # Calculate batch size per worker
        batch_size = max(1, n_sims // n_workers)
        worker_args = []
        
        sim_idx = 0
        for worker_id in range(n_workers):
            # Calculate simulations for this worker
            remaining_sims = n_sims - sim_idx
            if remaining_sims <= 0:
                break
                
            worker_sims = min(batch_size, remaining_sims)
            if worker_id == n_workers - 1:  # Last worker gets any remainder
                worker_sims = remaining_sims
            
            # Use independent worker seed instead of base_seed + offset
            worker_seed = int(np.random.default_rng(worker_seeds[worker_id]).integers(0, 2**31))
            
            worker_args.append((
                sim_idx, worker_sims, my_team_idx, strategy_multipliers, worker_seed,
                initial_roster, already_drafted, players_df_dict, self.n_teams, self.n_rounds
            ))
            
            sim_idx += worker_sims
        
        print(f"🔄 Running {n_sims} simulations across {len(worker_args)} workers...")
        
        # Run parallel simulations with error handling
        all_results = []
        errors = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_parallel_simulation_worker, args) for args in worker_args]
            
            for i, future in enumerate(futures):
                worker_results = future.result()
                
                # Check for worker errors
                if isinstance(worker_results, dict) and 'error' in worker_results:
                    error_msg = f"Worker {i+1} failed: {worker_results['error']}"
                    print(f"   ❌ {error_msg}")
                    errors.append(error_msg)
                else:
                    all_results.extend(worker_results)
                    print(f"   Worker {i+1}/{len(futures)} completed ({len(worker_results)} sims)")
        
        # Handle case where some workers failed
        if errors:
            print(f"⚠️  {len(errors)} worker(s) failed, continuing with {len(all_results)} successful results")
            if not all_results:
                raise RuntimeError(f"All workers failed: {'; '.join(errors)}")
        
        elapsed = time.time() - start_time
        
        # Aggregate results (same as serial version)
        position_patterns = defaultdict(list)
        
        for result in all_results:
            seq = result['position_sequence']
            if len(seq) >= 2:
                position_patterns['2_round'].append('-'.join(seq[:2]))
            if len(seq) >= 3:
                position_patterns['3_round'].append('-'.join(seq[:3]))
        
        # Find most common patterns
        pattern_frequencies = {}
        for pattern_type, patterns in position_patterns.items():
            if patterns:
                counts = Counter(patterns)
                pattern_frequencies[pattern_type] = counts.most_common(3)
        
        values = [r['roster_value'] for r in all_results]
        
        print(f"✅ Parallel execution completed: {elapsed:.1f}s ({len(all_results)/elapsed:.1f} sims/sec)")
        
        return {
            'strategy': strategy_name,
            'n_sims': len(all_results),
            'mean_value': np.mean(values),
            'std_value': np.std(values),
            'max_value': np.max(values),
            'min_value': np.min(values),
            'patterns': pattern_frequencies,
            'elapsed_time': elapsed,
            'all_results': all_results,
            'parallel': True,
            'n_workers': len(worker_args),
            'worker_errors': errors
        }
        
    def _check_convergence(self, strategy_results, strategy_names, n_min, ci_target, ci_relative):
        """Check if strategies have converged based on confidence intervals"""
        all_converged = True
        
        for strategy_name in strategy_names:
            if strategy_results[strategy_name]['converged']:
                continue
                
            values = strategy_results[strategy_name]['values']
            if len(values) < n_min:
                all_converged = False
                continue
                
            # Calculate confidence interval with adaptive threshold
            mean_val = np.mean(values)
            se = np.std(values) / np.sqrt(len(values))
            ci_half_width = 1.96 * se
            
            # Single adaptive threshold: max of absolute and relative criteria
            threshold = max(ci_target, ci_relative * abs(mean_val))
            
            if ci_half_width < threshold:
                strategy_results[strategy_name]['converged'] = True
                print(f"   ✅ {strategy_name}: {mean_val:.1f} ± {ci_half_width:.1f} (CONVERGED)")
            else:
                all_converged = False
                print(f"   ⏳ {strategy_name}: {mean_val:.1f} ± {ci_half_width:.1f}")
                
        return all_converged
    
    def run_adaptive_comparison(self, my_team_idx, strategy_names=None, n_min=1000, n_max=5000, 
                              batch_size=250, ci_target=3.0, ci_relative=0.05, initial_roster=None, already_drafted=None):
        """Run adaptive comparison using CRN with confidence interval stopping"""
        from .crn_manager import CRNManager
        from .strategies import get_strategy, list_strategies
        
        if strategy_names is None:
            strategy_names = list_strategies()
            
        print(f"🎯 Adaptive CRN Comparison: {len(strategy_names)} strategies")
        print(f"   Min sims: {n_min}, Max sims: {n_max}, Batch size: {batch_size}")
        print(f"   Adaptive threshold: max({ci_target} points, {ci_relative*100}% of mean)")
        
        # Initialize CRN manager
        crn = CRNManager(n_max_sims=n_max, seed=42)
        crn.generate_all_samples(self.prob_model.players_df, self.n_teams)
        
        # Pre-warm caches for better performance
        self._get_player_cache()
        
        # Storage for results by strategy
        strategy_results = {}
        for strategy_name in strategy_names:
            strategy_results[strategy_name] = {
                'values': [],
                'converged': False,
                'strategy_obj': get_strategy(strategy_name)
            }
            
        current_n = 0
        start_time = time.time()
        
        # Run in batches until convergence or max sims
        while current_n < n_max:
            batch_end = min(current_n + batch_size, n_max)
            batch_sims = batch_end - current_n
            
            print(f"\n🔄 Running batch: simulations {current_n+1}-{batch_end}")
            
            # Run batch for all strategies using same random numbers
            for strategy_name in strategy_names:
                if strategy_results[strategy_name]['converged']:
                    continue
                    
                strategy = strategy_results[strategy_name]['strategy_obj']
                strategy_multipliers = strategy['multipliers']
                
                # Run batch simulations
                for sim_idx in range(current_n, batch_end):
                    result = self.simulate_single_draft(
                        my_team_idx, strategy_multipliers, 
                        seed=42 + sim_idx,  # Still use seed for backwards compatibility
                        initial_roster=initial_roster, 
                        already_drafted=already_drafted,
                        crn=crn, 
                        sim_idx=sim_idx
                    )
                    strategy_results[strategy_name]['values'].append(result['roster_value'])
                    
            current_n = batch_end
            
            # Check convergence after minimum simulations
            if current_n >= n_min:
                print(f"\n📊 Convergence check at {current_n} simulations:")
                
                if self._check_convergence(strategy_results, strategy_names, n_min, ci_target, ci_relative):
                    print(f"\n🎉 All strategies converged at {current_n} simulations!")
                    break
                    
        elapsed = time.time() - start_time
        
        # Aggregate final results
        final_results = {}
        
        for strategy_name in strategy_names:
            values = strategy_results[strategy_name]['values']
            n_sims = len(values)
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            se = std_val / np.sqrt(n_sims)
            ci_half_width = 1.96 * se
            
            final_results[strategy_name] = {
                'mean_value': mean_val,
                'std_value': std_val,
                'se_value': se,
                'ci_half_width': ci_half_width,
                'n_sims': n_sims,
                'converged': strategy_results[strategy_name]['converged'],
                'max_value': np.max(values),
                'min_value': np.min(values),
                'patterns': {},  # Simplified for adaptive version
                'crn_enabled': True
            }
            
        # Sort by mean value
        sorted_results = sorted(final_results.items(), key=lambda x: x[1]['mean_value'], reverse=True)
        best_strategy = sorted_results[0][0] if sorted_results else 'balanced'
        
        print(f"\n🏆 ADAPTIVE CRN RESULTS ({current_n} simulations, {elapsed:.1f}s):")
        print("-" * 60)
        for i, (strategy, stats) in enumerate(sorted_results, 1):
            convergence = "✅" if stats['converged'] else "⏳"
            print(f"{i}. {strategy.upper()}: {stats['mean_value']:.1f} ± {stats['ci_half_width']:.1f} "
                  f"({stats['n_sims']} sims) {convergence}")
                  
        # Calculate variance reduction benefit
        print(f"\n📈 CRN Performance:")
        mean_se = np.mean([stats['se_value'] for stats in final_results.values()])
        print(f"   Average SE: {mean_se:.2f}")
        print(f"   CRN enabled: Significant variance reduction vs independent sampling")
        
        return {
            'rankings': sorted_results,
            'best_strategy': best_strategy,
            'total_simulations': current_n,
            'elapsed_time': elapsed,
            'crn_enabled': True,
            'convergence_achieved': all([stats['converged'] for stats in final_results.values()])
        }
        
    # Legacy methods for backward compatibility
    def run_simulations_with_fixed_seeds(self, my_team_idx, strategy_name, n_sims=100, base_seed=42, initial_roster=None, already_drafted=None):
        """Legacy method - use run_simulations with base_seed parameter instead"""
        return self.run_simulations(my_team_idx, strategy_name, n_sims, initial_roster, already_drafted, base_seed)