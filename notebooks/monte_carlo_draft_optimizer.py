"""Monte Carlo Draft Optimizer with CRN + Antithetic Variance Reduction"""

import numpy as np
from numba import njit, prange
from dataclasses import dataclass
from typing import Tuple, Optional
import hashlib
from functools import lru_cache

# Constants
POSITION_MAP = {'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3, 'K': 4, 'DST': 5}
ARCHETYPES = {
    'zero_rb': [0.1, 0.05, 0.5, 0.2, 0.1, 0.05],  # Position weights
    'rb_heavy': [0.1, 0.5, 0.2, 0.1, 0.05, 0.05],
    'balanced': [0.15, 0.3, 0.3, 0.15, 0.05, 0.05],
    'best_available': [0.17, 0.17, 0.17, 0.17, 0.16, 0.16]
}

@dataclass
class NumericState:
    """Pure numeric state for fast computation"""
    top_k_ids: np.ndarray  # shape (40,) int16 - player IDs sorted by ADP
    pos_counts: np.ndarray  # shape (6,) int16 - available by position
    my_roster: np.ndarray  # shape (6,) int8 - my roster counts
    all_team_rosters: np.ndarray  # shape (n_teams, 6) int8 - ALL teams' rosters
    drafted_players: np.ndarray  # shape (variable,) int16 - IDs of all drafted players
    pick_num: int
    my_team_idx: int
    n_teams: int
    
    # Position limits (typical league settings)
    POSITION_LIMITS = np.array([
        2,   # QB - most teams won't draft 3rd QB
        6,   # RB - reasonable max
        6,   # WR - reasonable max
        2,   # TE - most teams won't draft 3rd TE
        1,   # K  - never draft 2nd kicker
        1    # DST - never draft 2nd defense
    ], dtype=np.int8)
    
    def canonical_hash(self, k_hash: int = 20) -> int:
        """Deterministic hash for state - used for seeding and caching"""
        # Take top-K players for hashing
        topk_slice = tuple(int(x) for x in self.top_k_ids[:k_hash])
        pos_tuple = tuple(int(x) for x in self.pos_counts)
        roster_tuple = tuple(int(x) for x in self.my_roster)
        
        # Include key team roster info (teams near position limits)
        team_roster_key = []
        for team_idx in range(self.n_teams):
            for pos in range(6):
                if self.all_team_rosters[team_idx, pos] >= self.POSITION_LIMITS[pos] - 1:
                    # Team is near/at limit for this position
                    team_roster_key.append((team_idx, pos, self.all_team_rosters[team_idx, pos]))
        
        # Create string representation and hash
        state_str = f"{topk_slice}{pos_tuple}{roster_tuple}{tuple(team_roster_key)}{self.pick_num}"
        return int(hashlib.md5(state_str.encode()).hexdigest()[:8], 16)
    
    def get_picking_team(self) -> int:
        """Get which team is currently picking (snake draft)"""
        picks_per_round = self.n_teams
        current_round = (self.pick_num - 1) // picks_per_round
        position_in_round = (self.pick_num - 1) % picks_per_round
        
        if current_round % 2 == 0:  # Odd round (1, 3, 5...)
            return position_in_round
        else:  # Even round (2, 4, 6...) - reverse order
            return picks_per_round - position_in_round - 1

@njit
def compute_pick_probs_roster_aware(
    player_values: np.ndarray,
    player_positions: np.ndarray,
    team_roster: np.ndarray,  # shape (6,) - current team's roster counts
    position_limits: np.ndarray,  # shape (6,) - max per position
    archetype_weights: np.ndarray,
    temperature: float = 5.0,
    need_penalty: float = 0.1  # Multiplier when position is filled
) -> np.ndarray:
    """Compute pick probabilities with roster awareness"""
    n_players = len(player_values)
    
    # Base probabilities from value
    base_probs = np.exp(player_values / temperature)
    
    # Adjust by position preference AND roster needs
    adjusted_probs = np.zeros(n_players)
    for i in range(n_players):
        pos = player_positions[i]
        
        # Start with archetype preference
        prob = base_probs[i] * archetype_weights[pos]
        
        # Apply roster need adjustment
        current_count = team_roster[pos]
        limit = position_limits[pos]
        
        if current_count >= limit:
            # At or over limit - heavily penalize
            prob *= need_penalty
        elif current_count == limit - 1:
            # One away from limit - moderate penalty
            prob *= 0.3
        elif current_count == limit - 2:
            # Two away from limit - slight penalty for some positions
            if pos in [0, 3]:  # QB, TE - usually don't need many
                prob *= 0.7
        
        adjusted_probs[i] = prob
    
    # Normalize
    total = np.sum(adjusted_probs)
    if total > 0:
        return adjusted_probs / total
    return np.ones(n_players) / n_players

@njit
def simulate_pick_deterministic(
    available_ids: np.ndarray,
    pick_probs: np.ndarray,
    uniform: float
) -> int:
    """Deterministic pick given uniform random value"""
    cumsum = np.cumsum(pick_probs)
    idx = np.searchsorted(cumsum, uniform)
    if idx >= len(available_ids):
        idx = len(available_ids) - 1
    return available_ids[idx]

def rollout_with_uniforms(
    state: NumericState,
    candidate_id: int,
    uniforms: np.ndarray,
    player_data: dict,
    archetype_mix: np.ndarray = None
) -> float:
    """Single rollout using pre-generated uniforms with full roster tracking"""
    if archetype_mix is None:
        archetype_mix = np.array([0.25, 0.25, 0.25, 0.25])  # Equal mix
    
    # Deep copy state for simulation
    current_state = NumericState(
        top_k_ids=state.top_k_ids.copy(),
        pos_counts=state.pos_counts.copy(),
        my_roster=state.my_roster.copy(),
        all_team_rosters=state.all_team_rosters.copy(),
        drafted_players=np.append(state.drafted_players, candidate_id),
        pick_num=state.pick_num + 1,
        my_team_idx=state.my_team_idx,
        n_teams=state.n_teams
    )
    
    # Draft candidate for my team
    pos = player_data['positions'][candidate_id]
    current_state.my_roster[pos] += 1
    current_state.all_team_rosters[state.my_team_idx, pos] += 1
    current_state.top_k_ids = current_state.top_k_ids[current_state.top_k_ids != candidate_id]
    current_state.pos_counts[pos] -= 1
    
    my_roster_value = player_data['projections'][candidate_id]
    
    # Simulate remaining picks until our next turn
    uniform_idx = 0
    picks_until_next = calculate_picks_until_next(state)
    
    for pick_offset in range(picks_until_next):
        if len(current_state.top_k_ids) == 0:
            break
        
        # Determine which team is picking
        picking_team = current_state.get_picking_team()
        team_roster = current_state.all_team_rosters[picking_team]
        
        # Sample archetype for this team/pick
        archetype_idx = np.searchsorted(
            np.cumsum(archetype_mix), 
            uniforms[uniform_idx]
        )
        uniform_idx += 1
        
        # Get archetype weights
        archetype_weights = np.array(ARCHETYPES[list(ARCHETYPES.keys())[archetype_idx]])
        
        # Compute ROSTER-AWARE pick probabilities
        available_values = player_data['values'][current_state.top_k_ids]
        available_positions = player_data['positions'][current_state.top_k_ids]
        
        pick_probs = compute_pick_probs_roster_aware(
            available_values,
            available_positions,
            team_roster,
            current_state.POSITION_LIMITS,
            archetype_weights
        )
        
        # Make pick
        picked_id = simulate_pick_deterministic(
            current_state.top_k_ids,
            pick_probs,
            uniforms[uniform_idx]
        )
        uniform_idx += 1
        
        # Update state - track what this team drafted
        picked_pos = player_data['positions'][picked_id]
        current_state.all_team_rosters[picking_team, picked_pos] += 1
        current_state.top_k_ids = current_state.top_k_ids[
            current_state.top_k_ids != picked_id
        ]
        current_state.pos_counts[picked_pos] -= 1
        current_state.drafted_players = np.append(current_state.drafted_players, picked_id)
        current_state.pick_num += 1
    
    # Now it's our turn again - complete roster with awareness of what's left
    final_value = my_roster_value + complete_roster_greedy(
        current_state, player_data
    )
    
    return final_value

def evaluate_candidates_crn_adaptive(
    state: NumericState,
    candidates: np.ndarray,
    player_data: dict,
    n_pairs_initial: int = 50,
    n_pairs_max: int = 500,
    confidence_threshold: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Adaptive CRN evaluation with antithetic pairing"""
    n_candidates = len(candidates)
    candidate_evs = np.zeros(n_candidates)
    candidate_vars = np.zeros(n_candidates)
    
    # Deterministic base seed from state
    base_seed = state.canonical_hash()
    rng = np.random.default_rng(base_seed)
    
    # Initial evaluation with small sample
    uniforms_per_rollout = 100  # Max picks to simulate
    
    for pair_idx in range(n_pairs_initial):
        # Generate uniforms for this pair
        U = rng.random(uniforms_per_rollout)
        
        for c_idx, candidate in enumerate(candidates):
            # Primary rollout with U
            v1 = rollout_with_uniforms(state, candidate, U, player_data)
            
            # Antithetic rollout with 1-U
            v2 = rollout_with_uniforms(state, candidate, 1.0 - U, player_data)
            
            # Update statistics
            pair_value = (v1 + v2) / 2.0
            candidate_evs[c_idx] += pair_value
            candidate_vars[c_idx] += pair_value ** 2
    
    # Compute initial statistics
    candidate_evs /= n_pairs_initial
    candidate_vars = (candidate_vars / n_pairs_initial) - candidate_evs ** 2
    candidate_stds = np.sqrt(candidate_vars / n_pairs_initial)
    
    # Check if we need more samples
    best_idx = np.argmax(candidate_evs)
    second_best_idx = np.argsort(candidate_evs)[-2] if n_candidates > 1 else 0
    
    ev_gap = candidate_evs[best_idx] - candidate_evs[second_best_idx]
    noise_level = (candidate_stds[best_idx] + candidate_stds[second_best_idx]) / 2
    
    n_pairs_used = n_pairs_initial
    
    # Adaptive sampling for close calls
    while ev_gap < confidence_threshold * noise_level and n_pairs_used < n_pairs_max:
        additional_pairs = min(100, n_pairs_max - n_pairs_used)
        
        # Only simulate top 2 candidates to save compute
        top_2 = [best_idx, second_best_idx]
        
        for pair_idx in range(additional_pairs):
            U = rng.random(uniforms_per_rollout)
            
            for c_idx in top_2:
                candidate = candidates[c_idx]
                v1 = rollout_with_uniforms(state, candidate, U, player_data)
                v2 = rollout_with_uniforms(state, candidate, 1.0 - U, player_data)
                
                pair_value = (v1 + v2) / 2.0
                # Update running average
                old_n = n_pairs_used
                candidate_evs[c_idx] = (
                    candidate_evs[c_idx] * old_n + pair_value
                ) / (old_n + 1)
        
        n_pairs_used += additional_pairs
        
        # Recompute gap and noise
        ev_gap = candidate_evs[best_idx] - candidate_evs[second_best_idx]
        noise_level = noise_level * np.sqrt(old_n / n_pairs_used)  # Approximate
    
    # Log decision quality
    decision_quality = {
        'ev_gap': ev_gap,
        'noise_level': noise_level,
        'signal_to_noise': ev_gap / (noise_level + 1e-10),
        'n_pairs_used': n_pairs_used,
        'confident': ev_gap > confidence_threshold * noise_level
    }
    
    print(f"Decision: EV gap={ev_gap:.2f}, Noise={noise_level:.2f}, "
          f"SNR={decision_quality['signal_to_noise']:.1f}, "
          f"Sims={n_pairs_used}, Confident={decision_quality['confident']}")
    
    return candidate_evs, candidate_stds, decision_quality

@njit
def calculate_picks_until_next(state: NumericState) -> int:
    """Calculate how many picks until our next turn"""
    # Snake draft logic
    picks_per_round = state.n_teams
    current_round = (state.pick_num - 1) // picks_per_round
    position_in_round = (state.pick_num - 1) % picks_per_round
    
    if current_round % 2 == 0:  # Odd round (1, 3, 5...)
        # Normal order
        if position_in_round < state.my_team_idx:
            return state.my_team_idx - position_in_round
        else:
            # Next pick is in reverse order
            return (picks_per_round - position_in_round - 1) + \
                   (picks_per_round - state.my_team_idx)
    else:  # Even round (2, 4, 6...)
        # Reverse order
        reverse_idx = picks_per_round - state.my_team_idx - 1
        if position_in_round < reverse_idx:
            return reverse_idx - position_in_round
        else:
            # Next pick is in normal order
            return (picks_per_round - position_in_round - 1) + \
                   state.my_team_idx + 1

def complete_roster_greedy(
    state: NumericState,
    player_data: dict
) -> float:
    """Complete roster with remaining best available"""
    # Simplified - just sum top remaining values
    remaining_picks = calculate_remaining_picks(state)
    if len(state.top_k_ids) >= remaining_picks:
        top_values = player_data['projections'][state.top_k_ids[:remaining_picks]]
        return np.sum(top_values)
    return 0.0

def calculate_remaining_picks(state: NumericState) -> int:
    """Calculate how many more picks we have"""
    roster_slots = 16  # Total roster size
    return roster_slots - np.sum(state.my_roster)

class VectorCache:
    """Efficient fuzzy cache with vectorized lookup"""
    
    def __init__(self, max_size: int = 50_000, vector_dim: int = 60):  # Increased for roster tracking
        self.vectors = np.zeros((max_size, vector_dim), dtype=np.float32)
        self.values = np.zeros(max_size, dtype=np.float32)
        self.ages = np.zeros(max_size, dtype=np.int32)
        self.n_cached = 0
        self.current_age = 0
        self.vector_dim = vector_dim
    
    def state_to_vector(self, state: NumericState) -> np.ndarray:
        """Convert state to normalized vector for caching - now includes team roster info"""
        # Expand vector size to include team roster summary
        vec = np.zeros(60, dtype=np.float32)  # Increased from 48
        
        # Top-K players (first 40 dims)
        k = min(40, len(state.top_k_ids))
        vec[:k] = state.top_k_ids[:k] / 300.0  # Normalize by max player ID
        
        # Position counts (next 6 dims)
        vec[40:46] = state.pos_counts / 50.0  # Normalize by max count
        
        # My roster (dims 46-47)
        vec[46] = np.sum(state.my_roster) / 16.0
        vec[47] = np.std(state.my_roster) / 5.0  # Position balance
        
        # League roster summary (dims 48-59)
        # Count teams at/near position limits for each position
        for pos in range(6):
            teams_at_limit = np.sum(
                state.all_team_rosters[:, pos] >= state.POSITION_LIMITS[pos]
            )
            teams_near_limit = np.sum(
                state.all_team_rosters[:, pos] >= state.POSITION_LIMITS[pos] - 1
            )
            vec[48 + pos*2] = teams_at_limit / state.n_teams
            vec[49 + pos*2] = teams_near_limit / state.n_teams
        
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        return vec
    
    def lookup(self, state: NumericState, tolerance: float = 0.92) -> Optional[float]:
        """Fast fuzzy lookup with cosine similarity"""
        if self.n_cached == 0:
            return None
        
        query_vec = self.state_to_vector(state)
        
        # Batch dot product for cosine similarity
        similarities = self.vectors[:self.n_cached] @ query_vec
        
        best_idx = np.argmax(similarities)
        if similarities[best_idx] >= tolerance:
            # Weight by age (prefer recent entries)
            age_weight = 1.0 / (1.0 + 0.01 * (self.current_age - self.ages[best_idx]))
            return self.values[best_idx] * age_weight
        
        return None
    
    def insert(self, state: NumericState, value: float):
        """Insert new state-value pair"""
        if self.n_cached >= len(self.values):
            # Evict oldest
            oldest_idx = np.argmin(self.ages[:self.n_cached])
            idx = oldest_idx
        else:
            idx = self.n_cached
            self.n_cached += 1
        
        self.vectors[idx] = self.state_to_vector(state)
        self.values[idx] = value
        self.ages[idx] = self.current_age
        self.current_age += 1

class MCTSNode:
    """Node for Monte Carlo Tree Search with progressive widening"""
    
    def __init__(self, state: NumericState, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # Player ID that led to this state
        
        self.children = {}
        self.untried_actions = None  # Will be set on first expansion
        
        self.visits = 0
        self.total_value = 0.0
        self.ucb_c = 1.414  # UCB exploration constant
    
    def get_ucb_value(self, parent_visits: int) -> float:
        """Upper Confidence Bound for tree policy"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.total_value / self.visits
        exploration = self.ucb_c * np.sqrt(np.log(parent_visits) / self.visits)
        return exploitation + exploration
    
    def get_allowed_actions(self, player_data: dict, k0: int = 5, alpha: float = 0.5) -> np.ndarray:
        """Progressive widening: k = k0 * N^alpha"""
        max_actions = int(k0 * (max(1, self.visits) ** alpha))
        
        # Get top candidates by value
        available_values = player_data['values'][self.state.top_k_ids]
        top_indices = np.argsort(available_values)[::-1][:max_actions]
        
        return self.state.top_k_ids[top_indices]
    
    def select_child(self) -> 'MCTSNode':
        """Select best child using UCB"""
        if not self.children:
            return self
        
        ucb_values = [
            child.get_ucb_value(self.visits) 
            for child in self.children.values()
        ]
        best_idx = np.argmax(ucb_values)
        best_action = list(self.children.keys())[best_idx]
        
        return self.children[best_action]
    
    def expand(self, player_data: dict) -> Optional['MCTSNode']:
        """Expand node with progressive widening"""
        if self.untried_actions is None:
            self.untried_actions = list(self.get_allowed_actions(player_data))
        
        if not self.untried_actions:
            return None
        
        # Take first untried action
        action = self.untried_actions.pop(0)
        
        # Create new state
        new_state = self.apply_action(action)
        
        # Create child node
        child = MCTSNode(new_state, parent=self, action=action)
        self.children[action] = child
        
        return child
    
    def apply_action(self, player_id: int) -> NumericState:
        """Apply action to create new state"""
        new_state = NumericState(
            top_k_ids=self.state.top_k_ids[self.state.top_k_ids != player_id],
            pos_counts=self.state.pos_counts.copy(),
            my_roster=self.state.my_roster.copy(),
            pick_num=self.state.pick_num + 1,
            my_team_idx=self.state.my_team_idx,
            n_teams=self.state.n_teams
        )
        
        # Update roster and position counts
        # (Would need player position info here)
        
        return new_state
    
    def backpropagate(self, value: float):
        """Backpropagate value up the tree"""
        self.visits += 1
        self.total_value += value
        
        if self.parent:
            self.parent.backpropagate(value)

def mcts_evaluate(
    state: NumericState,
    player_data: dict,
    cache: VectorCache,
    n_iterations: int = 1000,
    depth_limit: int = 3
) -> Tuple[int, float, dict]:
    """MCTS evaluation with caching and progressive widening"""
    
    # Check cache first
    cached_value = cache.lookup(state)
    if cached_value is not None:
        return -1, cached_value, {'source': 'cache'}
    
    root = MCTSNode(state)
    
    for iteration in range(n_iterations):
        node = root
        depth = 0
        
        # Selection phase - traverse to leaf
        while depth < depth_limit and node.children and node.untried_actions == []:
            node = node.select_child()
            depth += 1
        
        # Expansion phase
        if depth < depth_limit:
            child = node.expand(player_data)
            if child:
                node = child
                depth += 1
        
        # Rollout phase - use CRN evaluation
        if depth >= depth_limit or not node.state.top_k_ids.size:
            rollout_value = complete_roster_greedy(node.state, player_data)
        else:
            # Use fast rollout
            candidates = node.get_allowed_actions(player_data)
            if candidates.size > 0:
                evs, _, _ = evaluate_candidates_crn_adaptive(
                    node.state, candidates[:1], player_data, 
                    n_pairs_initial=20, n_pairs_max=50
                )
                rollout_value = evs[0]
            else:
                rollout_value = 0.0
        
        # Backpropagation
        node.backpropagate(rollout_value)
    
    # Select best action
    if not root.children:
        return -1, 0.0, {'source': 'no_actions'}
    
    visit_counts = {action: child.visits for action, child in root.children.items()}
    best_action = max(visit_counts, key=visit_counts.get)
    best_value = root.children[best_action].total_value / root.children[best_action].visits
    
    # Cache the result
    cache.insert(state, best_value)
    
    stats = {
        'source': 'mcts',
        'iterations': n_iterations,
        'best_action_visits': root.children[best_action].visits,
        'total_actions_tried': len(root.children)
    }
    
    return best_action, best_value, stats

class DraftOptimizer:
    """Main draft optimization engine with full league roster tracking"""
    
    def __init__(self, player_data: dict, league_config: dict):
        self.player_data = player_data
        self.league_config = league_config
        self.cache = VectorCache(max_size=50_000, vector_dim=60)
        
        # Performance tracking
        self.decision_log = []
        self.latency_log = []
        
        # Initialize empty rosters for all teams
        self.n_teams = league_config.get('n_teams', 10)
        self.all_team_rosters = np.zeros((self.n_teams, 6), dtype=np.int8)
        self.drafted_players = np.array([], dtype=np.int16)
    
    def update_draft_state(self, pick_data: dict):
        """Update state with external pick data
        
        Args:
            pick_data: {
                'pick_num': int,
                'team_idx': int,
                'player_id': int,
                'position': str or int
            }
        """
        team_idx = pick_data['team_idx']
        player_id = pick_data['player_id']
        
        # Handle position as string or int
        if isinstance(pick_data['position'], str):
            pos = POSITION_MAP[pick_data['position']]
        else:
            pos = pick_data['position']
        
        # Update team roster
        self.all_team_rosters[team_idx, pos] += 1
        
        # Track drafted player
        self.drafted_players = np.append(self.drafted_players, player_id)
        
        # Log the update
        print(f"Updated: Team {team_idx} drafted {pick_data.get('position', pos)} "
              f"(Player {player_id}). Team now has {self.all_team_rosters[team_idx, pos]} at position.")
    
    def batch_update_draft_state(self, picks_list: list):
        """Update state with multiple picks at once
        
        Args:
            picks_list: List of pick_data dicts
        """
        for pick_data in picks_list:
            self.update_draft_state(pick_data)
    
    def get_current_state(self, available_players: np.ndarray, pick_num: int, my_team_idx: int) -> NumericState:
        """Create current state from available players and draft position"""
        # Filter out drafted players
        available_ids = np.array([p for p in available_players if p not in self.drafted_players])
        
        # Sort by value/ADP and take top-K
        values = self.player_data['values'][available_ids]
        sorted_indices = np.argsort(values)[::-1][:40]  # Top 40
        top_k_ids = available_ids[sorted_indices]
        
        # Calculate position counts from available players
        pos_counts = np.zeros(6, dtype=np.int16)
        for player_id in available_ids:
            pos = self.player_data['positions'][player_id]
            pos_counts[pos] += 1
        
        return NumericState(
            top_k_ids=top_k_ids,
            pos_counts=pos_counts,
            my_roster=self.all_team_rosters[my_team_idx].copy(),
            all_team_rosters=self.all_team_rosters.copy(),
            drafted_players=self.drafted_players.copy(),
            pick_num=pick_num,
            my_team_idx=my_team_idx,
            n_teams=self.n_teams
        )
    
    def make_pick_decision(
        self,
        state: NumericState,
        use_mcts: bool = False,
        verbose: bool = True
    ) -> Tuple[int, dict]:
        """Main decision function - returns player ID to draft"""
        import time
        start_time = time.time()
        
        # Get candidates
        candidates = self.get_top_candidates(state, M=12)
        
        if len(candidates) == 0:
            return -1, {'error': 'No candidates available'}
        
        if len(candidates) == 1:
            return candidates[0], {'reason': 'Only one candidate'}
        
        # Evaluate with CRN + adaptive sampling
        evs, stds, quality = evaluate_candidates_crn_adaptive(
            state, candidates, self.player_data,
            n_pairs_initial=50,
            n_pairs_max=500,
            confidence_threshold=3.0
        )
        
        # Select best
        best_idx = np.argmax(evs)
        best_player = candidates[best_idx]
        
        # Calculate decision metrics
        second_best_idx = np.argsort(evs)[-2]
        ev_gap = evs[best_idx] - evs[second_best_idx]
        
        latency = time.time() - start_time
        
        decision_info = {
            'player_id': int(best_player),
            'expected_value': float(evs[best_idx]),
            'ev_gap': float(ev_gap),
            'confidence': quality['confident'],
            'signal_to_noise': float(quality['signal_to_noise']),
            'simulations_used': quality['n_pairs_used'],
            'latency_ms': latency * 1000,
            'candidates_evaluated': len(candidates)
        }
        
        # Log for analysis
        self.decision_log.append(decision_info)
        self.latency_log.append(latency)
        
        if verbose:
            player_name = self.player_data.get('names', {}).get(best_player, f'Player_{best_player}')
            print(f"\n🎯 PICK DECISION: {player_name}")
            print(f"   EV: {evs[best_idx]:.1f} pts (gap: {ev_gap:.1f})")
            print(f"   Confidence: {'HIGH' if quality['confident'] else 'LOW'} (SNR: {quality['signal_to_noise']:.1f})")
            print(f"   Simulations: {quality['n_pairs_used']} | Latency: {latency*1000:.0f}ms")
        
        return best_player, decision_info
    
    def get_top_candidates(self, state: NumericState, M: int = 12) -> np.ndarray:
        """Get top M candidates based on value + position need + roster construction"""
        if len(state.top_k_ids) == 0:
            return np.array([], dtype=np.int32)
        
        # Base values
        values = self.player_data['values'][state.top_k_ids]
        
        # Position scarcity adjustment
        positions = self.player_data['positions'][state.top_k_ids]
        adjusted_values = values.copy()
        
        for i, player_id in enumerate(state.top_k_ids):
            pos = positions[i]
            
            # Global scarcity bonus
            if state.pos_counts[pos] < 10:
                adjusted_values[i] += (10 - state.pos_counts[pos]) * 2
            
            # My roster need bonus/penalty
            my_count = state.my_roster[pos]
            limit = state.POSITION_LIMITS[pos]
            
            if my_count >= limit:
                # At limit - heavily penalize
                adjusted_values[i] -= 50
            elif my_count == 0 and pos in [0, 1, 2]:  # Need starter at QB/RB/WR
                adjusted_values[i] += 10
            
            # League-wide scarcity (many teams need this position)
            teams_needing = np.sum(state.all_team_rosters[:, pos] < limit - 1)
            if teams_needing > state.n_teams * 0.7:  # 70% of teams need it
                adjusted_values[i] += 5
        
        # Get top M
        top_indices = np.argsort(adjusted_values)[::-1][:M]
        return state.top_k_ids[top_indices]
    
    def backtest(
        self,
        historical_drafts: list,
        n_seasons: int = 1000
    ) -> dict:
        """Backtest strategy against historical data"""
        from sklearn.metrics import brier_score_loss
        
        results = {
            'ev_captured': [],
            'win_rates': [],
            'brier_scores': [],
            'latencies_p95': []
        }
        
        for draft in historical_drafts:
            # Run optimizer on historical draft
            optimizer_picks = []
            actual_picks = draft['actual_picks']
            
            for pick_state in draft['states']:
                player_id, _ = self.make_pick_decision(pick_state, verbose=False)
                optimizer_picks.append(player_id)
            
            # Calculate EV captured
            optimizer_ev = np.sum(self.player_data['projections'][optimizer_picks])
            actual_ev = np.sum(self.player_data['projections'][actual_picks])
            results['ev_captured'].append(optimizer_ev / actual_ev)
            
            # Simulate seasons
            win_rate = self.simulate_season_outcomes(optimizer_picks, n_seasons)
            results['win_rates'].append(win_rate)
            
            # Calculate Brier score for availability predictions
            # (Would need predicted vs actual availability data)
            
            # Latency analysis
            if self.latency_log:
                p95_latency = np.percentile(self.latency_log, 95)
                results['latencies_p95'].append(p95_latency)
        
        return {
            'mean_ev_ratio': np.mean(results['ev_captured']),
            'mean_win_rate': np.mean(results['win_rates']),
            'p95_latency_ms': np.mean(results['latencies_p95']) * 1000 if results['latencies_p95'] else 0
        }
    
    def simulate_season_outcomes(self, roster: list, n_seasons: int) -> float:
        """Simulate season outcomes to calculate win rate"""
        # Simplified - would implement full season simulation
        roster_projection = np.sum(self.player_data['projections'][roster])
        
        # Simulate opponent rosters and compare
        wins = 0
        for _ in range(n_seasons):
            # Generate random opponent score
            opponent_score = np.random.normal(1500, 200)  # League average
            if roster_projection > opponent_score:
                wins += 1
        
        return wins / n_seasons

# Example usage
def main():
    """Example usage of the optimizer with roster tracking"""
    
    # Load player data (would come from your existing data pipeline)
    player_data = {
        'values': np.random.randn(300),  # VBD values
        'projections': np.random.uniform(50, 300, 300),  # Fantasy points
        'positions': np.random.randint(0, 4, 300),  # Position indices
        'names': {i: f'Player_{i}' for i in range(300)}
    }
    
    league_config = {
        'n_teams': 10,
        'roster_slots': 16,
        'scoring': 'PPR'
    }
    
    # Initialize optimizer
    optimizer = DraftOptimizer(player_data, league_config)
    
    # Simulate some picks that have already happened
    prior_picks = [
        {'pick_num': 1, 'team_idx': 0, 'player_id': 1, 'position': 'RB'},
        {'pick_num': 2, 'team_idx': 1, 'player_id': 2, 'position': 'RB'},
        {'pick_num': 3, 'team_idx': 2, 'player_id': 3, 'position': 'WR'},
        {'pick_num': 4, 'team_idx': 3, 'player_id': 4, 'position': 'RB'},
        {'pick_num': 5, 'team_idx': 4, 'player_id': 5, 'position': 'WR'},
        {'pick_num': 6, 'team_idx': 5, 'player_id': 6, 'position': 'RB'},
        {'pick_num': 7, 'team_idx': 6, 'player_id': 7, 'position': 'QB'},
    ]
    
    # Update optimizer with prior picks
    optimizer.batch_update_draft_state(prior_picks)
    
    # Get current state for my pick (8th pick, team index 7)
    available_players = np.arange(300, dtype=np.int16)  # All player IDs
    state = optimizer.get_current_state(
        available_players=available_players,
        pick_num=8,
        my_team_idx=7
    )
    
    # Make pick decision
    player_id, decision_info = optimizer.make_pick_decision(state)
    
    print(f"\n📊 Decision Summary:")
    print(f"   Selected Player: {player_id}")
    print(f"   Expected Value: {decision_info['expected_value']:.1f}")
    print(f"   Confidence: {'HIGH' if decision_info['confidence'] else 'LOW'}")
    print(f"   Latency: {decision_info['latency_ms']:.0f}ms")

if __name__ == '__main__':
    main()