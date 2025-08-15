#!/usr/bin/env python3
"""
Fantasy Football Draft Engine
Advanced draft state management and AI recommendation system

Run with: uv run python draft_engine.py
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import math

@dataclass
class Player:
    id: str
    name: str
    position: str
    team: str
    fantasy_pts: float
    vbd: float
    tier: int
    adp: float = None
    bye_week: int = None
    
    def __post_init__(self):
        # Clean up player name
        if hasattr(self, 'name') and self.name:
            # Remove team suffix if present (e.g., "Josh Allen BUF" -> "Josh Allen")
            parts = self.name.split()
            if len(parts) > 2 and parts[-1].isupper() and len(parts[-1]) <= 3:
                self.name = ' '.join(parts[:-1])

@dataclass
class Pick:
    team_id: int
    player: Player
    pick_number: int
    round: int
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class Team:
    def __init__(self, team_id: int, name: str = None):
        self.team_id = team_id
        self.name = name or f"Team {team_id}"
        self.players: List[Player] = []
        self.roster: Dict[str, List[Player]] = {
            'QB': [], 'RB': [], 'WR': [], 'TE': [], 'K': [], 'DST': [], 'BENCH': []
        }
    
    def add_player(self, player: Player):
        self.players.append(player)
        
        # Add to position roster or bench
        position = player.position if player.position != 'DEF' else 'DST'
        
        # Check if we need to put on bench based on roster limits
        starter_slots = self.get_starter_slots_needed(position)
        current_starters = len([p for p in self.roster[position] if p not in self.roster['BENCH']])
        
        if current_starters < starter_slots:
            self.roster[position].append(player)
        else:
            self.roster['BENCH'].append(player)
    
    def get_starter_slots_needed(self, position: str) -> int:
        """Get number of starter slots needed for position"""
        starter_requirements = {
            'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'K': 1, 'DST': 1
        }
        return starter_requirements.get(position, 0)
    
    def needs_position(self, position: str) -> bool:
        """Check if team still needs players at this position"""
        current_count = len(self.roster.get(position, []))
        required = self.get_starter_slots_needed(position)
        return current_count < required
    
    def get_position_count(self, position: str) -> int:
        return len(self.roster.get(position, []))
    
    def remove_player(self, player: Player):
        """Remove player from roster (for undo functionality)"""
        if player in self.players:
            self.players.remove(player)
            
            # Remove from position roster
            for pos_list in self.roster.values():
                if player in pos_list:
                    pos_list.remove(player)
                    break

class DraftState:
    def __init__(self, config: dict, user_team_id: int, draft_position: int):
        self.config = config
        self.user_team_id = user_team_id
        self.draft_position = draft_position
        self.teams: Dict[int, Team] = {}
        self.picks: List[Pick] = []
        self.current_pick = 1
        
        # Initialize teams
        team_names = config.get('team_names', [])
        for i in range(1, config['basic_settings']['teams'] + 1):
            name = team_names[i-1] if i-1 < len(team_names) else f"Team {i}"
            self.teams[i] = Team(i, name)
    
    def make_pick(self, team_id: int, player: Player):
        """Make a draft pick"""
        # Calculate round
        teams_count = self.config['basic_settings']['teams']
        round_num = math.ceil(self.current_pick / teams_count)
        
        # Create pick
        pick = Pick(team_id, player, self.current_pick, round_num)
        self.picks.append(pick)
        
        # Add player to team
        self.teams[team_id].add_player(player)
        
        # Advance pick
        self.current_pick += 1
    
    def undo_pick(self, player: Player):
        """Undo the last pick (for corrections)"""
        # Find and remove the pick
        pick_to_remove = None
        for pick in reversed(self.picks):
            if pick.player.id == player.id:
                pick_to_remove = pick
                break
        
        if pick_to_remove:
            self.picks.remove(pick_to_remove)
            self.teams[pick_to_remove.team_id].remove_player(player)
            self.current_pick -= 1
    
    def get_team_on_clock(self) -> int:
        """Get which team is currently on the clock"""
        teams_count = self.config['basic_settings']['teams']
        round_num = math.ceil(self.current_pick / teams_count)
        pick_in_round = ((self.current_pick - 1) % teams_count) + 1
        
        if round_num % 2 == 1:  # Odd round (1, 3, 5...)
            return pick_in_round
        else:  # Even round (2, 4, 6...)
            return teams_count - pick_in_round + 1
    
    def get_current_pick_info(self) -> dict:
        """Get detailed info about current pick"""
        teams_count = self.config['basic_settings']['teams']
        round_num = math.ceil(self.current_pick / teams_count)
        team_on_clock = self.get_team_on_clock()
        
        return {
            'pick': self.current_pick,
            'round': round_num,
            'team_on_clock': team_on_clock,
            'is_your_turn': team_on_clock == self.user_team_id
        }
    
    def get_your_next_pick(self) -> int:
        """Calculate your next pick number"""
        teams_count = self.config['basic_settings']['teams']
        current_round = math.ceil(self.current_pick / teams_count)
        
        # Find next pick for user team
        for pick_num in range(self.current_pick, self.current_pick + teams_count * 2):
            round_num = math.ceil(pick_num / teams_count)
            pick_in_round = ((pick_num - 1) % teams_count) + 1
            
            if round_num % 2 == 1:  # Odd round
                team_id = pick_in_round
            else:  # Even round
                team_id = teams_count - pick_in_round + 1
            
            if team_id == self.user_team_id:
                return pick_num
        
        return None
    
    def get_picks_until_your_turn(self) -> int:
        """How many picks until your next turn"""
        next_pick = self.get_your_next_pick()
        if next_pick:
            return next_pick - self.current_pick
        return float('inf')
    
    def get_user_roster(self) -> Dict[str, List[Player]]:
        """Get the user's current roster"""
        return self.teams[self.user_team_id].roster
    
    def get_pick_by_number(self, pick_number: int) -> Optional[Pick]:
        """Get pick by its number"""
        for pick in self.picks:
            if pick.pick_number == pick_number:
                return pick
        return None
    
    def get_drafted_player_ids(self) -> set:
        """Get set of player IDs that have been drafted"""
        drafted_ids = {pick.player.id for pick in self.picks}
        return drafted_ids
    
    def get_teams_needing_position(self, position: str) -> List[Team]:
        """Get teams that still need players at this position"""
        return [team for team in self.teams.values() if team.needs_position(position)]

class DraftIntelligence:
    def __init__(self, config: dict, players_df: pd.DataFrame):
        self.config = config
        self.players_df = players_df
        self.players_dict = self._create_players_dict()
        self.position_tiers = self._calculate_position_tiers()
    
    def _create_players_dict(self) -> Dict[str, Player]:
        """Convert DataFrame to Player objects"""
        players = {}
        
        for idx, row in self.players_df.iterrows():
            # Get name from either mapped column or original
            name = row.get('UNNAMED:_0_LEVEL_0_PLAYER', row.get('Player', f'Player_{idx}'))
            position = row.get('POSITION', row.get('Position', 'UNKNOWN'))
            player_id = f"{name}_{position}_{idx}"
            
            # Extract team from the Team column or name if present
            team = row.get('TEAM', row.get('Team', 'UNK'))
            if team == 'UNK':
                # Try to extract from name if no team column
                name_parts = name.split()
                team = name_parts[-1] if len(name_parts) > 1 and name_parts[-1].isupper() else "UNK"
            
            # Calculate tier based on VBD or fantasy points
            vbd_score = row.get('FANTASY_PTS', row.get('Custom_VBD', 0))
            if vbd_score == 0:
                vbd_score = row.get('VBD', 0)  # Fallback to other possible columns
                
            tier = self._calculate_tier(position, vbd_score)
            
            player = Player(
                id=player_id,
                name=name,
                position=position,
                team=team,
                fantasy_pts=row.get('FANTASY_PTS', row.get('Custom_VBD', 0)),
                vbd=vbd_score,
                tier=tier,
                adp=row.get('ECR', row.get('ADP', None)),
                bye_week=row.get('BYE_WEEK', row.get('Bye', None))
            )
            
            players[player_id] = player
        
        return players
    
    def _calculate_tier(self, position: str, vbd_score: float) -> int:
        """Calculate tier based on VBD score and position"""
        if vbd_score >= 100:
            return 1
        elif vbd_score >= 70:
            return 2
        elif vbd_score >= 50:
            return 3
        elif vbd_score >= 30:
            return 4
        else:
            return 5
    
    def _calculate_position_tiers(self) -> Dict[str, List[float]]:
        """Calculate tier break points for each position"""
        tiers = {}
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            # Try different column names for position
            pos_col = 'POSITION' if 'POSITION' in self.players_df.columns else 'Position'
            pos_players = self.players_df[self.players_df[pos_col] == position]
            
            if not pos_players.empty:
                # Try different column names for scores
                score_col = None
                for col in ['FANTASY_PTS', 'Custom_VBD', 'VBD', 'Fantasy_Points']:
                    if col in pos_players.columns:
                        score_col = col
                        break
                
                if score_col:
                    vbd_scores = pos_players[score_col].sort_values(ascending=False)
                    
                    # Define tier breaks based on VBD drops
                    tier_breaks = []
                    for i in range(1, min(len(vbd_scores), 5)):
                        if i < len(vbd_scores):
                            tier_breaks.append(vbd_scores.iloc[i])
                    
                    tiers[position] = tier_breaks
        
        return tiers
    
    def get_recommendations(self, draft_state: DraftState, top_n: int = 5) -> List[Tuple[Player, float, str]]:
        """Get top N recommended players with scores and reasoning"""
        recommendations = []
        drafted_ids = draft_state.get_drafted_player_ids()
        
        for player in self.players_dict.values():
            if player.id not in drafted_ids:
                score = self._calculate_recommendation_score(player, draft_state)
                reasoning = self._generate_reasoning(player, draft_state, score)
                recommendations.append((player, score, reasoning))
        
        # Sort by score and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_n]
    
    def _calculate_recommendation_score(self, player: Player, draft_state: DraftState) -> float:
        """Calculate comprehensive recommendation score"""
        base_score = player.vbd
        
        # Factor 1: Positional need
        position_need = self._calculate_position_need(player.position, draft_state)
        
        # Factor 2: Scarcity analysis
        scarcity_factor = self._calculate_scarcity(player, draft_state)
        
        # Factor 3: Tier urgency
        tier_urgency = self._calculate_tier_urgency(player, draft_state)
        
        # Factor 4: Draft round context
        round_context = self._get_round_context_weight(draft_state)
        
        return base_score * position_need * scarcity_factor * tier_urgency * round_context
    
    def _calculate_position_need(self, position: str, draft_state: DraftState) -> float:
        """Calculate positional need multiplier"""
        user_team = draft_state.teams[draft_state.user_team_id]
        current_count = user_team.get_position_count(position)
        required = user_team.get_starter_slots_needed(position)
        
        if current_count == 0 and required > 0:
            return 2.0  # Critical need
        elif current_count < required:
            return 1.5  # Still need starters
        elif current_count == required:
            return 1.0  # Adequate
        elif current_count < required + 2:
            return 0.7  # Some depth
        else:
            return 0.3  # Overstocked
    
    def _calculate_scarcity(self, player: Player, draft_state: DraftState) -> float:
        """Calculate position scarcity factor"""
        # Count remaining quality players at position (available = NOT drafted)
        drafted_ids = draft_state.get_drafted_player_ids()
        available_at_position = [
            p for p in self.players_dict.values()
            if p.position == player.position and p.id not in drafted_ids and p.tier <= 3
        ]
        
        # Count teams that still need this position
        teams_needing = len(draft_state.get_teams_needing_position(player.position))
        
        if len(available_at_position) == 0:
            return 2.0
        
        scarcity_ratio = teams_needing / len(available_at_position)
        
        if scarcity_ratio > 2.0:
            return 1.8
        elif scarcity_ratio > 1.5:
            return 1.4
        elif scarcity_ratio > 1.0:
            return 1.1
        else:
            return 1.0
    
    def _calculate_tier_urgency(self, player: Player, draft_state: DraftState) -> float:
        """Calculate tier break urgency"""
        picks_until_turn = draft_state.get_picks_until_your_turn()
        
        # Count players in same tier still available
        drafted_ids = draft_state.get_drafted_player_ids()
        same_tier_available = len([
            p for p in self.players_dict.values()
            if p.position == player.position and p.tier == player.tier and p.id not in drafted_ids
        ])
        
        # High urgency if last few in tier and many picks until your turn
        if same_tier_available <= 2 and picks_until_turn > 5:
            return 1.5
        elif same_tier_available <= 1:
            return 1.3
        else:
            return 1.0
    
    def _get_round_context_weight(self, draft_state: DraftState) -> float:
        """Adjust for draft round - early rounds favor talent, later favor need"""
        current_round = draft_state.get_current_pick_info()['round']
        
        if current_round <= 3:
            return 0.8  # Favor talent over need
        elif current_round <= 6:
            return 1.0  # Balanced
        elif current_round <= 10:
            return 1.2  # Favor need
        else:
            return 1.4  # Heavy need focus
    
    def _generate_reasoning(self, player: Player, draft_state: DraftState, score: float) -> str:
        """Generate human-readable reasoning for recommendation"""
        reasons = []
        
        # Check positional need
        user_team = draft_state.teams[draft_state.user_team_id]
        current_count = user_team.get_position_count(player.position)
        required = user_team.get_starter_slots_needed(player.position)
        
        if current_count == 0:
            reasons.append(f"Critical {player.position} need")
        elif current_count < required:
            reasons.append(f"Need {player.position} starter")
        
        # Check tier status
        if player.tier == 1:
            reasons.append("Elite tier player")
        elif player.tier == 2:
            reasons.append("High-quality option")
        
        # Check scarcity
        drafted_ids = draft_state.get_drafted_player_ids()
        same_tier_left = len([
            p for p in self.players_dict.values()
            if p.position == player.position and p.tier == player.tier and p.id not in drafted_ids
        ])
        
        if same_tier_left <= 2:
            reasons.append(f"Last tier {player.tier} {player.position}")
        
        # Check value
        if player.vbd > 80:
            reasons.append("High value")
        
        return " • ".join(reasons) if reasons else "Solid pick"
    
    def get_available_players_with_scores(self, draft_state: DraftState, 
                                        position_filter: str = None,
                                        tier_filter: str = None,
                                        sort_by: str = "Smart Score") -> List[Tuple[Player, float]]:
        """Get available players with recommendation scores"""
        drafted_ids = draft_state.get_drafted_player_ids()
        available_players = []
        
        for player in self.players_dict.values():
            if player.id in drafted_ids:
                continue
                
            # Apply filters
            if position_filter and player.position != position_filter:
                continue
            if tier_filter and str(player.tier) != tier_filter:
                continue
            
            # Calculate score based on sort preference
            if sort_by == "Smart Score":
                score = self._calculate_recommendation_score(player, draft_state)
            elif sort_by == "VBD":
                score = player.vbd
            elif sort_by == "Fantasy Points":
                score = player.fantasy_pts
            elif sort_by == "ADP":
                score = -player.adp if player.adp else 999  # Lower ADP = higher priority
            else:
                score = player.vbd
            
            available_players.append((player, score))
        
        # Sort by score
        available_players.sort(key=lambda x: x[1], reverse=True)
        return available_players