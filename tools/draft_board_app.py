import streamlit as st
import pandas as pd
import numpy as np
import yaml
import json
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import os

# Set page config
st.set_page_config(
    page_title="Fantasy Football Draft Board",
    page_icon="🏈",
    layout="wide"
)

# Import our custom modules
from draft_engine import DraftState, DraftIntelligence, Player, Team
from data_processor import load_player_data, load_league_config

def main():
    st.title("🏈 Fantasy Football Draft Board")
    
    # Initialize session state
    if 'draft_state' not in st.session_state:
        st.session_state.draft_state = None
    if 'draft_intelligence' not in st.session_state:
        st.session_state.draft_intelligence = None
    
    # Load configuration and data
    try:
        config = load_league_config()
        players_df = load_player_data()
        
        if st.session_state.draft_intelligence is None:
            st.session_state.draft_intelligence = DraftIntelligence(config, players_df)
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Sidebar - Draft Setup
    with st.sidebar:
        st.header("🔧 Draft Setup")
        
        # Team selection
        team_names = config['team_names']
        user_team_idx = st.selectbox(
            "Your Team:",
            range(len(team_names)),
            format_func=lambda x: f"Team {x+1}: {team_names[x]}"
        )
        user_team_id = user_team_idx + 1
        
        # Draft position
        draft_position = st.number_input(
            "Your Draft Position:",
            min_value=1,
            max_value=config['basic_settings']['teams'],
            value=7
        )
        
        st.divider()
        
        # Draft state management
        st.header("💾 Draft Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("New Draft", type="primary"):
                st.session_state.draft_state = DraftState(config, user_team_id, draft_position)
                st.rerun()
        
        with col2:
            if st.button("Load Draft"):
                # TODO: Implement load functionality
                st.info("Load feature coming soon!")
        
        # Save current draft
        if st.session_state.draft_state:
            if st.button("Save Draft"):
                filename = save_draft_state(st.session_state.draft_state)
                st.success(f"Saved to {filename}")
    
    # Main interface
    if st.session_state.draft_state is None:
        st.info("👆 Click 'New Draft' in the sidebar to get started!")
        
        # Show preview of available data
        st.header("📊 Available Players Preview")
        st.dataframe(players_df[['UNNAMED:_0_LEVEL_0_PLAYER', 'POSITION', 'FANTASY_PTS']].head(20))
        
    else:
        render_draft_interface(
            st.session_state.draft_state,
            st.session_state.draft_intelligence,
            config
        )

def render_draft_interface(draft_state: DraftState, intelligence: DraftIntelligence, config: dict):
    """Main draft interface"""
    
    # Draft status header
    current_pick_info = draft_state.get_current_pick_info()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Round", current_pick_info['round'])
    with col2:
        st.metric("Pick", f"{current_pick_info['pick']}/224")
    with col3:
        if current_pick_info['is_your_turn']:
            st.metric("Status", "🟢 YOUR PICK")
        else:
            st.metric("On Clock", f"Team {current_pick_info['team_on_clock']}")
    with col4:
        st.metric("Next Pick", f"Pick {draft_state.get_your_next_pick()}")
    
    # Main content area
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        render_recommendations_and_players(draft_state, intelligence)
    
    with col_right:
        render_draft_board(draft_state, config)
        st.divider()
        render_your_roster(draft_state, config)

def render_recommendations_and_players(draft_state: DraftState, intelligence: DraftIntelligence):
    """Smart recommendations and player pool"""
    
    st.header("🎯 Smart Recommendations")
    
    # Get top recommendations
    recommendations = intelligence.get_recommendations(draft_state, top_n=5)
    
    if recommendations:
        for i, (player, score, reasoning) in enumerate(recommendations, 1):
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.write(f"**{i}. {player.name}** ({player.position}, {player.team})")
                    st.caption(reasoning)
                
                with col2:
                    st.write(f"Score: {score:.1f}")
                    st.write(f"Tier {player.tier} | VBD: {player.vbd:.1f}")
                
                with col3:
                    if st.button(f"Draft", key=f"rec_{player.id}"):
                        draft_player(draft_state, player)
                        st.rerun()
    
    st.divider()
    st.header("👥 Available Players")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        position_filter = st.selectbox("Position:", ["ALL", "QB", "RB", "WR", "TE", "K", "DST"])
    with col2:
        tier_filter = st.selectbox("Tier:", ["ALL", "1", "2", "3", "4", "5+"])
    with col3:
        sort_by = st.selectbox("Sort by:", ["Smart Score", "VBD", "Fantasy Points", "ADP"])
    
    # Get available players
    available_players = intelligence.get_available_players_with_scores(
        draft_state, 
        position_filter if position_filter != "ALL" else None,
        tier_filter if tier_filter != "ALL" else None,
        sort_by
    )
    
    # Player table
    for player, score in available_players[:30]:  # Show top 30
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
            
            with col1:
                st.write(f"**{player.name}** ({player.position}, {player.team})")
                if hasattr(player, 'bye_week'):
                    st.caption(f"Bye: {player.bye_week} | Proj: {player.fantasy_pts:.1f}")
            
            with col2:
                st.write(f"Tier {player.tier}")
            
            with col3:
                st.write(f"VBD: {player.vbd:.1f}")
            
            with col4:
                st.write(f"Score: {score:.1f}")
            
            with col5:
                # Team selector for pick
                team_for_pick = st.selectbox(
                    "Team:",
                    range(1, 15),
                    key=f"team_{player.id}",
                    index=draft_state.get_team_on_clock() - 1,
                    label_visibility="collapsed"
                )
                
                if st.button("Draft", key=f"draft_{player.id}"):
                    draft_state.make_pick(team_for_pick, player)
                    st.rerun()

def render_draft_board(draft_state: DraftState, config: dict):
    """Visual draft board"""
    st.header("📋 Draft Board")
    
    # Create draft grid
    picks_data = []
    total_rounds = 16  # Show first 16 rounds
    teams = config['basic_settings']['teams']
    
    for round_num in range(1, total_rounds + 1):
        round_picks = []
        
        # Snake draft order
        if round_num % 2 == 1:  # Odd rounds: 1->14
            team_order = list(range(1, teams + 1))
        else:  # Even rounds: 14->1
            team_order = list(range(teams, 0, -1))
        
        for team_id in team_order:
            pick_num = (round_num - 1) * teams + team_order.index(team_id) + 1
            
            pick = draft_state.get_pick_by_number(pick_num)
            if pick:
                round_picks.append(f"{pick.player.name[:12]}...")
            elif pick_num == draft_state.current_pick:
                round_picks.append("⏰")
            else:
                round_picks.append("---")
        
        picks_data.append(round_picks)
    
    # Display as dataframe
    df = pd.DataFrame(picks_data, columns=[f"T{i}" for i in range(1, teams + 1)])
    df.index = [f"R{i}" for i in range(1, total_rounds + 1)]
    
    # Highlight user's team
    user_col = f"T{draft_state.user_team_id}"
    
    st.dataframe(
        df,
        use_container_width=True,
        height=400
    )

def render_your_roster(draft_state: DraftState, config: dict):
    """Your current roster"""
    st.header("🏆 Your Roster")
    
    roster = draft_state.get_user_roster()
    roster_req = config['roster']['roster_slots']
    
    for position in ['QB', 'RB', 'WR', 'TE', 'DEF', 'K']:
        players = roster.get(position, [])
        required = roster_req.get(position, 0)
        
        st.write(f"**{position}** ({len(players)}/{required})")
        
        if players:
            for player in players:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"  • {player.name}")
                with col2:
                    if st.button("↺", key=f"undo_{player.id}", help="Undo pick"):
                        draft_state.undo_pick(player)
                        st.rerun()
        else:
            st.write("  • _Empty_")
    
    # Show bench
    bench = roster.get('BENCH', [])
    st.write(f"**BENCH** ({len(bench)}/7)")
    for player in bench:
        st.write(f"  • {player.name} ({player.position})")

def draft_player(draft_state: DraftState, player: Player):
    """Handle drafting a player"""
    team_on_clock = draft_state.get_team_on_clock()
    draft_state.make_pick(team_on_clock, player)

def save_draft_state(draft_state: DraftState) -> str:
    """Save draft state to file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"draft_state_{timestamp}.json"
    
    draft_data = {
        'picks': [
            {
                'team_id': pick.team_id,
                'player_id': pick.player.id,
                'pick_number': pick.pick_number,
                'round': pick.round,
                'timestamp': pick.timestamp.isoformat() if pick.timestamp else None
            }
            for pick in draft_state.picks
        ],
        'user_team_id': draft_state.user_team_id,
        'draft_position': draft_state.draft_position,
        'current_pick': draft_state.current_pick
    }
    
    with open(filename, 'w') as f:
        json.dump(draft_data, f, indent=2)
    
    return filename

if __name__ == "__main__":
    main()