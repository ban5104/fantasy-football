"""
Fantasy Football Draft API
FastAPI backend for real-time draft management and probability calculations
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import json
from pathlib import Path

app = FastAPI(title="Fantasy Football Draft API")

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
ESPN_DATA = DATA_DIR / "espn_projections_20250814.csv"
ADP_DATA = DATA_DIR / "fantasypros_adp_20250815.csv"
VBD_DATA = Path(__file__).parent.parent / "draft_cheat_sheet.csv"

# Draft state
class DraftState:
    def __init__(self):
        self.current_pick = 1
        self.drafted_players = set()
        self.my_picks = [8, 17, 32, 41, 56, 65, 80, 89]
        self.connections = []
        
draft_state = DraftState()

def load_data():
    """Load and merge all data sources"""
    # Load ESPN projections
    espn_df = pd.read_csv(ESPN_DATA)
    
    # Load ADP data
    adp_df = pd.read_csv(ADP_DATA)
    
    # Load VBD scores
    vbd_df = pd.read_csv(VBD_DATA)
    
    # Merge datasets
    df = espn_df.merge(
        adp_df[['Player', 'ADP', 'Rank']], 
        left_on='player_name', 
        right_on='Player', 
        how='left'
    )
    
    df = df.merge(
        vbd_df[['Player', 'Custom_VBD', 'Draft_Rank']], 
        left_on='player_name', 
        right_on='Player', 
        how='left'
    )
    
    return df

def compute_softmax_scores(rank_series, tau=5.0):
    """Convert rankings to probability scores using softmax"""
    scores = np.exp(-rank_series / tau)
    return scores

def compute_pick_probabilities(available_df, espn_weight=0.8, adp_weight=0.2):
    """Calculate pick probabilities using 80% ESPN + 20% ADP"""
    if len(available_df) == 0:
        return pd.Series()
    
    # Get ESPN scores
    espn_scores = compute_softmax_scores(available_df['overall_rank'].fillna(300))
    
    # Get ADP scores
    adp_scores = compute_softmax_scores(available_df['Rank'].fillna(300))
    
    # Combine with weights
    combined_scores = espn_weight * espn_scores.values + adp_weight * adp_scores.values
    
    # Normalize to probabilities
    probs = combined_scores / combined_scores.sum()
    
    return pd.Series(probs, index=available_df.index)

def calculate_availability_probability(player_rank, current_pick, target_pick):
    """Calculate probability player is available at target pick"""
    picks_between = target_pick - current_pick
    
    if picks_between <= 0:
        return 0.0
    
    # Use survival probability calculation
    survival_prob = 1.0
    for i in range(picks_between):
        pick_prob = 1.0 / max(1, (player_rank - current_pick - i))
        survival_prob *= (1 - pick_prob)
    
    return survival_prob * 100

@app.get("/api/players")
async def get_players(
    current_pick: Optional[int] = None,
    position_filter: Optional[str] = None
):
    """Get all players with calculated metrics"""
    df = load_data()
    
    # Filter out drafted players
    df = df[~df['player_name'].isin(draft_state.drafted_players)]
    
    # Apply position filter if provided
    if position_filter and position_filter != "ALL":
        df = df[df['position'] == position_filter]
    
    # Calculate probabilities
    pick_probs = compute_pick_probabilities(df)
    df['pick_probability'] = pick_probs * 100
    
    # Calculate availability at my next picks
    current = current_pick or draft_state.current_pick
    
    for pick in draft_state.my_picks[:3]:  # Next 3 picks
        if pick > current:
            col_name = f'prob_pick_{pick}'
            df[col_name] = df['overall_rank'].apply(
                lambda r: calculate_availability_probability(r, current, pick)
            )
    
    # Calculate decision score
    df['decision_score'] = df['Custom_VBD'].fillna(0) * df.get(f'prob_pick_{draft_state.my_picks[0]}', 0) / 100
    
    # Sort by VBD rank
    df = df.sort_values('Draft_Rank', na_position='last')
    
    return df.head(100).to_dict('records')

@app.get("/api/draft-state")
async def get_draft_state():
    """Get current draft state"""
    return {
        "current_pick": draft_state.current_pick,
        "drafted_players": list(draft_state.drafted_players),
        "my_picks": draft_state.my_picks,
        "next_pick": next((p for p in draft_state.my_picks if p > draft_state.current_pick), None),
        "picks_until_next": next((p - draft_state.current_pick for p in draft_state.my_picks if p > draft_state.current_pick), None)
    }

@app.post("/api/draft-player")
async def draft_player(player_name: str):
    """Mark a player as drafted"""
    draft_state.drafted_players.add(player_name)
    draft_state.current_pick += 1
    
    # Notify all websocket connections
    for connection in draft_state.connections:
        await connection.send_json({
            "type": "player_drafted",
            "player": player_name,
            "current_pick": draft_state.current_pick
        })
    
    return {"success": True, "current_pick": draft_state.current_pick}

@app.post("/api/undo-draft")
async def undo_draft(player_name: str):
    """Remove a player from drafted list"""
    if player_name in draft_state.drafted_players:
        draft_state.drafted_players.remove(player_name)
        draft_state.current_pick = max(1, draft_state.current_pick - 1)
    
    return {"success": True, "current_pick": draft_state.current_pick}

@app.post("/api/set-pick")
async def set_current_pick(pick: int):
    """Manually set the current pick"""
    draft_state.current_pick = pick
    return {"success": True, "current_pick": draft_state.current_pick}

@app.post("/api/simulate")
async def simulate_pick(player_name: str, picks_until: int):
    """Run Monte Carlo simulation for a player"""
    df = load_data()
    
    # Get player rank
    player_row = df[df['player_name'] == player_name]
    if player_row.empty:
        return {"error": "Player not found"}
    
    player_rank = player_row['overall_rank'].values[0]
    
    # Run simulation
    simulations = 1000
    available_count = 0
    
    for _ in range(simulations):
        pick_sim = draft_state.current_pick
        drafted_sim = set(draft_state.drafted_players)
        
        for _ in range(picks_until):
            available = df[~df['player_name'].isin(drafted_sim)]
            if len(available) == 0:
                break
            
            # Pick based on probabilities
            probs = compute_pick_probabilities(available)
            picked_idx = np.random.choice(available.index, p=probs)
            drafted_sim.add(available.loc[picked_idx, 'player_name'])
            pick_sim += 1
            
            if available.loc[picked_idx, 'player_name'] == player_name:
                break
        
        if player_name not in drafted_sim:
            available_count += 1
    
    probability = (available_count / simulations) * 100
    
    return {
        "player": player_name,
        "picks_until": picks_until,
        "probability_available": probability,
        "simulations": simulations
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    draft_state.connections.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages if needed
    except WebSocketDisconnect:
        draft_state.connections.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)