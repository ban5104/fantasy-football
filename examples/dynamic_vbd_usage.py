"""
Simplified Dynamic VBD usage example

Demonstrates core Dynamic VBD functionality with BEER method only.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import yaml
from dynamic_vbd import DynamicVBDTransformer, create_probability_forecast, create_draft_state


def load_config():
    """Load league configuration"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'league-config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_sample_projections():
    """Create sample player projections for demonstration"""
    return pd.DataFrame({
        'Player': [
            'Josh Allen', 'Lamar Jackson', 'Mahomes', 'Burrow',
            'CMC', 'Bijan Robinson', 'Saquon', 'Henry', 'Jacobs', 'Cook',
            'Jefferson', 'Hill', 'Diggs', 'Adams', 'Brown', 'DK',
            'Kelce', 'Andrews', 'Kittle', 'Hockenson',
            'Ravens DST', 'Bills DST', 'Steelers DST',
            'Tucker', 'Bass', 'McManus'
        ],
        'POSITION': (
            ['QB'] * 4 + 
            ['RB'] * 6 + 
            ['WR'] * 6 + 
            ['TE'] * 4 + 
            ['DEF'] * 3 + 
            ['K'] * 3
        ),
        'FANTASY_PTS': [
            # QBs
            330, 325, 315, 300,
            # RBs  
            280, 275, 270, 265, 260, 255,
            # WRs
            250, 245, 240, 235, 230, 225,
            # TEs
            180, 175, 170, 165,
            # DEF
            120, 115, 110,
            # K
            130, 125, 120
        ]
    })


def example_static_vs_dynamic_vbd():
    """
    Example comparing static VBD vs dynamic VBD rankings using BEER method
    """
    print("=== Simplified Dynamic VBD Usage Example ===\n")
    
    # Load configuration and projections
    config = load_config()
    projections_df = create_sample_projections()
    
    print(f"Loaded {len(projections_df)} player projections")
    print(f"Positions: {projections_df['POSITION'].value_counts().to_dict()}\n")
    
    # Initialize Dynamic VBD transformer
    transformer = DynamicVBDTransformer(config)
    print(f"Dynamic VBD enabled: {transformer.enabled}")
    print(f"Scale parameter: {transformer.scale}\n")
    
    # === SCENARIO 1: Early draft, balanced probabilities ===
    print("--- Scenario 1: Early Draft (Pick 15) ---")
    draft_state_early = create_draft_state(
        current_pick=15,
        drafted_players=['CMC', 'Jefferson', 'Josh Allen', 'Bijan Robinson']
    )
    
    # Balanced position probabilities over next 5 picks
    forecast_early = create_probability_forecast(
        horizon_picks=5,
        position_probs={'RB': 0.35, 'WR': 0.35, 'QB': 0.15, 'TE': 0.15}
    )
    
    # Get dynamic rankings
    dynamic_rankings_early = transformer.transform(
        projections_df, forecast_early, draft_state_early
    )
    
    if 'VBD_BEER' in dynamic_rankings_early.columns:
        top_players_early = dynamic_rankings_early.nlargest(10, 'VBD_BEER')
        print("Top 10 players by Dynamic VBD (BEER):")
        print(top_players_early[['Player', 'POSITION', 'VBD_BEER']].to_string(index=False))
    else:
        print("VBD columns not found - dynamic VBD may be disabled or integration issue")
    
    print("\n" + "="*50 + "\n")
    
    # === SCENARIO 2: RB run happening ===
    print("--- Scenario 2: RB Run in Progress (Pick 35) ---")
    draft_state_rb_run = create_draft_state(
        current_pick=35,
        drafted_players=[
            'CMC', 'Bijan Robinson', 'Saquon', 'Henry', 'Jacobs', 'Cook',  # Most top RBs gone
            'Jefferson', 'Hill', 'Diggs', 'Josh Allen', 'Kelce'
        ]
    )
    
    # High probability of continued RB drafting over next 3 picks
    forecast_rb_run = create_probability_forecast(
        horizon_picks=3,
        position_probs={'RB': 0.70, 'WR': 0.20, 'TE': 0.10}
    )
    
    # Filter to only available players (not drafted)
    available_players = projections_df[
        ~projections_df['Player'].isin(draft_state_rb_run.drafted_players)
    ].copy()
    
    dynamic_rankings_rb_run = transformer.transform(
        available_players, forecast_rb_run, draft_state_rb_run
    )
    
    if 'VBD_BEER' in dynamic_rankings_rb_run.columns:
        top_available = dynamic_rankings_rb_run.nlargest(8, 'VBD_BEER')
        print("Top 8 available players during RB run:")
        print(top_available[['Player', 'POSITION', 'VBD_BEER']].to_string(index=False))
        
        # Show how RB rankings changed
        rb_players = dynamic_rankings_rb_run[dynamic_rankings_rb_run['POSITION'] == 'RB']
        if not rb_players.empty:
            print(f"\nRemaining RBs ranked by Dynamic VBD:")
            print(rb_players.nlargest(5, 'VBD_BEER')[['Player', 'VBD_BEER']].to_string(index=False))
    
    # Cache statistics
    print(f"\nCache Statistics:")
    print(f"Cache size: {len(transformer._cache)}")


def example_basic_integration():
    """
    Basic integration example
    """
    print("\n" + "="*60)
    print("=== Basic Integration Example ===\n")
    
    # Load configuration and create sample data
    config = load_config()
    base_projections = create_sample_projections()
    
    # Mock current draft state
    current_draft_state = create_draft_state(
        current_pick=25,
        drafted_players=['CMC', 'Jefferson', 'Josh Allen']
    )
    
    # Get available players
    available_df = base_projections[
        ~base_projections['Player'].isin(current_draft_state.drafted_players)
    ].copy()
    
    # Simple position probabilities for next 5 picks
    position_probabilities = {'RB': 0.4, 'WR': 0.3, 'TE': 0.2, 'QB': 0.1}
    
    print(f"Position probabilities over next 5 picks:")
    for pos, prob in sorted(position_probabilities.items()):
        print(f"  {pos}: {prob:.1%}")
    
    # Create forecast object
    forecast = create_probability_forecast(
        horizon_picks=5,
        position_probs=position_probabilities
    )
    
    # Transform rankings using dynamic VBD
    transformer = DynamicVBDTransformer(config)
    dynamic_rankings = transformer.transform(available_df, forecast, current_draft_state)
    
    if 'VBD_BEER' in dynamic_rankings.columns:
        print(f"\nTop 10 available players with dynamic VBD:")
        top_10 = dynamic_rankings.nlargest(10, 'VBD_BEER')
        print(top_10[['Player', 'POSITION', 'VBD_BEER']].to_string(index=False))
    
    print("\n" + "="*60)


if __name__ == "__main__":
    try:
        example_static_vs_dynamic_vbd()
        example_basic_integration()
        
        print("\n✓ Simplified Dynamic VBD examples completed successfully!")
        print("\nNext steps:")
        print("1. Integrate with your existing probability forecasting system")
        print("2. Tune parameters (scale, kappa) based on your league's behavior")
        print("3. Test during live drafts")
        
    except Exception as e:
        print(f"\n✗ Example failed: {e}")
        import traceback
        traceback.print_exc()