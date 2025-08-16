#!/usr/bin/env python3
"""
Debug script to trace Dynamic VBD execution step by step
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

# Import modules
try:
    from dynamic_vbd import DynamicVBDTransformer, create_probability_forecast, create_draft_state
    from scoring import load_league_config
    print("✅ Successfully imported all modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def debug_dynamic_vbd():
    """Debug Dynamic VBD step by step"""
    
    print("=== DYNAMIC VBD DEBUG TRACE ===\n")
    
    # Load configuration and data
    config = load_league_config('config/league-config.yaml')
    
    # Load data
    vbd_files = [f for f in os.listdir('data/output/') if 'vbd_rankings_top300' in f and f.endswith('.csv')]
    latest_vbd = sorted(vbd_files)[-1]
    df = pd.read_csv(f'data/output/{latest_vbd}')
    
    print(f"Loaded {len(df)} players")
    print(f"Config scale: {config['dynamic_vbd']['params']['scale']}")
    
    # Initialize transformer  
    transformer = DynamicVBDTransformer(config)
    
    # Create test scenario
    probabilities = create_probability_forecast(14, {'RB': 0.60, 'WR': 0.25, 'QB': 0.05, 'TE': 0.08})
    draft_state = create_draft_state(50, ['Player1', 'Player2'])
    
    print(f"\nTest scenario: RB probability {probabilities.position_probs['RB']}")
    
    # Step 1: Check RB data availability
    rb_df = df[df['POSITION'] == 'RB'].copy()
    rb_df = rb_df.sort_values('FANTASY_PTS', ascending=False).reset_index(drop=True)
    
    print(f"\n1. RB DATA ANALYSIS:")
    print(f"   Total RB players: {len(rb_df)}")
    print(f"   Top 5 RBs:")
    for i in range(min(5, len(rb_df))):
        print(f"     {i+1}. {rb_df.iloc[i]['PLAYER']}: {rb_df.iloc[i]['FANTASY_PTS']:.1f} pts")
    
    # Step 2: Check BEER baseline calculation
    teams = config.get('basic_settings', {}).get('teams', 14)
    baseline_idx = int(teams * 2.5) - 1  # RB has 2 starters
    
    print(f"\n2. BEER BASELINE CALCULATION:")
    print(f"   Teams: {teams}")
    print(f"   RB starters: 2")
    print(f"   BEER baseline rank: {baseline_idx + 1} (index {baseline_idx})")
    
    if baseline_idx < len(rb_df) - 1:
        pts_at_baseline = rb_df.iloc[baseline_idx]['FANTASY_PTS']
        pts_at_next = rb_df.iloc[baseline_idx + 1]['FANTASY_PTS']
        point_diff = pts_at_baseline - pts_at_next
        
        print(f"   Player at baseline: {rb_df.iloc[baseline_idx]['PLAYER']} ({pts_at_baseline:.1f} pts)")
        print(f"   Next player: {rb_df.iloc[baseline_idx + 1]['PLAYER']} ({pts_at_next:.1f} pts)")
        print(f"   Point difference: {point_diff:.1f}")
    else:
        print(f"   ❌ ISSUE: Baseline index {baseline_idx} >= available players {len(rb_df)}")
        print(f"   Using last player as baseline")
        
    # Step 3: Manual adjustment calculation
    print(f"\n3. ADJUSTMENT CALCULATION:")
    expected_picks = probabilities.position_probs['RB'] * probabilities.horizon_picks
    adjustment = transformer.scale * np.tanh(expected_picks / transformer.kappa)
    
    print(f"   Expected picks: {expected_picks:.1f}")
    print(f"   Adjustment: {adjustment:.3f}")
    
    # Step 4: Call the actual Dynamic VBD computation
    print(f"\n4. DYNAMIC VBD COMPUTATION:")
    
    # Enable debug logging temporarily
    original_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        baseline_overrides = transformer._compute_adjustments(df.copy(), probabilities)
        print(f"   Generated overrides: {baseline_overrides}")
        
        # Check what happens in the transform
        df_transformed = transformer.transform(df.copy(), probabilities, draft_state)
        
        # Compare VBD values
        original_rb = df[df['POSITION'] == 'RB'].nlargest(1, 'VBD_BEER')
        transformed_rb = df_transformed[df_transformed['POSITION'] == 'RB'].nlargest(1, 'VBD_BEER')
        
        if not original_rb.empty and not transformed_rb.empty:
            orig_vbd = original_rb.iloc[0]['VBD_BEER']
            trans_vbd = transformed_rb.iloc[0]['VBD_BEER']
            
            print(f"\n5. VBD COMPARISON:")
            print(f"   Original VBD: {orig_vbd:.2f}")
            print(f"   Transformed VBD: {trans_vbd:.2f}")
            print(f"   Difference: {trans_vbd - orig_vbd:+.2f}")
            
            if abs(trans_vbd - orig_vbd) < 0.1:
                print(f"   ❌ NO CHANGE DETECTED")
                
                # Check if baseline overrides are empty
                if not baseline_overrides:
                    print(f"   Issue: No baseline overrides generated")
                elif 'RB' not in baseline_overrides:
                    print(f"   Issue: No RB overrides in baseline_overrides")
                else:
                    print(f"   Issue: Overrides present but not applied correctly")
                    
            else:
                print(f"   ✅ CHANGE DETECTED")
                
    except Exception as e:
        print(f"   ❌ Error in computation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logging.getLogger().setLevel(original_level)


if __name__ == "__main__":
    debug_dynamic_vbd()