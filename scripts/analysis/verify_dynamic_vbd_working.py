#!/usr/bin/env python3
"""
Final verification that Dynamic VBD is working correctly
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np

# Import modules
try:
    from dynamic_vbd import DynamicVBDTransformer, create_probability_forecast, create_draft_state
    from scoring import load_league_config
    print("✅ Successfully imported all modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_all_scenarios():
    """Test all three scenarios and verify they produce different rankings"""
    
    print("=== FINAL DYNAMIC VBD VERIFICATION ===\n")
    
    # Load configuration and data
    config = load_league_config('config/league-config.yaml')
    
    # Load data
    vbd_files = [f for f in os.listdir('data/output/') if 'vbd_rankings_top300' in f and f.endswith('.csv')]
    latest_vbd = sorted(vbd_files)[-1]
    df = pd.read_csv(f'data/output/{latest_vbd}')
    
    print(f"Using scale: {config['dynamic_vbd']['params']['scale']}")
    print(f"Dataset: {len(df)} players")
    
    # Initialize transformer
    transformer = DynamicVBDTransformer(config)
    
    # Define test scenarios with very different probabilities
    scenarios = {
        'Balanced': create_probability_forecast(28, {
            'RB': 0.35, 'WR': 0.30, 'QB': 0.15, 'TE': 0.15, 'DST': 0.03, 'K': 0.02
        }),
        'Heavy RB Run': create_probability_forecast(14, {
            'RB': 0.70, 'WR': 0.20, 'QB': 0.05, 'TE': 0.04, 'DST': 0.005, 'K': 0.005
        }),
        'WR/TE Focus': create_probability_forecast(7, {
            'RB': 0.10, 'WR': 0.50, 'QB': 0.10, 'TE': 0.28, 'DST': 0.01, 'K': 0.01
        })
    }
    
    draft_state = create_draft_state(50, ['Player1', 'Player2'])
    
    # Test each scenario
    results = {}
    for scenario_name, probabilities in scenarios.items():
        print(f"\n🔍 Testing {scenario_name} scenario...")
        
        # Get the probability for detailed output
        rb_prob = probabilities.position_probs['RB']
        wr_prob = probabilities.position_probs['WR']
        horizon = probabilities.horizon_picks
        
        print(f"   RB probability: {rb_prob:.1%}, WR probability: {wr_prob:.1%}")
        print(f"   Horizon: {horizon} picks")
        
        # Transform the data
        df_transformed = transformer.transform(df.copy(), probabilities, draft_state)
        
        # Get top players
        top_rb = df_transformed[df_transformed['POSITION'] == 'RB'].nlargest(1, 'VBD_BEER')
        top_wr = df_transformed[df_transformed['POSITION'] == 'WR'].nlargest(1, 'VBD_BEER')
        top_qb = df_transformed[df_transformed['POSITION'] == 'QB'].nlargest(1, 'VBD_BEER')
        
        # Store results
        results[scenario_name] = {
            'RB': {'player': top_rb.iloc[0]['PLAYER'], 'vbd': top_rb.iloc[0]['VBD_BEER']},
            'WR': {'player': top_wr.iloc[0]['PLAYER'], 'vbd': top_wr.iloc[0]['VBD_BEER']},
            'QB': {'player': top_qb.iloc[0]['PLAYER'], 'vbd': top_qb.iloc[0]['VBD_BEER']}
        }
        
        print(f"   Top RB VBD: {results[scenario_name]['RB']['vbd']:.1f}")
        print(f"   Top WR VBD: {results[scenario_name]['WR']['vbd']:.1f}")
        print(f"   Top QB VBD: {results[scenario_name]['QB']['vbd']:.1f}")
    
    # Analyze differences
    print(f"\n📊 SCENARIO COMPARISON:")
    print("=" * 80)
    print("Position | Balanced | Heavy RB | WR/TE    | Max Diff | Status")
    print("---------|----------|----------|----------|----------|--------")
    
    success = True
    for pos in ['RB', 'WR', 'QB']:
        values = [results[scenario][pos]['vbd'] for scenario in results.keys()]
        min_val = min(values)
        max_val = max(values)
        diff = max_val - min_val
        
        status = "✅ Good" if diff >= 5.0 else "❌ Small"
        if diff < 5.0:
            success = False
        
        print(f"{pos:8s} | {values[0]:8.1f} | {values[1]:8.1f} | {values[2]:8.1f} | {diff:8.1f} | {status}")
    
    # Overall assessment
    print("\n" + "=" * 80)
    if success:
        print("🎉 SUCCESS: Dynamic VBD is working correctly!")
        print("   All scenarios produce meaningfully different VBD values")
        print("   Rankings should now change based on draft probability forecasts")
        
        # Show some example ranking differences
        print(f"\n📋 Example ranking impact:")
        
        # Get combined rankings for each scenario
        for scenario_name in results.keys():
            df_scenario = transformer.transform(df.copy(), scenarios[scenario_name], draft_state)
            top_10 = df_scenario.nlargest(10, 'VBD_BEER')[['PLAYER', 'POSITION', 'VBD_BEER']]
            
            print(f"\n{scenario_name} top 5:")
            for i, row in top_10.head(5).iterrows():
                print(f"   {row.name+1}. {row['PLAYER'][:20]:20s} ({row['POSITION']}) - {row['VBD_BEER']:.1f}")
        
    else:
        print("⚠️  PARTIAL SUCCESS: Some differences detected but could be larger")
        print("   Consider further increasing scale parameter if more dramatic changes needed")
    
    return success


if __name__ == "__main__":
    test_all_scenarios()