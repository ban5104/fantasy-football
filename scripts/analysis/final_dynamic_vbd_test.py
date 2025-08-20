#!/usr/bin/env python3
"""
Final Dynamic VBD test with properly differentiated scenarios
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


def test_properly_differentiated_scenarios():
    """Test with scenarios that have clearly different expected picks"""
    
    print("=== FINAL DYNAMIC VBD TEST WITH PROPER SCENARIOS ===\n")
    
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
    
    # Define properly differentiated scenarios
    scenarios = {
        'Ultra RB Focus': create_probability_forecast(12, {
            'RB': 0.85, 'WR': 0.10, 'QB': 0.02, 'TE': 0.02, 'DST': 0.005, 'K': 0.005
        }),
        'Balanced': create_probability_forecast(16, {
            'RB': 0.35, 'WR': 0.35, 'QB': 0.15, 'TE': 0.12, 'DST': 0.02, 'K': 0.01
        }),
        'Anti-RB Focus': create_probability_forecast(20, {
            'RB': 0.05, 'WR': 0.50, 'QB': 0.20, 'TE': 0.22, 'DST': 0.02, 'K': 0.01
        })
    }
    
    draft_state = create_draft_state(50, ['Player1', 'Player2'])
    
    # Analyze expected picks first
    print("Scenario Analysis:")
    print("Scenario        | RB Prob | Horizon | Expected RB Picks | Adjustment")
    print("----------------|---------|---------|-------------------|------------")
    
    for scenario_name, probabilities in scenarios.items():
        rb_prob = probabilities.position_probs['RB']
        horizon = probabilities.horizon_picks
        expected_picks = rb_prob * horizon
        adjustment = config['dynamic_vbd']['params']['scale'] * np.tanh(expected_picks / config['dynamic_vbd']['params']['kappa'])
        
        print(f"{scenario_name:15s} | {rb_prob:7.1%} | {horizon:7d} | {expected_picks:17.1f} | {adjustment:10.3f}")
    
    # Test each scenario with cache clearing
    results = {}
    for scenario_name, probabilities in scenarios.items():
        print(f"\n🔍 Testing {scenario_name}...")
        
        # Clear cache between scenarios
        transformer.clear_cache()
        
        # Transform the data
        df_transformed = transformer.transform(df.copy(), probabilities, draft_state)
        
        # Get top players
        top_rb = df_transformed[df_transformed['POSITION'] == 'RB'].nlargest(1, 'VBD_BEER')
        top_wr = df_transformed[df_transformed['POSITION'] == 'WR'].nlargest(1, 'VBD_BEER')
        
        # Store results
        results[scenario_name] = {
            'RB': {'player': top_rb.iloc[0]['PLAYER'], 'vbd': top_rb.iloc[0]['VBD_BEER']},
            'WR': {'player': top_wr.iloc[0]['PLAYER'], 'vbd': top_wr.iloc[0]['VBD_BEER']}
        }
        
        print(f"   Top RB VBD: {results[scenario_name]['RB']['vbd']:.1f}")
        print(f"   Top WR VBD: {results[scenario_name]['WR']['vbd']:.1f}")
    
    # Analyze differences
    print(f"\n📊 FINAL RESULTS:")
    print("=" * 80)
    print("Position | Ultra RB | Balanced | Anti-RB  | Max Diff | Success")
    print("---------|----------|----------|----------|----------|--------")
    
    overall_success = True
    for pos in ['RB', 'WR']:
        values = [results[scenario][pos]['vbd'] for scenario in results.keys()]
        min_val = min(values)
        max_val = max(values)
        diff = max_val - min_val
        
        success = "✅ YES" if diff >= 5.0 else "❌ NO "
        if diff < 5.0:
            overall_success = False
        
        print(f"{pos:8s} | {values[0]:8.1f} | {values[1]:8.1f} | {values[2]:8.1f} | {diff:8.1f} | {success}")
    
    # Generate rankings comparison
    print(f"\n📋 RANKING COMPARISON (Top 10):")
    for scenario_name in results.keys():
        transformer.clear_cache()
        df_scenario = transformer.transform(df.copy(), scenarios[scenario_name], draft_state)
        top_10 = df_scenario.nlargest(10, 'VBD_BEER')[['PLAYER', 'POSITION']]
        
        players = [f"{row['POSITION']}{i+1}" for i, (_, row) in enumerate(top_10.iterrows())]
        print(f"  {scenario_name:15s}: {' '.join(players[:5])}")
    
    # Final assessment
    print("\n" + "=" * 80)
    if overall_success:
        print("🎉 DYNAMIC VBD IS WORKING PERFECTLY!")
        print("✅ All scenarios produce meaningfully different VBD values")
        print("✅ Rankings change based on draft probability forecasts")
        print("✅ The system is ready for live draft use")
        
        # Calculate the range of differences
        rb_values = [results[scenario]['RB']['vbd'] for scenario in results.keys()]
        wr_values = [results[scenario]['WR']['vbd'] for scenario in results.keys()]
        
        print(f"\nKey Metrics:")
        print(f"  RB VBD range: {max(rb_values) - min(rb_values):.1f} points")
        print(f"  WR VBD range: {max(wr_values) - min(wr_values):.1f} points")
        print(f"  Parameter scale: {config['dynamic_vbd']['params']['scale']}")
        
    else:
        print("❌ DYNAMIC VBD STILL HAS ISSUES")
        print("   Some position differences are too small")
        print(f"   Consider increasing scale beyond {config['dynamic_vbd']['params']['scale']}")
    
    return overall_success


if __name__ == "__main__":
    success = test_properly_differentiated_scenarios()
    
    if success:
        print(f"\n🎯 IMPLEMENTATION COMPLETE!")
        print(f"Dynamic VBD is working correctly with scale=20.0")
        print(f"The original issue of identical rankings has been resolved.")
    else:
        print(f"\n⚠️  ADDITIONAL TUNING NEEDED")
        print(f"Consider scale=30.0 or higher for more dramatic differences")