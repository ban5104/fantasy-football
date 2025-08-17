#!/usr/bin/env python3
"""
Test script for Dynamic VBD parameter tuning
Tests different scale and kappa values to validate ranking differentiation
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass

# Import Dynamic VBD modules
try:
    from dynamic_vbd import DynamicVBDTransformer, create_probability_forecast, create_draft_state
    from scoring import load_league_config
    from vbd import calculate_all_vbd_methods
    print("✅ Successfully imported all modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def test_parameter_combination(scale: float, kappa: float, df: pd.DataFrame, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Test a specific scale/kappa combination and return ranking differences"""
    
    # Create modified config
    config = base_config.copy()
    config['dynamic_vbd']['params']['scale'] = scale
    config['dynamic_vbd']['params']['kappa'] = kappa
    
    # Initialize transformer
    transformer = DynamicVBDTransformer(config)
    
    # Define test scenarios
    scenarios = {
        'balanced': create_probability_forecast(28, {'RB': 0.35, 'WR': 0.30, 'QB': 0.15, 'TE': 0.15, 'DST': 0.03, 'K': 0.02}),
        'rb_run': create_probability_forecast(14, {'RB': 0.60, 'WR': 0.25, 'QB': 0.05, 'TE': 0.08, 'DST': 0.01, 'K': 0.01}),
        'wr_te': create_probability_forecast(7, {'RB': 0.15, 'WR': 0.45, 'QB': 0.10, 'TE': 0.25, 'DST': 0.03, 'K': 0.02})
    }
    
    draft_state = create_draft_state(50, ['Player_1', 'Player_2'])
    
    # Calculate rankings for each scenario
    rankings = {}
    vbd_values = {}
    
    for scenario_name, probabilities in scenarios.items():
        try:
            df_transformed = transformer.transform(df.copy(), probabilities, draft_state)
            df_ranked = df_transformed.sort_values('VBD_BEER', ascending=False).reset_index(drop=True)
            df_ranked['RANK'] = range(1, len(df_ranked) + 1)
            
            rankings[scenario_name] = df_ranked[['PLAYER', 'POSITION', 'VBD_BEER', 'RANK']].head(20)
            
            # Store VBD values for analysis
            top_rb = df_ranked[df_ranked['POSITION'] == 'RB'].iloc[0]['VBD_BEER']
            top_wr = df_ranked[df_ranked['POSITION'] == 'WR'].iloc[0]['VBD_BEER']
            vbd_values[scenario_name] = {'RB': top_rb, 'WR': top_wr}
            
        except Exception as e:
            print(f"Error in scenario {scenario_name}: {e}")
            return None
    
    # Calculate differences
    rb_diffs = []
    wr_diffs = []
    
    scenarios_list = list(vbd_values.keys())
    for i in range(len(scenarios_list)):
        for j in range(i + 1, len(scenarios_list)):
            s1, s2 = scenarios_list[i], scenarios_list[j]
            rb_diff = abs(vbd_values[s1]['RB'] - vbd_values[s2]['RB'])
            wr_diff = abs(vbd_values[s1]['WR'] - vbd_values[s2]['WR'])
            rb_diffs.append(rb_diff)
            wr_diffs.append(wr_diff)
    
    # Count ranking changes in top 10
    ranking_changes = 0
    balanced_top10 = set(rankings['balanced']['PLAYER'].head(10))
    rb_run_top10 = set(rankings['rb_run']['PLAYER'].head(10))
    wr_te_top10 = set(rankings['wr_te']['PLAYER'].head(10))
    
    ranking_changes += len(balanced_top10.symmetric_difference(rb_run_top10))
    ranking_changes += len(balanced_top10.symmetric_difference(wr_te_top10))
    ranking_changes += len(rb_run_top10.symmetric_difference(wr_te_top10))
    
    return {
        'scale': scale,
        'kappa': kappa,
        'max_rb_diff': max(rb_diffs) if rb_diffs else 0,
        'max_wr_diff': max(wr_diffs) if wr_diffs else 0,
        'avg_rb_diff': np.mean(rb_diffs) if rb_diffs else 0,
        'avg_wr_diff': np.mean(wr_diffs) if wr_diffs else 0,
        'ranking_changes': ranking_changes,
        'rankings': rankings,
        'vbd_values': vbd_values
    }


def main():
    """Main testing function"""
    print("=== DYNAMIC VBD PARAMETER TUNING TEST ===\n")
    
    # Load configuration and data
    try:
        config = load_league_config('config/league-config.yaml')
        print(f"✅ Loaded config: {config['basic_settings']['teams']} teams")
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return
    
    # Load or create test data
    try:
        # Try to load existing VBD data
        vbd_files = [f for f in os.listdir('data/output/') if 'vbd_rankings_top300' in f and f.endswith('.csv')]
        if vbd_files:
            latest_vbd = sorted(vbd_files)[-1]
            df = pd.read_csv(f'data/output/{latest_vbd}')
            print(f"✅ Loaded data: {latest_vbd} ({len(df)} players)")
        else:
            raise FileNotFoundError("No VBD files found")
            
    except Exception as e:
        print(f"⚠️  Data loading failed: {e}")
        print("📊 Creating synthetic test data...")
        
        # Create synthetic data for testing
        positions = ['RB', 'WR', 'QB', 'TE']
        players_per_pos = 25
        
        synthetic_data = []
        for pos in positions:
            base_pts = 200 if pos == 'QB' else 180 if pos == 'RB' else 160 if pos == 'WR' else 120
            for i in range(players_per_pos):
                synthetic_data.append({
                    'PLAYER': f'{pos} Player {i+1}',
                    'POSITION': pos,
                    'FANTASY_PTS': base_pts - (i * 3) + np.random.normal(0, 2),
                    'VBD_BEER': max(0, (base_pts - i * 3) - (base_pts * 0.6)),
                    'VBD_VORP': max(0, (base_pts - i * 3) - (base_pts * 0.5)),
                    'VBD_VOLS': max(0, (base_pts - i * 3) - (base_pts * 0.7)),
                    'VBD_BLENDED': max(0, (base_pts - i * 3) - (base_pts * 0.6))
                })
        
        df = pd.DataFrame(synthetic_data)
        print(f"✅ Created synthetic data: {len(df)} players")
    
    # Test parameter combinations
    test_params = [
        # Current (baseline)
        (3.0, 5.0),
        # Recommended from analysis
        (12.0, 5.0),
        (3.0, 1.2),
        (6.0, 1.9),
        # Additional test points
        (8.0, 3.0),
        (10.0, 2.0),
        (15.0, 5.0)
    ]
    
    print(f"\n📊 Testing {len(test_params)} parameter combinations...\n")
    
    results = []
    for scale, kappa in test_params:
        print(f"Testing scale={scale}, kappa={kappa}...")
        result = test_parameter_combination(scale, kappa, df, config)
        if result:
            results.append(result)
            print(f"  Max VBD diffs: RB={result['max_rb_diff']:.1f}, WR={result['max_wr_diff']:.1f}")
            print(f"  Ranking changes: {result['ranking_changes']}")
        else:
            print(f"  ❌ Failed")
        print()
    
    # Analysis and recommendations
    print("=" * 60)
    print("PARAMETER TESTING RESULTS")
    print("=" * 60)
    
    print("\nSummary Table:")
    print("Scale | Kappa | Max RB Diff | Max WR Diff | Ranking Changes | Status")
    print("------|-------|-------------|-------------|-----------------|--------")
    
    for r in results:
        status = "✅ Good" if r['max_rb_diff'] > 5 and r['max_wr_diff'] > 4 and r['ranking_changes'] > 2 else "❌ Insufficient"
        print(f"{r['scale']:5.1f} | {r['kappa']:5.1f} | {r['max_rb_diff']:11.1f} | {r['max_wr_diff']:11.1f} | {r['ranking_changes']:15d} | {status}")
    
    # Find best parameter combination
    best_result = max(results, key=lambda x: x['max_rb_diff'] + x['max_wr_diff'] + x['ranking_changes'])
    
    print(f"\n🎯 RECOMMENDED PARAMETERS:")
    print(f"   Scale: {best_result['scale']}")
    print(f"   Kappa: {best_result['kappa']}")
    print(f"   Expected RB VBD differences: {best_result['max_rb_diff']:.1f} points")
    print(f"   Expected WR VBD differences: {best_result['max_wr_diff']:.1f} points")
    print(f"   Expected ranking changes: {best_result['ranking_changes']} positions")
    
    # Show example ranking differences with best parameters
    print(f"\n📋 EXAMPLE RANKINGS WITH OPTIMAL PARAMETERS:")
    print(f"Scale={best_result['scale']}, Kappa={best_result['kappa']}")
    print("\nTop 10 Players by Scenario:")
    
    scenarios = ['balanced', 'rb_run', 'wr_te']
    for i in range(10):
        line = f"Rank {i+1:2d}: "
        for scenario in scenarios:
            if i < len(best_result['rankings'][scenario]):
                player = best_result['rankings'][scenario].iloc[i]
                line += f"{player['POSITION']}{player['PLAYER'].split()[-1]:2s} "
            else:
                line += "    "
        print(line)
    
    print(f"\n✅ Parameter tuning analysis complete!")
    print(f"💡 Update config/league-config.yaml with recommended parameters")


if __name__ == "__main__":
    main()