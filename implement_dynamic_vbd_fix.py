#!/usr/bin/env python3
"""
Implementation script for Dynamic VBD parameter fix
Updates configuration and validates the fix works correctly
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import yaml
from datetime import datetime

# Import modules
try:
    from dynamic_vbd import DynamicVBDTransformer, create_probability_forecast, create_draft_state
    from scoring import load_league_config
    print("✅ Successfully imported all modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def update_config_file(new_scale: float = 20.0):
    """Update the league configuration file with new Dynamic VBD parameters"""
    config_path = 'config/league-config.yaml'
    
    try:
        # Load existing config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update dynamic VBD parameters
        if 'dynamic_vbd' not in config:
            config['dynamic_vbd'] = {}
        
        if 'params' not in config['dynamic_vbd']:
            config['dynamic_vbd']['params'] = {}
        
        old_scale = config['dynamic_vbd']['params'].get('scale', 3.0)
        config['dynamic_vbd']['params']['scale'] = new_scale
        config['dynamic_vbd']['enabled'] = True
        
        # Write back to file
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Updated {config_path}")
        print(f"   Scale changed from {old_scale} to {new_scale}")
        
        return config
        
    except Exception as e:
        print(f"❌ Failed to update config: {e}")
        return None


def test_dynamic_vbd_differences(config):
    """Test that Dynamic VBD now produces meaningful differences"""
    
    print("\n🧪 TESTING DYNAMIC VBD WITH NEW PARAMETERS")
    print("=" * 60)
    
    # Load data
    try:
        vbd_files = [f for f in os.listdir('data/output/') if 'vbd_rankings_top300' in f and f.endswith('.csv')]
        if vbd_files:
            latest_vbd = sorted(vbd_files)[-1]
            df = pd.read_csv(f'data/output/{latest_vbd}')
            print(f"✅ Loaded data: {latest_vbd} ({len(df)} players)")
        else:
            raise FileNotFoundError("No VBD files found")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False
    
    # Initialize transformer with new parameters
    transformer = DynamicVBDTransformer(config)
    print(f"Using scale={transformer.scale}, kappa={transformer.kappa}")
    
    # Define test scenarios
    scenarios = {
        'Balanced': create_probability_forecast(28, {
            'RB': 0.35, 'WR': 0.30, 'QB': 0.15, 'TE': 0.15, 'DST': 0.03, 'K': 0.02
        }),
        'RB Run': create_probability_forecast(14, {
            'RB': 0.60, 'WR': 0.25, 'QB': 0.05, 'TE': 0.08, 'DST': 0.01, 'K': 0.01
        }),
        'WR/TE Focus': create_probability_forecast(7, {
            'RB': 0.15, 'WR': 0.45, 'QB': 0.10, 'TE': 0.25, 'DST': 0.03, 'K': 0.02
        })
    }
    
    draft_state = create_draft_state(50, ['Player_1', 'Player_2'])
    
    # Calculate VBD for each scenario
    results = {}
    for scenario_name, probabilities in scenarios.items():
        try:
            df_transformed = transformer.transform(df.copy(), probabilities, draft_state)
            
            # Get top player VBD values for each position
            top_rb = df_transformed[df_transformed['POSITION'] == 'RB'].nlargest(1, 'VBD_BEER')
            top_wr = df_transformed[df_transformed['POSITION'] == 'WR'].nlargest(1, 'VBD_BEER')
            
            if not top_rb.empty and not top_wr.empty:
                results[scenario_name] = {
                    'RB': {
                        'player': top_rb.iloc[0]['PLAYER'],
                        'vbd': top_rb.iloc[0]['VBD_BEER']
                    },
                    'WR': {
                        'player': top_wr.iloc[0]['PLAYER'],
                        'vbd': top_wr.iloc[0]['VBD_BEER']
                    }
                }
            
        except Exception as e:
            print(f"❌ Error in scenario {scenario_name}: {e}")
            return False
    
    # Analyze differences
    print(f"\n📊 VBD RESULTS BY SCENARIO:")
    print("Scenario      | Top RB (VBD)           | Top WR (VBD)")
    print("--------------|------------------------|------------------------")
    
    for scenario_name, data in results.items():
        rb_info = f"{data['RB']['player'][:15]:15s} ({data['RB']['vbd']:6.1f})"
        wr_info = f"{data['WR']['player'][:15]:15s} ({data['WR']['vbd']:6.1f})"
        print(f"{scenario_name:13s} | {rb_info} | {wr_info}")
    
    # Calculate differences
    scenario_names = list(results.keys())
    rb_vbds = [results[s]['RB']['vbd'] for s in scenario_names]
    wr_vbds = [results[s]['WR']['vbd'] for s in scenario_names]
    
    rb_range = max(rb_vbds) - min(rb_vbds)
    wr_range = max(wr_vbds) - min(wr_vbds)
    
    print(f"\n📈 VBD DIFFERENCES:")
    print(f"RB VBD range: {rb_range:.1f} points (min: {min(rb_vbds):.1f}, max: {max(rb_vbds):.1f})")
    print(f"WR VBD range: {wr_range:.1f} points (min: {min(wr_vbds):.1f}, max: {max(wr_vbds):.1f})")
    
    # Determine success
    success_threshold = 10.0  # 10+ point differences indicate success
    
    if rb_range >= success_threshold or wr_range >= success_threshold:
        print(f"\n✅ SUCCESS: VBD differences are now meaningful!")
        print(f"   Target was {success_threshold}+ points, achieved {max(rb_range, wr_range):.1f}")
        return True
    else:
        print(f"\n❌ INSUFFICIENT: VBD differences still too small")
        print(f"   Target: {success_threshold}+ points, achieved: {max(rb_range, wr_range):.1f}")
        print(f"   Consider increasing scale further (current: {transformer.scale})")
        return False


def main():
    """Main implementation function"""
    print("=== DYNAMIC VBD PARAMETER FIX IMPLEMENTATION ===\n")
    
    # Step 1: Update configuration
    print("1️⃣ UPDATING CONFIGURATION")
    config = update_config_file(new_scale=20.0)
    if not config:
        print("❌ Failed to update configuration")
        return
    
    # Step 2: Test the fix
    print("\n2️⃣ TESTING THE FIX")
    success = test_dynamic_vbd_differences(config)
    
    # Step 3: Provide next steps
    print(f"\n3️⃣ NEXT STEPS")
    if success:
        print("✅ Dynamic VBD is now working correctly!")
        print("📝 Recommended actions:")
        print("   - Test with your draft scenarios in the notebook")
        print("   - Monitor for any over-adjustment in rankings")
        print("   - Consider expanding player database for better long-term stability")
    else:
        print("⚠️  Additional tuning needed:")
        print("   - Try scale=25.0 or 30.0 if differences are still too small")
        print("   - Consider expanding the player database")
        print("   - Check if dataset has sufficient player depth")
    
    print(f"\n🎯 ANALYSIS COMPLETE")
    print(f"Configuration updated at: config/league-config.yaml")
    print(f"Full analysis report: dynamic_vbd_analysis_report.md")


if __name__ == "__main__":
    main()