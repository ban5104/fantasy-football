"""
Championship DNA Analysis - Standalone Script
Run this in your Jupyter notebook to get complete analysis
"""

# Configuration (adjust as needed)
STRATEGY = 'balanced'  # or 'zero_rb', 'rb_heavy', 'hero_rb', 'elite_qb'
MY_PICK = 5            # Your draft position (1-14)
N_SIMS = 200           # Number of simulations

print("🧬 Championship DNA Analysis - Self-Contained")
print("=" * 60)

# 1. Import Championship DNA analyzer
import sys
sys.path.append('..' if 'notebooks' in str(sys.path) else '.')

try:
    from championship_dna_analyzer import ChampionshipDNA, run_championship_analysis
    print("✅ Championship DNA analyzer loaded")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root or notebooks directory")
    sys.exit(1)

# 2. Championship Blueprint Analysis
print(f"\n🔬 Analyzing {STRATEGY.upper()} strategy for Pick #{MY_PICK}")
print("=" * 50)

results = run_championship_analysis(strategy=STRATEGY, round_num=3)

if results:
    print("\n✅ Championship DNA analysis complete")
    print(f"📊 Analyzed top 10% performers from {STRATEGY} strategy")
    print("📋 Generated 3-card guidance system:")
    print("   • Championship Blueprint (ideal roster composition)")
    print("   • Pick Windows (round-based position probabilities)")  
    print("   • Pivot Alerts (tier scarcity warnings)")
    
    # Store results for further analysis
    north_star = results['north_star']
    windows = results['windows']
    tiers = results['tiers']
    alerts = results['alerts']
    
    # 3. Dynamic Round Analysis
    print("\n🔄 Dynamic Round-by-Round Analysis")
    print("=" * 50)
    
    analyzer = ChampionshipDNA()
    champions = analyzer.load_champions(strategy=STRATEGY, top_pct=0.1)
    
    if champions is not None:
        rounds_to_analyze = [1, 3, 5, 7]
        
        for round_num in rounds_to_analyze:
            print(f"\n--- ROUND {round_num} ANALYSIS ---")
            round_windows = analyzer.calculate_windows(champions, round_num)
            analyzer.display_windows(round_windows, round_num)
            
            if round_num < max(rounds_to_analyze):
                print("-" * 30)
    
    # 4. Advanced Tier Breakdown
    print("\n🎯 Advanced Tier Analysis by Position")
    print("=" * 50)
    
    if champions is not None:
        for pos in ['RB', 'WR', 'QB', 'TE']:
            if pos in north_star:
                print(f"\n{pos} TIER BREAKDOWN:")
                print("-" * 25)
                
                pos_tiers = analyzer.create_tiers(champions, pos)
                
                if pos_tiers:
                    # Group by tier
                    tier_groups = {}
                    for player, data in pos_tiers.items():
                        tier = data['tier']
                        if tier not in tier_groups:
                            tier_groups[tier] = []
                        tier_groups[tier].append((player, data))
                    
                    # Display each tier
                    for tier in sorted(tier_groups.keys()):
                        players = sorted(tier_groups[tier], key=lambda x: x[1]['champion_rate'], reverse=True)
                        print(f"  Tier {tier}: {len(players)} players")
                        
                        # Show top 3 players in tier
                        for i, (player, data) in enumerate(players[:3]):
                            print(f"    {i+1}. {player[:20]:20} {data['champion_rate']:.1%} champion rate")
                        
                        if len(players) > 3:
                            print(f"    ... and {len(players)-3} more")
                        print()
                else:
                    print(f"  No tier data available for {pos}")
    
    print("\n" + "="*60)
    print("🏁 Championship DNA Analysis Complete")
    print("="*60)
    
    print(f"\n📋 SUMMARY FOR {STRATEGY.upper()} STRATEGY:")
    print(f"🎯 North Star: {dict(north_star)}")
    print(f"📊 Round 3 Windows: {len(windows)} position types analyzed")  
    print(f"⚠️  Pivot Alerts: {len(alerts)} urgent recommendations")
    
else:
    print("❌ Championship DNA analysis failed")
    print(f"   No data found for strategy: {STRATEGY}")
    print(f"   Generate data with: python monte_carlo_runner.py export --strategy {STRATEGY} --pick {MY_PICK} --n-sims {N_SIMS}")