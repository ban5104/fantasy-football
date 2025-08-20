"""
Efficient helper functions for Monte Carlo visualizations
Following best practices for fast re-runs
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

class SimulationCache:
    """Cache intermediate results for faster notebook re-runs"""
    
    def __init__(self, cache_dir=None):
        if cache_dir is None:
            # Use absolute path from project root
            base_path = Path(__file__).parent.parent
            self.cache_dir = base_path / 'data' / 'cache'
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
    
    def save(self, df, name):
        """Save DataFrame to parquet with atomic write"""
        filepath = self.cache_dir / f"{name}.parquet"
        temp_filepath = filepath.with_suffix('.parquet.tmp')
        
        # Write to temp file first
        df.to_parquet(temp_filepath)
        
        # Atomic rename
        temp_filepath.replace(filepath)
        
        print(f"💾 Cached {name} ({len(df)} rows)")
        return filepath
    
    def load(self, name):
        """Load DataFrame from cache if exists"""
        filepath = self.cache_dir / f"{name}.parquet"
        if filepath.exists():
            df = pd.read_parquet(filepath)
            print(f"📂 Loaded {name} from cache ({len(df)} rows)")
            return df
        return None
    
    def exists(self, name):
        """Check if cache exists"""
        return (self.cache_dir / f"{name}.parquet").exists()


def calculate_replacement_vectorized(df, num_teams=14):
    """Fully vectorized replacement value calculation"""
    
    starters_per_pos = {'RB': 2, 'WR': 2, 'QB': 1, 'TE': 1, 'K': 1, 'DST': 1}
    
    # Vectorized approach using nlargest
    def get_replacement_value(group, n_starters):
        rank = num_teams * n_starters
        sorted_vals = group.sort_values(ascending=False)
        if len(sorted_vals) >= rank:
            return sorted_vals.iloc[rank - 1]
        return sorted_vals.min() if len(sorted_vals) > 0 else 0
    
    # Calculate all replacement values at once
    replacement_values = []
    for pos, n_starters in starters_per_pos.items():
        pos_df = df[df['pos'] == pos]
        if len(pos_df) > 0:
            repl = pos_df.groupby('sim')['sampled_points'].apply(
                lambda x: get_replacement_value(x, n_starters)
            ).reset_index()
            repl['pos'] = pos
            replacement_values.append(repl)
    
    # Combine and merge
    repl_df = pd.concat(replacement_values, ignore_index=True)
    repl_df.columns = ['sim', 'replacement', 'pos']
    
    # Merge back efficiently
    df = df.merge(repl_df, on=['sim', 'pos'], how='left')
    
    # Vectorized bench marginal calculation
    df['bench_marginal'] = np.where(
        df['is_bench'],
        np.maximum(0, df['sampled_points'] - df['replacement']),
        0
    )
    
    return df, repl_df


def load_full_player_universe(base_path='..'):
    """Load full player universe with projections for position ranking"""
    import os
    
    # Load rankings data (full universe)
    rankings_file = os.path.join(base_path, 'data/rankings_top300_20250814.csv')
    espn_file = os.path.join(base_path, 'data/espn_projections_20250814.csv')
    
    full_players = None
    
    # Try rankings file first (has 300+ players)
    if os.path.exists(rankings_file):
        try:
            full_df = pd.read_csv(rankings_file)
            full_df['player_name'] = (full_df['PLAYER']
                                    .str.replace(r'\s+[A-Z]{2,3}$', '', regex=True)
                                    .str.strip())
            full_df['pos'] = full_df['POSITION'].str.extract(r'([A-Z]+)')[0]
            full_df['proj'] = full_df['FANTASY_PTS'].fillna(50)
            full_df['overall_rank'] = full_df['OVERALL_RANK']
            full_players = full_df[['player_name', 'pos', 'proj', 'overall_rank']].copy()
            print(f"✅ Loaded full player universe: {len(full_players)} players")
        except Exception as e:
            print(f"Warning: Error loading rankings file: {e}")
    
    # Fallback to ESPN file
    if full_players is None and os.path.exists(espn_file):
        try:
            espn_df = pd.read_csv(espn_file)
            espn_df['pos'] = espn_df['position'].str.extract(r'([A-Z]+)')[0]
            espn_df['proj'] = 100  # Default projection for ESPN-only data
            full_players = espn_df[['player_name', 'pos', 'proj', 'overall_rank']].copy()
            print(f"✅ Loaded ESPN player universe: {len(full_players)} players")
        except Exception as e:
            print(f"Warning: Error loading ESPN file: {e}")
    
    if full_players is None:
        print("❌ Could not load full player universe")
        return None
    
    # Calculate position ranks
    full_players['position_rank'] = full_players.groupby('pos')['proj'].rank(method='min', ascending=False)
    
    return full_players


def calculate_position_ranks_fixed(df, full_universe=None):
    """Calculate position ranks against full player universe, not just drafted players"""
    
    if full_universe is None:
        full_universe = load_full_player_universe()
        if full_universe is None:
            # Fallback to original method if can't load full universe
            print("⚠️ Using fallback position ranking (drafted players only)")
            df['pos_rank'] = df.groupby(['sim', 'pos'])['sampled_points'].rank(method='min', ascending=False)
            return df
    
    # Create player name to position rank mapping
    rank_mapping = dict(zip(full_universe['player_name'], full_universe['position_rank']))
    
    # Map position ranks
    df['pos_rank'] = df['player_name'].map(rank_mapping)
    
    # Fill missing ranks with high values (worst rank)
    max_rank_by_pos = full_universe.groupby('pos')['position_rank'].max().to_dict()
    for pos in df['pos'].unique():
        mask = (df['pos'] == pos) & df['pos_rank'].isna()
        if mask.any():
            df.loc[mask, 'pos_rank'] = max_rank_by_pos.get(pos, 999)
    
    print(f"✅ Fixed position ranks using full player universe ({len(full_universe)} players)")
    return df


def create_overview_summary(df, strategy, n_sims, elapsed_time):
    """Create comprehensive overview summary"""
    
    # Calculate key metrics
    roster_values = df.groupby('sim')['roster_value'].first()
    starter_values = df.groupby('sim')['starter_points'].first()
    bench_marginal = df[df['is_bench']].groupby('sim')['bench_marginal'].sum()
    
    # Statistics
    roster_mean = roster_values.mean()
    roster_ci = 1.96 * roster_values.std() / np.sqrt(n_sims)
    
    starter_mean = starter_values.mean()
    starter_ci = 1.96 * starter_values.std() / np.sqrt(n_sims)
    
    bench_mean = bench_marginal.mean()
    bench_ci = 1.96 * bench_marginal.std() / np.sqrt(n_sims)
    
    # VaR calculation
    var_10 = np.percentile(roster_values, 10)
    cvar_10 = roster_values[roster_values <= var_10].mean()
    
    # Print formatted summary
    print("="*70)
    print(f"📊 SIMULATION OVERVIEW - {strategy.upper()}")
    print("="*70)
    print(f"Simulations:    {n_sims}")
    print(f"Time Elapsed:   {elapsed_time:.2f}s")
    print(f"Random Seed:    42 (reproducible)")
    print("-"*70)
    print(f"Total Value:    {roster_mean:.1f} ± {roster_ci:.1f} (95% CI)")
    print(f"Starter Points: {starter_mean:.1f} ± {starter_ci:.1f}")
    print(f"Bench Marginal: {bench_mean:.1f} ± {bench_ci:.1f}")
    print("-"*70)
    print(f"VaR (10%):      {var_10:.1f}")
    print(f"CVaR (10%):     {cvar_10:.1f}")
    print(f"Downside Risk:  {roster_values[roster_values < roster_mean].std():.1f}")
    print("="*70)
    
    return {
        'roster_mean': roster_mean,
        'roster_ci': roster_ci,
        'starter_mean': starter_mean,
        'bench_mean': bench_mean,
        'var_10': var_10,
        'cvar_10': cvar_10
    }


def load_simulation_data(strategy=None, my_pick=None, n_sims=None, cache_dir='../data/cache'):
    """Load pre-saved simulation data from parquet files"""
    
    cache_path = Path(cache_dir)
    
    # Search for available parquet files
    pattern = "*.parquet"
    if strategy:
        pattern = f"{strategy}_*.parquet"
    
    available_files = list(cache_path.glob(pattern))
    
    if not available_files:
        print(f"❌ No simulation data found in {cache_path}")
        print("Run: python monte_carlo_runner.py export --strategy balanced --n-sims 200")
        return None
    
    # Filter by criteria if provided
    matching_files = []
    for file in available_files:
        # Try to match pattern: {strategy}_pick{N}_n{N}_r{N}.parquet
        # Handle strategies with underscores like zero_rb, rb_heavy
        filename = file.stem
        
        # Look for _pick pattern to split correctly
        if '_pick' not in filename:
            continue
            
        # Split at _pick to get strategy and rest
        parts = filename.split('_pick')
        if len(parts) != 2:
            continue
            
        try:
            file_strategy = parts[0]  # Could be "zero_rb", "rb_heavy", etc.
            
            # Parse the rest: 5_n200_r14
            rest_parts = parts[1].split('_')
            if len(rest_parts) < 2:
                continue
                
            file_pick = int(rest_parts[0])  # "5"
            
            # Find n{number} part
            n_part = None
            for part in rest_parts[1:]:
                if part.startswith('n'):
                    n_part = part
                    break
            
            if not n_part:
                continue
                
            file_n_sims = int(n_part[1:])  # Remove 'n' prefix
            
            if strategy and file_strategy != strategy:
                continue
            if my_pick and file_pick != my_pick:
                continue
            if n_sims and file_n_sims != n_sims:
                continue
                
            matching_files.append((file, file_strategy, file_pick, file_n_sims))
        except (ValueError, IndexError):
            # Skip files with malformed names
            continue
    
    if not matching_files:
        print(f"❌ No files match criteria: strategy={strategy}, pick={my_pick}, n_sims={n_sims}")
        print("Available files:")
        for file in available_files:
            print(f"  {file.name}")
        return None
    
    # Use the most recent matching file by timestamp
    best_file = max(matching_files, key=lambda x: x[0].stat().st_mtime)
    filepath, file_strategy, file_pick, file_n_sims = best_file
    
    # Get file timestamp for display
    import datetime
    timestamp = datetime.datetime.fromtimestamp(filepath.stat().st_mtime)
    
    # Show alternative files if multiple exist
    if len(matching_files) > 1:
        print(f"⚠️ Found {len(matching_files)} matching files - using LATEST by timestamp:")
        for i, (f, s, p, n) in enumerate(sorted(matching_files, key=lambda x: x[0].stat().st_mtime, reverse=True)):
            ts = datetime.datetime.fromtimestamp(f.stat().st_mtime)
            marker = "✅ USING" if i == 0 else "   older"
            print(f"   {marker}: {f.name} ({ts.strftime('%H:%M:%S')})")
        print()
    
    # Load data
    df = pd.read_parquet(filepath)
    
    print(f"📂 Loaded simulation data:")
    print(f"   File: {filepath.name}")
    print(f"   Created: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Strategy: {file_strategy}")
    print(f"   Pick: #{file_pick}")
    print(f"   Simulations: {file_n_sims}")
    print(f"   Records: {len(df)}")
    
    return df


def load_multiple_strategies(my_pick=5, n_sims=100, cache_dir='../data/cache'):
    """Load simulation data for all strategies at once"""
    
    strategies = ['balanced', 'zero_rb', 'rb_heavy', 'hero_rb', 'elite_qb']
    all_data = []
    
    print(f"📂 Loading data for all strategies (pick #{my_pick}, {n_sims} sims)...")
    
    for strategy in strategies:
        df = load_simulation_data(strategy, my_pick, n_sims, cache_dir)
        if df is not None:
            all_data.append(df)
            print(f"   ✅ {strategy}: {len(df)} records")
        else:
            print(f"   ❌ {strategy}: not found")
    
    if not all_data:
        print("\n❌ No strategy data found. Generate with:")
        for strategy in strategies:
            print(f"   python monte_carlo_runner.py export --strategy {strategy} --pick {my_pick} --n-sims {n_sims}")
        return None
    
    # Combine all strategies
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Combined data: {len(combined_df)} total records")
    
    return combined_df


def ensure_simulation_data_exists(strategies=['balanced'], my_pick=5, n_sims=100, cache_dir='../data/cache'):
    """Check if simulation data exists, provide generation commands if not"""
    
    cache_path = Path(cache_dir)
    missing_strategies = []
    
    for strategy in strategies:
        df = load_simulation_data(strategy, my_pick, n_sims, cache_dir)
        if df is None:
            missing_strategies.append(strategy)
    
    if missing_strategies:
        print(f"\n🚨 Missing simulation data for: {missing_strategies}")
        print("Generate with these commands:")
        for strategy in missing_strategies:
            print(f"   python monte_carlo_runner.py export --strategy {strategy} --pick {my_pick} --n-sims {n_sims}")
        
        return False
    
    return True


# ===== HIGH-IMPACT VISUALIZATIONS =====

def create_position_value_curves(df, full_universe=None, n_rounds=14):
    """Create position value decay curves with uncertainty bands"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    if full_universe is None:
        full_universe = load_full_player_universe()
        if full_universe is None:
            print("❌ Cannot create position curves without full player universe")
            return None
    
    # Check if we have simulation data for uncertainty bands
    has_sim_data = df is not None and len(df) > 0 and 'sampled_points' in df.columns
    
    # Calculate which round each position rank typically gets drafted
    position_curves = {}
    position_uncertainty = {}
    
    for pos in ['RB', 'WR', 'QB', 'TE']:
        pos_players = full_universe[full_universe['pos'] == pos].copy()
        if len(pos_players) == 0:
            continue
            
        # Sort by projection (higher = better)
        pos_players = pos_players.sort_values('proj', ascending=False).reset_index(drop=True)
        pos_players['pos_rank'] = range(1, len(pos_players) + 1)
        
        # Estimate round based on overall rank (assuming snake draft)
        # In 14-team league: picks 1-14 = round 1, 15-28 = round 2, etc.
        pos_players['estimated_round'] = ((pos_players['overall_rank'] - 1) // 14) + 1
        pos_players['estimated_round'] = pos_players['estimated_round'].clip(1, n_rounds)
        
        position_curves[pos] = pos_players[['pos_rank', 'proj', 'estimated_round']].head(50)  # Top 50 at position
        
        # Calculate uncertainty from simulation data if available
        if has_sim_data:
            pos_sim_data = df[df['pos'] == pos].copy()
            if len(pos_sim_data) > 0:
                # Group by player and get their average rank and projection percentiles
                player_stats = pos_sim_data.groupby('player_name')['sampled_points'].agg([
                    ('p10', lambda x: np.percentile(x, 10)),
                    ('p25', lambda x: np.percentile(x, 25)),
                    ('p50', lambda x: np.percentile(x, 50)),
                    ('p75', lambda x: np.percentile(x, 75)),
                    ('p90', lambda x: np.percentile(x, 90)),
                    ('mean', 'mean')
                ]).reset_index()
                
                # Sort by mean and assign position rank
                player_stats = player_stats.sort_values('mean', ascending=False).reset_index(drop=True)
                player_stats['pos_rank'] = range(1, len(player_stats) + 1)
                
                position_uncertainty[pos] = player_stats[['pos_rank', 'p10', 'p25', 'p50', 'p75', 'p90']].head(50)
    
    # Create visualization with uncertainty bands
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = {'RB': '#1f77b4', 'WR': '#ff7f0e', 'QB': '#2ca02c', 'TE': '#d62728'}
    
    for i, pos in enumerate(['RB', 'WR', 'QB', 'TE']):
        if pos not in position_curves:
            continue
            
        data = position_curves[pos]
        ax = axes[i]
        
        # If we have uncertainty data, use it
        if pos in position_uncertainty:
            uncert = position_uncertainty[pos]
            # Merge with position curve data
            merged = data.merge(uncert[['pos_rank', 'p10', 'p25', 'p50', 'p75', 'p90']], 
                              on='pos_rank', how='left')
            
            # Plot median line
            ax.plot(data['pos_rank'], data['proj'], '-', color=colors[pos], 
                   linewidth=2.5, label='Expected', zorder=3)
            
            # Add 50% confidence band (25th-75th percentile)
            valid_idx = ~merged['p25'].isna()
            if valid_idx.any():
                ax.fill_between(merged.loc[valid_idx, 'pos_rank'], 
                               merged.loc[valid_idx, 'p25'], 
                               merged.loc[valid_idx, 'p75'],
                               color=colors[pos], alpha=0.3, label='50% CI', zorder=1)
                
                # Add 80% confidence band (10th-90th percentile)
                ax.fill_between(merged.loc[valid_idx, 'pos_rank'], 
                               merged.loc[valid_idx, 'p10'], 
                               merged.loc[valid_idx, 'p90'],
                               color=colors[pos], alpha=0.15, label='80% CI', zorder=0)
        else:
            # No uncertainty data - just plot the line as before
            ax.plot(data['pos_rank'], data['proj'], 'o-', color=colors[pos], 
                   linewidth=2, markersize=4, label='Expected')
        
        ax.set_title(f'{pos} Value Decay {"with Uncertainty" if pos in position_uncertainty else ""}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Position Rank')
        ax.set_ylabel('Projected Points')
        ax.grid(True, alpha=0.3)
        
        # Add legend if we have uncertainty bands
        if pos in position_uncertainty:
            ax.legend(loc='upper right', fontsize=9)
        
        # Add round markers
        for round_num in range(1, min(8, n_rounds + 1)):  # Show first 7 rounds
            round_players = data[data['estimated_round'] == round_num]
            if len(round_players) > 0:
                ax.axvline(round_players['pos_rank'].iloc[0], color='gray', alpha=0.5, linestyle='--')
                ax.text(round_players['pos_rank'].iloc[0], ax.get_ylim()[1] * 0.9, 
                       f'R{round_num}', rotation=90, alpha=0.7, fontsize=8)
    
    plt.tight_layout()
    title = 'Position Value Decay with Confidence Intervals' if position_uncertainty else 'Position Value Decay by Round'
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    return fig


def create_availability_heatmap(all_strategies_df, top_n=50):
    """Create player availability heatmap showing when players get drafted"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if all_strategies_df is None or len(all_strategies_df) == 0:
        print("❌ No strategy data available for availability heatmap")
        return None
    
    # Calculate draft frequency for each player across all strategies
    player_freq = all_strategies_df.groupby(['player_name', 'pos']).agg({
        'sim': 'nunique',
        'sampled_points': 'mean'
    }).reset_index()
    
    total_sims = all_strategies_df['sim'].nunique() * all_strategies_df['strategy'].nunique()
    player_freq['draft_rate'] = player_freq['sim'] / total_sims
    
    # Get top players by draft frequency
    top_players = player_freq.nlargest(top_n, 'draft_rate')
    
    # Create heatmap data: player vs strategy
    strategies = all_strategies_df['strategy'].unique()
    heatmap_data = []
    
    for _, player in top_players.iterrows():
        player_name = player['player_name']
        row_data = {'Player': f"{player_name} ({player['pos']})", 'Overall': player['draft_rate']}
        
        for strategy in strategies:
            strategy_data = all_strategies_df[
                (all_strategies_df['player_name'] == player_name) & 
                (all_strategies_df['strategy'] == strategy)
            ]
            if len(strategy_data) > 0:
                strategy_sims = all_strategies_df[all_strategies_df['strategy'] == strategy]['sim'].nunique()
                rate = len(strategy_data) / strategy_sims
            else:
                rate = 0
            row_data[strategy.replace('_', ' ').title()] = rate
        
        heatmap_data.append(row_data)
    
    heatmap_df = pd.DataFrame(heatmap_data)
    
    # Create visualization
    plt.figure(figsize=(12, max(8, len(heatmap_data) * 0.3)))
    
    # Prepare data for seaborn (exclude Player and Overall columns for heatmap)
    strategy_cols = [col for col in heatmap_df.columns if col not in ['Player', 'Overall']]
    heatmap_values = heatmap_df[strategy_cols].values
    
    # Create heatmap
    sns.heatmap(
        heatmap_values,
        xticklabels=strategy_cols,
        yticklabels=heatmap_df['Player'],
        annot=True,
        fmt='.1%',
        cmap='YlOrRd',
        cbar_kws={'label': 'Draft Rate'}
    )
    
    plt.title(f'Player Availability Heatmap (Top {top_n} Most Drafted)', fontsize=14, fontweight='bold')
    plt.xlabel('Strategy')
    plt.ylabel('Player (Position)')
    plt.tight_layout()
    
    return plt.gcf()


def create_interactive_availability_heatmap(all_strategies_df, top_n=30):
    """Create interactive availability heatmap with round slider"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output
    except ImportError:
        print("⚠️ ipywidgets not available - falling back to static heatmap")
        return create_availability_heatmap(all_strategies_df, top_n)
    
    if all_strategies_df is None or len(all_strategies_df) == 0:
        print("❌ No strategy data available for availability heatmap")
        return None
    
    # Add round information to data
    df = all_strategies_df.copy()
    df['round'] = ((df.groupby(['strategy', 'sim']).cumcount()) // 1) + 1
    df['round'] = df['round'].clip(1, 14)
    
    # Create output widget
    output = widgets.Output()
    
    def update_heatmap(round_num):
        with output:
            clear_output(wait=True)
            
            if round_num == 0:
                # Show overall draft rate
                round_df = df
                title_suffix = "Overall"
            else:
                # Filter to specific round
                round_df = df[df['round'] <= round_num]
                title_suffix = f"Through Round {round_num}"
            
            # Calculate draft frequency for filtered data
            player_freq = round_df.groupby(['player_name', 'pos']).agg({
                'sim': 'nunique',
                'sampled_points': 'mean'
            }).reset_index()
            
            # Calculate rates based on filtered simulations
            strategies = round_df['strategy'].unique()
            n_sims_per_strategy = round_df.groupby('strategy')['sim'].nunique().max()
            
            player_freq['draft_rate'] = player_freq['sim'] / (len(strategies) * n_sims_per_strategy)
            
            # Get top players
            top_players = player_freq.nlargest(top_n, 'draft_rate')
            
            # Create heatmap data
            heatmap_data = []
            
            for _, player in top_players.iterrows():
                player_name = player['player_name']
                row_data = {'Player': f"{player_name[:20]} ({player['pos']})"} 
                
                for strategy in strategies:
                    strategy_data = round_df[
                        (round_df['player_name'] == player_name) & 
                        (round_df['strategy'] == strategy)
                    ]
                    if len(strategy_data) > 0:
                        strategy_sims = round_df[round_df['strategy'] == strategy]['sim'].nunique()
                        rate = strategy_data['sim'].nunique() / strategy_sims
                    else:
                        rate = 0
                    row_data[strategy.replace('_', ' ').title()] = rate
                
                heatmap_data.append(row_data)
            
            if not heatmap_data:
                print(f"No players drafted by round {round_num}")
                return
            
            heatmap_df = pd.DataFrame(heatmap_data)
            
            # Create visualization
            fig = plt.figure(figsize=(10, max(6, len(heatmap_data) * 0.25)))
            
            # Prepare data for seaborn
            strategy_cols = [col for col in heatmap_df.columns if col != 'Player']
            heatmap_values = heatmap_df[strategy_cols].values
            
            # Create heatmap with adjusted colormap
            sns.heatmap(
                heatmap_values,
                xticklabels=strategy_cols,
                yticklabels=heatmap_df['Player'],
                annot=True,
                fmt='.0%',
                cmap='YlOrRd',
                vmin=0,
                vmax=1,
                cbar_kws={'label': 'Draft Rate'},
                linewidths=0.5
            )
            
            plt.title(f'Player Availability - {title_suffix}', fontsize=14, fontweight='bold')
            plt.xlabel('Strategy')
            plt.ylabel('Player')
            plt.tight_layout()
            plt.show()
    
    # Create slider widget
    round_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=14,
        step=1,
        description='Round:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        style={'description_width': 'initial'}
    )
    
    # Add custom labels
    round_label = widgets.HTML(
        value='<b>0 = Overall | 1-14 = Through Round N</b>'
    )
    
    # Create interactive widget
    interactive_widget = widgets.interactive(update_heatmap, round_num=round_slider)
    
    # Display widgets
    display(widgets.VBox([
        widgets.HTML('<h3>🎯 Interactive Player Availability by Round</h3>'),
        round_label,
        interactive_widget.children[0],  # Slider
        output
    ]))
    
    # Initial update
    update_heatmap(0)
    
    return interactive_widget


def create_roster_timeline(all_strategies_df, strategy='balanced'):
    """Create roster construction timeline (stacked area chart)"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    if all_strategies_df is None:
        print("❌ No strategy data available for roster timeline")
        return None
    
    # Filter to specific strategy
    strategy_df = all_strategies_df[all_strategies_df['strategy'] == strategy].copy()
    if len(strategy_df) == 0:
        print(f"❌ No data for strategy: {strategy}")
        return None
    
    # Add round information (assuming 14-team league)
    # This is a simplified approach - in reality we'd need pick order from simulation
    strategy_df['round'] = ((strategy_df.groupby('sim').cumcount()) // 1) + 1
    strategy_df['round'] = strategy_df['round'].clip(1, 14)
    
    # Calculate position distribution by round
    position_by_round = strategy_df.groupby(['round', 'pos']).size().unstack(fill_value=0)
    
    # Convert to percentages
    position_pct = position_by_round.div(position_by_round.sum(axis=1), axis=0) * 100
    
    # Create stacked area chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    positions = ['RB', 'WR', 'QB', 'TE', 'K', 'DST']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Filter to available positions
    available_positions = [pos for pos in positions if pos in position_pct.columns]
    position_data = position_pct[available_positions]
    
    ax.stackplot(position_data.index, *[position_data[pos] for pos in available_positions], 
                labels=available_positions, colors=colors[:len(available_positions)], alpha=0.8)
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Percentage of Picks')
    ax.set_title(f'Roster Construction Timeline - {strategy.replace("_", " ").title()} Strategy', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, min(14, position_data.index.max()))
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    return fig


def create_strategy_radar_chart(all_strategies_df):
    """Create strategy differentiation radar chart"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    if all_strategies_df is None:
        print("❌ No strategy data available for radar chart")
        return None
    
    # Calculate strategy characteristics
    strategy_stats = {}
    
    for strategy in all_strategies_df['strategy'].unique():
        strategy_df = all_strategies_df[all_strategies_df['strategy'] == strategy]
        
        # Calculate metrics
        total_value = strategy_df.groupby('sim')['roster_value'].first()
        rb_count = strategy_df[strategy_df['pos'] == 'RB'].groupby('sim').size()
        wr_count = strategy_df[strategy_df['pos'] == 'WR'].groupby('sim').size()
        # Use QB average points as proxy for QB priority/quality
        qb_quality = strategy_df[strategy_df['pos'] == 'QB'].groupby('sim')['sampled_points'].mean()
        
        strategy_stats[strategy] = {
            'Expected Value': total_value.mean(),
            'Consistency': 1 / (total_value.std() / total_value.mean()) if total_value.std() > 0 else 1,
            'RB Heavy': rb_count.mean() if len(rb_count) > 0 else 0,
            'WR Heavy': wr_count.mean() if len(wr_count) > 0 else 0,
            'QB Priority': qb_quality.mean() / 10 if len(qb_quality) > 0 else 0,  # Scaled QB quality
            'Upside': total_value.quantile(0.9) - total_value.median()
        }
    
    # Normalize all metrics to 0-1 scale
    metrics = list(strategy_stats[list(strategy_stats.keys())[0]].keys())
    
    # Get min/max for normalization
    all_values = {metric: [] for metric in metrics}
    for strategy_data in strategy_stats.values():
        for metric, value in strategy_data.items():
            all_values[metric].append(value)
    
    # Normalize
    for strategy in strategy_stats:
        for metric in metrics:
            min_val = min(all_values[metric])
            max_val = max(all_values[metric])
            if max_val > min_val:
                strategy_stats[strategy][metric] = (strategy_stats[strategy][metric] - min_val) / (max_val - min_val)
            else:
                strategy_stats[strategy][metric] = 0.5
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Set up angles
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (strategy, stats) in enumerate(strategy_stats.items()):
        values = [stats[metric] for metric in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=strategy.replace('_', ' ').title(), 
                color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
    
    # Customize chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title('Strategy Differentiation Radar Chart', size=16, fontweight='bold', pad=20)
    
    return fig


def create_win_probability_curves(all_strategies_df):
    """Create win probability curves visualization"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    if all_strategies_df is None:
        print("❌ No strategy data available for win probability curves")
        return None
    
    # Calculate total values per sim per strategy
    strategy_values = {}
    for strategy in all_strategies_df['strategy'].unique():
        values = all_strategies_df[all_strategies_df['strategy'] == strategy].groupby('sim')['roster_value'].first()
        strategy_values[strategy] = values.values
    
    # Calculate win probabilities against each other
    strategies = list(strategy_values.keys())
    thresholds = np.linspace(1200, 1600, 100)  # Point thresholds
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Cumulative Distribution Functions
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (strategy, values) in enumerate(strategy_values.items()):
        # Calculate empirical CDF
        sorted_values = np.sort(values)
        y = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        
        ax1.plot(sorted_values, y, label=strategy.replace('_', ' ').title(), 
                linewidth=2, color=colors[i % len(colors)])
    
    ax1.set_xlabel('Roster Value (Points)')
    ax1.set_ylabel('Cumulative Probability')
    ax1.set_title('Cumulative Distribution Functions', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Head-to-head win probabilities
    win_matrix = np.zeros((len(strategies), len(thresholds)))
    
    for i, strategy in enumerate(strategies):
        values = strategy_values[strategy]
        for j, threshold in enumerate(thresholds):
            win_prob = np.mean(values >= threshold)
            win_matrix[i, j] = win_prob
    
    for i, strategy in enumerate(strategies):
        ax2.plot(thresholds, win_matrix[i], label=strategy.replace('_', ' ').title(), 
                linewidth=2, color=colors[i % len(colors)])
    
    ax2.set_xlabel('Point Threshold')
    ax2.set_ylabel('Probability of Exceeding Threshold')
    ax2.set_title('Win Probability Curves', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig


def create_decision_cards(sims_df, current_pick=1, top_n=3):
    """Create decision cards showing top recommendations with confidence intervals"""
    import numpy as np
    
    try:
        from IPython.display import display, HTML
    except ImportError:
        print("⚠️ IPython not available - returning text summary")
        return create_text_decision_summary(sims_df, current_pick, top_n)
    
    if sims_df is None or len(sims_df) == 0:
        print("❌ No simulation data available for decision cards")
        return None
    
    # Calculate player statistics with confidence intervals
    player_stats = sims_df.groupby(['player_name', 'pos']).agg({
        'sampled_points': ['mean', 'std', 
                          lambda x: np.percentile(x, 25),
                          lambda x: np.percentile(x, 75)],
        'is_starter': 'mean',
        'sim': 'nunique'
    }).reset_index()
    
    # Flatten column names
    player_stats.columns = ['player_name', 'pos', 'mean_points', 'std_points', 
                           'p25', 'p75', 'starter_rate', 'draft_count']
    
    # Calculate confidence interval
    player_stats['ci_lower'] = player_stats['mean_points'] - 1.96 * player_stats['std_points']
    player_stats['ci_upper'] = player_stats['mean_points'] + 1.96 * player_stats['std_points']
    
    # Sort by expected value
    player_stats = player_stats.sort_values('mean_points', ascending=False)
    
    # Get top N recommendations
    top_players = player_stats.head(top_n)
    
    # Create HTML cards
    html_content = """
    <div style="font-family: Arial, sans-serif; margin: 20px 0;">
        <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">
            🎯 Next Pick Recommendations (Pick #{})
        </h2>
        <div style="display: flex; gap: 20px; flex-wrap: wrap;">
    """.format(current_pick)
    
    position_colors = {
        'RB': '#1f77b4', 'WR': '#ff7f0e', 
        'QB': '#2ca02c', 'TE': '#d62728',
        'K': '#9467bd', 'DST': '#8c564b'
    }
    
    for i, player in top_players.iterrows():
        pos_color = position_colors.get(player['pos'], '#666')
        
        # Determine confidence level
        if player['std_points'] < 10:
            confidence = "High"
            conf_color = "#27ae60"
        elif player['std_points'] < 20:
            confidence = "Medium"
            conf_color = "#f39c12"
        else:
            confidence = "Low"
            conf_color = "#e74c3c"
        
        card_html = """
        <div style="border: 2px solid {}; border-radius: 10px; padding: 15px; 
                    width: 250px; background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h3 style="margin: 0; color: {};">#{}</h3>
                <span style="background: {}; color: white; padding: 3px 8px; 
                           border-radius: 15px; font-size: 12px; font-weight: bold;">
                    {}
                </span>
            </div>
            
            <h4 style="margin: 5px 0; color: #2c3e50; font-size: 18px;">
                {}
            </h4>
            
            <div style="margin: 10px 0;">
                <div style="color: #7f8c8d; font-size: 12px; margin-bottom: 5px;">
                    Expected Points
                </div>
                <div style="font-size: 24px; font-weight: bold; color: #2c3e50;">
                    {:.0f}
                </div>
                <div style="color: #95a5a6; font-size: 12px;">
                    ({:.0f} - {:.0f})
                </div>
            </div>
            
            <div style="display: flex; justify-content: space-between; margin-top: 15px; 
                       padding-top: 10px; border-top: 1px solid #ecf0f1;">
                <div>
                    <div style="color: #7f8c8d; font-size: 11px;">Starter %</div>
                    <div style="font-weight: bold; color: #34495e;">{:.0%}</div>
                </div>
                <div>
                    <div style="color: #7f8c8d; font-size: 11px;">Confidence</div>
                    <div style="font-weight: bold; color: {};">{}</div>
                </div>
            </div>
        </div>
        """.format(
            pos_color, pos_color, i+1, pos_color, player['pos'],
            player['player_name'][:20],
            player['mean_points'],
            player['p25'], player['p75'],
            player['starter_rate'],
            conf_color, confidence
        )
        
        html_content += card_html
    
    html_content += """
        </div>
        <div style="margin-top: 20px; padding: 10px; background: #ecf0f1; 
                   border-radius: 5px; font-size: 12px; color: #7f8c8d;">
            <strong>Note:</strong> Confidence intervals show 25th-75th percentile range. 
            Recommendations based on {} simulations.
        </div>
    </div>
    """.format(sims_df['sim'].nunique())
    
    display(HTML(html_content))
    
    return top_players


def create_text_decision_summary(sims_df, current_pick=1, top_n=3):
    """Create text-based decision summary when HTML is not available"""
    if sims_df is None or len(sims_df) == 0:
        return None
    
    # Calculate player statistics
    player_stats = sims_df.groupby(['player_name', 'pos']).agg({
        'sampled_points': ['mean', 'std'],
        'is_starter': 'mean',
        'sim': 'nunique'
    }).reset_index()
    
    player_stats.columns = ['player_name', 'pos', 'mean_points', 'std_points', 
                           'starter_rate', 'draft_count']
    
    # Sort and get top players
    top_players = player_stats.nlargest(top_n, 'mean_points')
    
    print(f"\n{'='*60}")
    print(f"🎯 TOP {top_n} RECOMMENDATIONS FOR PICK #{current_pick}")
    print(f"{'='*60}")
    
    for i, player in top_players.iterrows():
        print(f"\n{i+1}. {player['player_name']} ({player['pos']})")
        print(f"   Expected: {player['mean_points']:.0f} ± {player['std_points']:.0f} points")
        print(f"   Starter Rate: {player['starter_rate']:.0%}")
    
    print(f"\n{'='*60}")
    
    return top_players