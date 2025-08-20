# 🧬 Championship DNA System - Complete Guide

## Quick Start Commands

### 1. Generate Fresh Simulation Data (Required First!)
```bash
# Generate data with your draft position (pick 5 shown)
PYTHONPATH=. uv run python monte_carlo_runner.py export --strategy balanced --n-sims 1000 --pick 5

# For pick 14 (turn pick)
PYTHONPATH=. uv run python monte_carlo_runner.py export --strategy balanced --n-sims 1000 --pick 14
```

### 2. View Championship DNA Analysis

#### A. Command Line (Quick View)
```bash
# Basic analysis
PYTHONPATH=. uv run python notebooks/championship_dna_analyzer.py

# With specific parameters
PYTHONPATH=. uv run python -c "
from notebooks.championship_dna_analyzer import run_championship_analysis
# Analyze different rounds (1-14)
for round_num in [1, 3, 5, 7, 10]:
    print(f'\\n=== ROUND {round_num} ===')
    run_championship_analysis(strategy='balanced', round_num=round_num)
"
```

#### B. Jupyter Notebook (Interactive)
```bash
# Start Jupyter
jupyter notebook

# Then open: notebooks/championship_dna_notebook.ipynb
```

#### C. Hybrid System (Advanced)
```bash
# Pre-draft analysis with tier windows
PYTHONPATH=. uv run python notebooks/championship_dna_hybrid.py pre-draft --pick 5

# Live draft mode (during your draft)
PYTHONPATH=. uv run python notebooks/championship_dna_hybrid.py live --pick 5
```

## What You'll See

### 🎯 North Star Blueprint
```
QB: 2 players (≥2 Tier-2+)
RB: 4 players (≥2 Tier-2+)  
TE: 2 players (≥2 Tier-2+)
WR: 5 players (≥2 Tier-2+)
```
This is your target roster composition based on winning teams.

### 📊 Round-by-Round Windows
```
Round 1: RB: 100% chance
Round 2: RB: 70%, TE: 30%
Round 3: TE: 55%, QB: 45%
```
Shows what positions champions draft in each round.

### ⚠️ Pivot Alerts
```
• Only 2 Tier-1 RBs left → Prioritize RB now
• Position run detected: 5 WRs in last 6 picks
```
Real-time guidance on when to adjust strategy.

## Understanding the Data

The system analyzes your simulation data to find:
1. **Top 10% of rosters** by total points
2. **Common patterns** in these winning rosters
3. **Optimal draft sequences** that lead to championships

## Customization Options

### Different Strategies
```bash
# Compare strategies
for strategy in balanced zero_rb rb_heavy hero_rb elite_qb; do
    PYTHONPATH=. uv run python monte_carlo_runner.py export --strategy $strategy --n-sims 200 --pick 5
done

# Then analyze each
PYTHONPATH=. uv run python notebooks/championship_dna_analyzer.py --strategy zero_rb
```

### Different Draft Positions
```bash
# Analyze multiple draft slots
for pick in 1 5 7 10 14; do
    echo "Analyzing pick $pick..."
    PYTHONPATH=. uv run python monte_carlo_runner.py export --strategy balanced --n-sims 500 --pick $pick
done
```

## Jupyter Notebook Features

The notebook (`championship_dna_notebook.ipynb`) provides:
- Interactive visualizations
- Round-by-round heat maps
- Player frequency analysis
- Custom tier adjustments
- Live draft tracking integration

### To Use the Notebook:
1. Generate data first (see commands above)
2. Open Jupyter: `jupyter notebook`
3. Navigate to `notebooks/championship_dna_notebook.ipynb`
4. Run all cells or step through interactively

## Live Draft Integration

During your actual draft:
1. Use `backup_draft.py` to track picks
2. Run DNA analysis between picks:
```bash
# Quick recommendation for next pick
PYTHONPATH=. uv run python notebooks/championship_dna_hybrid.py live --pick 5
```

## Troubleshooting

### "No files found for strategy"
→ Run the export command first to generate data

### "Only X players found - tier analysis unreliable"  
→ Normal with 200 sims. Use 1000+ for better tier data

### "Missing round data"
→ Fixed! Re-export data after the Round 7 fix

## Advanced Usage

### Create Custom Analysis
```python
from notebooks.championship_dna_analyzer import ChampionshipDNA

analyzer = ChampionshipDNA()
champions = analyzer.load_champions(strategy='balanced', top_pct=0.1)

# Get North Star
north_star = analyzer.get_north_star(champions)

# Analyze specific round
round_3_windows = analyzer.calculate_windows(champions, round_num=3)

# Get tier rankings
rb_tiers = analyzer.create_tiers(champions, 'RB')
```

## Key Insights from Your Data

Based on the balanced strategy at pick #5:
- **Round 1**: Always RB (100%)
- **Round 2**: Continue RB (70%) or pivot to TE (30%)
- **Round 3**: TE/QB window opens
- **Rounds 7-9**: WR accumulation phase
- **Late rounds**: Depth and upside

The system discovered these patterns from YOUR league settings - not generic advice!