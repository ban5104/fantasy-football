# Fantasy Football Draft Analysis & Assistance System

A comprehensive fantasy football draft analysis system combining sophisticated VBD (Value-Based Drafting) calculations with real-time draft assistance and AI-powered recommendations.

## 🏈 What This Does

**Pre-Draft Analysis:**
- Scrapes current projections from FantasyPros for all positions
- Calculates VBD scores using multiple methods (VOLS, VORP, BEER, Dynamic VBD)
- Generates position-based rankings and tier analysis
- Creates customizable draft cheat sheets

**Live Draft Assistance:**
- Interactive 3-panel draft board with AI recommendations
- Real-time Dynamic VBD adjustments based on draft flow
- Enhanced emergency backup system with Dynamic VBD integration
- Snake draft visualization with position color-coding
- Multiple fallback options (full-featured and simplified trackers)

**Advanced Features:**
- **Dynamic VBD** with real-time baseline adjustments during live drafts
- **Position scarcity detection** and automatic tier break identification
- Multi-factor AI recommendations considering value, scarcity, and roster needs
- **Simplified implementation** for faster performance and easier maintenance
- Compatible with 8-16 team leagues (fully configurable)

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install dependencies (UV recommended)
uv sync

# Alternative: Direct pip install
uv pip install -r requirements_draft_board.txt

# Verify installation
export PATH="$(uv python find | head -n1 | xargs dirname):$PATH"
python3 -c "import pandas, numpy; print('✓ Ready to go!')"
```

### 2. Configure Your League
Edit `config/league-config.yaml`:
```yaml
basic_settings:
  teams: 14
  draft_type: "snake"

scoring:
  passing: {yards: 0.04, td: 4, int: -1}
  rushing: {yards: 0.1, td: 6}
  receiving: {yards: 0.1, td: 6, reception: 1}  # PPR scoring

roster:
  roster_slots: {QB: 1, RB: 2, WR: 2, TE: 1, FLEX: 1, K: 1, DST: 1}
```

### 3. Get Current Data
```bash
# Automated pipeline: scraping → scoring → VBD → rankings  
python scripts/daily_update.py

# Manual approach using Jupyter notebooks
jupyter notebook notebooks/02_analysis/03_vbd_calculations.ipynb
```

### 4. Draft Day
```bash
# Primary: Interactive draft board (Jupyter)
jupyter notebook notebooks/minimal_draft_board.ipynb

# Backup: Enhanced terminal tracker with Dynamic VBD
python backup_draft.py --dynamic-vbd

# Simplified version for faster startup
python backup_draft_simplified.py

# Reset for new draft
rm data/draft/draft_picks_latest.csv
```

## 📊 Core Features

### VBD Analysis Methods
- **VOLS** (Value Over Like Starters) - Conservative baseline approach
- **VORP** (Value Over Replacement Player) - Balanced replacement strategy  
- **BEER** (Best Eleven Every Round) - Aggressive draft approach
- **Dynamic VBD** - **Real-time baseline adjustments** responding to draft flow
- **Blended** - Optimized weighted combination (50% BEER + 25% VORP + 25% VOLS)
- **Simplified Dynamic VBD** - Streamlined implementation for core functionality

### Draft Tools
| Tool | Purpose | When to Use |
|------|---------|-------------|
| `minimal_draft_board.ipynb` | **PRIMARY** - 3-panel draft interface | Normal draft day |
| `backup_draft.py` | Enhanced emergency tracker with Dynamic VBD | ESPN API fails |
| `backup_draft_simplified.py` | Simplified tracker for faster startup | Quick backup needs |
| `interactive_draft_board.ipynb` | Advanced team roster tracking | Detailed analysis |
| `live_draft_tracker.py` | ESPN API integration | Real-time monitoring |

### Output Files
**Main Rankings:** `data/output/vbd_rankings_top300_YYYYMMDD.csv`  
**Individual Methods:** `rankings_vbd_[method]_top300_YYYYMMDD.csv`  
**Draft Sheet:** `draft_cheat_sheet.csv`  
**Live Draft:** `data/draft/draft_picks_latest.csv`

## 🎯 Usage Workflows

### Weekly Data Refresh
```bash
python scripts/daily_update.py  # Full automated pipeline

# Verify Dynamic VBD integration
python3 test_simple_integration.py
```

### Draft Preparation
1. Run data refresh to get current projections
2. Review VBD rankings: `data/output/vbd_rankings_top300_*.csv`
3. Customize league settings if needed
4. Generate draft cheat sheet

### Live Draft
1. **Primary**: Open `notebooks/minimal_draft_board.ipynb`
2. Set your team ID and draft position in Cell 1
3. Use 3-panel interface: controls, draft board, AI recommendations
4. **Backup**: Run `python backup_draft.py --dynamic-vbd` if main system fails
   - **Fast backup**: Use `python backup_draft_simplified.py` for quick startup

### Post-Draft Analysis
- Review team construction in `notebooks/post_draft_analysis.ipynb`
- Analyze draft efficiency and missed opportunities
- Generate trade and waiver wire recommendations

## 🛠️ System Architecture

```
Data Flow: FantasyPros → VBD Analysis → Dynamic Adjustments → Draft Tools

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │ ── │  Static VBD     │ ── │ Dynamic VBD     │ ── │   Draft Tools   │
│                 │    │                 │    │                 │    │                 │
│ • FantasyPros   │    │ • VOLS/VORP     │    │ • Real-time     │    │ • Draft Board   │
│ • Player DB     │    │ • BEER/Blended  │    │ • Scarcity      │    │ • AI Recomm.    │
│ • ESPN API      │    │ • Statistical   │    │ • Flow Analysis │    │ • Backup Track  │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Components
- **Data Pipeline**: `scripts/daily_update.py` → `src/` modules → `data/output/`
- **VBD Engine**: `src/vbd.py` + `src/dynamic_vbd.py` for real-time baseline adjustments
- **Draft Intelligence**: `src/draft_engine.py` with multi-factor recommendations + position scarcity detection
- **User Interfaces**: Jupyter notebooks + enhanced terminal backup with Dynamic VBD
- **Testing Suite**: Comprehensive tests for VBD calculations and draft integration

## 📁 Project Structure

```
├── config/league-config.yaml     # League settings and scoring
├── scripts/daily_update.py       # Automated data pipeline  
├── notebooks/                    # Interactive analysis tools
│   ├── minimal_draft_board.ipynb # PRIMARY draft interface
│   └── 02_analysis/              # VBD calculations
├── src/                          # Core Python modules
│   ├── vbd.py                   # Core VBD calculation engine (VOLS/VORP/BEER)
│   ├── dynamic_vbd.py           # Real-time baseline adjustments
│   ├── draft_engine.py          # AI recommendations with scarcity analysis
│   └── statistical_analysis.py  # Advanced modeling and predictions
├── data/
│   ├── output/                  # VBD rankings and analysis
│   └── draft/                   # Live draft tracking
└── backup_draft.py              # Emergency draft tracker
```

## ⚙️ Configuration

### League Customization
- **Teams**: 8-16 teams supported
- **Scoring**: PPR, Half-PPR, Standard, Custom
- **Roster**: Flexible positions (QB, RB, WR, TE, FLEX, K, DST)
- **VBD**: Adjustable replacement levels and blend weights

### Draft Settings
- **Snake draft**: Automatic order calculation
- **Dynamic VBD**: Real-time baseline adjustments
- **AI recommendations**: Multi-factor scoring system
- **Backup modes**: Multiple fallback options

## 🔧 Troubleshooting

### Environment Issues
```bash
# If UV sync fails
uv pip install -r requirements_draft_board.txt

# If imports fail
export PATH="$(uv python find | head -n1 | xargs dirname):$PATH"
PYTHONPATH=. python3 your_script.py

# Test core functionality
python3 -c "import sys; sys.path.insert(0, '.'); import src.vbd; print('✓ Core modules working')"

# Test Dynamic VBD integration
python3 test_simple_integration.py
PYTHONPATH=. python3 run_tests.py
```

### Common Problems
- **No data**: Run `python scripts/daily_update.py` first
- **Import errors**: Use `PYTHONPATH=.` prefix for Python commands
- **Draft fails**: Use `python backup_draft.py` as emergency backup
- **Old data**: Check timestamps on files in `data/output/`

## 📈 Advanced Features

### Dynamic VBD
Real-time baseline adjustments based on:
- **Draft flow analysis** - Position runs and scarcity detection
- **Draft stage awareness** - Early/middle/late draft behavior patterns  
- **Position availability** - Remaining quality players at each position
- **Mathematical optimization** - Sigmoid-based scaling for smooth transitions
- **Performance optimized** - Simplified calculations for live draft speed

### AI Recommendations
Multi-factor analysis considering:
- **Value**: VBD score relative to alternatives
- **Need**: Roster construction requirements
- **Scarcity**: Remaining quality at position
- **Timing**: Optimal draft round for position

### Statistical Analysis
- Predictive modeling for player performance
- Monte Carlo simulations for draft outcomes
- Bayesian inference for updated valuations
- Tier analysis for strategic timing

## 📊 Sample Output

**VBD Rankings Preview:**
```
Player              Pos  VBD_BLENDED  VBD_BEER  VBD_VORP  Rank
Christian McCaffrey RB   42.8         45.2      38.1      1
Cooper Kupp        WR   38.4         41.0      34.5      2
Jonathan Taylor    RB   36.7         39.2      32.8      3
```

**AI Recommendation:**
```
🎯 RECOMMENDED: Cooper Kupp (WR)
📊 VBD Score: 38.4 (Rank #2 overall)  
⚡ Scarcity: HIGH - Only 3 elite WRs remain
🏗️ Roster Fit: Addresses WR1 need perfectly
⏰ Timing: Optimal round for WR position
```

---

**Built for competitive fantasy football managers who want data-driven draft advantages with reliable backup systems.**

*For detailed technical documentation, see [CLAUDE.md](CLAUDE.md)*