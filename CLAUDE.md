# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this fantasy football draft analysis and assistance system.

## System Overview

This is a production-ready fantasy football draft analysis system implementing **Enhanced Probabilistic VBD** - a statistical framework combining traditional value-based drafting with real-time selection probabilities and roster-aware utility calculations.

### Core Statistical Components
- **Multi-Method VBD Engine**: VOLS/VORP/BEER with configurable replacement levels
- **Probabilistic Selection Theory**: Dynamic replacement levels based on draft flow analysis
- **Roster Construction Optimization**: Bayesian positional need calculations  
- **Real-Time Utility Scoring**: `Utility = P(available) × (VBD - R_dynamic) × (1 + roster_need)`
- **Draft Intelligence System**: Multi-factor recommendations with scarcity detection

### Implementation Roadmap: Enhanced Probabilistic VBD

**Phase 1: Current State (Operational)**
- Static VBD calculations with multiple methods
- Dynamic baseline adjustments using sigmoid scaling
- Real-time draft flow analysis and position scarcity detection

**Phase 2: Probabilistic Enhancement (In Development)**  
- Integration of ESPN selection probability data
- Dynamic replacement level calculation: `R_dynamic = best_player_with_survival_prob < 0.3`
- Positional Need Index (PNI) for roster construction optimization
- Unified utility scoring replacing isolated VBD metrics

**Phase 3: Statistical Validation (Planned)**
- Bayesian inference for probability calibration  
- Monte Carlo simulation for draft outcome modeling
- Performance benchmarking against historical draft data
- A/B testing framework for methodology comparison

## Development Commands

### Environment Setup
```bash
# Install dependencies (UV recommended)
uv sync
# Fallback: uv pip install -r requirements_draft_board.txt

# Verify setup
export PATH="$(uv python find | head -n1 | xargs dirname):$PATH"
python3 -c "import pandas, numpy; print('✓ Dependencies ready')"
```

### Core Workflows

#### Data Pipeline (Automated)
```bash
# Full pipeline: scraping → scoring → VBD → rankings
python scripts/daily_update.py

# Individual steps
python scripts/scrape_projections.py          # Data collection
jupyter notebook notebooks/02_analysis/03_vbd_calculations.ipynb  # VBD analysis
```

#### Draft Day Operations
```bash
# Primary: Interactive draft board
jupyter notebook notebooks/minimal_draft_board.ipynb

# Backup: Emergency terminal tracker
python backup_draft.py

# Advanced: Real-time ESPN integration
python live_draft_tracker.py
```

#### Testing & Validation
```bash
# Run all tests
export PATH="$(uv python find | head -n1 | xargs dirname):$PATH"
PYTHONPATH=. python3 run_tests.py

# Test specific components
PYTHONPATH=. python3 -m pytest tests/test_dynamic_vbd.py -v
PYTHONPATH=. python3 -m pytest tests/test_backup_draft.py -v

# Integration testing
python3 test_backup_draft_integration.py
python3 test_simple_integration.py
```

## Architecture Overview

### Data Flow Pipeline
```
FantasyPros → Raw Projections → Fantasy Points → VBD Calculations → Rankings
     ↓              ↓               ↓                ↓                    ↓
scraping.py → data/raw/ → scoring.py → vbd.py → data/output/
                                       ↓
                               dynamic_vbd.py (real-time adjustments)
                                       ↓
                           Draft Tools (notebooks/backup_draft.py)
```

### Core Modules

#### Data Processing Pipeline
- **`src/scraping.py`** - Web scraping FantasyPros for all positions (QB/RB/WR/TE/K/DST)
- **`src/scoring.py`** - Converts raw stats to fantasy points using configurable league scoring
- **`src/vbd.py`** - Core VBD calculations (VOLS, VORP, BEER, Blended methods)
- **`src/dynamic_vbd.py`** - Real-time VBD adjustments based on draft state
- **`src/statistical_analysis.py`** - Advanced statistical modeling and analysis
- **`src/utils.py`** - Data validation, logging, file I/O utilities

#### Draft Assistance System
- **`src/draft_engine.py`** - AI recommendation engine with multi-factor analysis
- **`src/data_processor.py`** - Flexible CSV handling for various data sources
- **`backup_draft.py`** - Enhanced terminal-based emergency draft tracker with Dynamic VBD integration
- **`backup_draft_simplified.py`** - Simplified version focused on core Dynamic VBD functionality
- **`live_draft_tracker.py`** - ESPN API integration for real-time draft monitoring
- **`draft_board_app.py`** - Streamlit web interface for draft boards

#### Interactive Notebooks
- **`notebooks/minimal_draft_board.ipynb`** - **PRIMARY**: 3-panel draft interface
- **`notebooks/draft_preparation.ipynb`** - **CORE**: Comprehensive pre-draft analysis and strategic preparation tools
- **`notebooks/interactive_draft_board.ipynb`** - Advanced draft tracking with team rosters
- **`notebooks/auto_draft_board.ipynb`** - Automated draft board generation

### Configuration System

#### Central Configuration
- **`config/league-config.yaml`** - Master configuration driving all calculations
  - League settings (14 teams, scoring rules, roster requirements)
  - VBD weights and calculation parameters
  - Dynamic VBD settings and thresholds
  - Draft stage configurations

#### Dynamic VBD Configuration
```yaml
dynamic_vbd:
  enabled: true
  params:
    scale: 3.0              # Max baseline adjustment magnitude
    kappa: 5.0              # Sigmoid steepness for adjustments
  draft_stages:
    early_threshold: 0.3    # First 30% of picks
    late_threshold: 0.7     # Last 30% of picks
```

### VBD Methods Implementation

#### Traditional Methods (Phase 1)
- **VOLS** (Value Over Like Starters): `baseline = teams × starters` 
- **VORP** (Value Over Replacement): `baseline = teams × (starters + 1)`
- **BEER** (Best Eleven Every Round): `baseline = teams × (starters + 0.5)`
- **Blended**: Weighted combination (50% BEER + 25% VORP + 25% VOLS)

#### Dynamic VBD Enhancement (Phase 1)
- **Real-time baseline adjustments**: `adjustment = scale × tanh(expected_picks / kappa)`
- **Position scarcity detection** using draft flow analysis
- **Sigmoid-based scaling** for smooth value transitions
- **Draft stage awareness** (early/middle/late draft behaviors)

#### Probabilistic VBD Enhancement (Phase 2)
- **Dynamic replacement calculation**:
  ```python
  def calculate_dynamic_replacement(position, selection_probs, horizon=20):
      available = get_available_at_position(position)
      likely_survivors = [p for p in available if selection_probs[p] < 0.3]
      return likely_survivors[0].fantasy_points if likely_survivors else baseline
  ```
- **Positional Need Index**:
  ```python
  def calculate_PNI(position, my_roster, selection_probs):
      slots_needed = get_remaining_slots(position, my_roster)
      expected_supply = sum([1 - p for p in selection_probs if p < 0.5])
      shortfall = max(0, slots_needed - expected_supply) 
      return shortfall * position_scarcity_cost(position)
  ```
- **Unified utility scoring**:
  ```python
  utility = selection_prob × (VBD - dynamic_replacement) × (1 + beta × PNI)
  ```

### Data Architecture

#### Input Data Sources
- **`data/raw/projections_*_YYYYMMDD.csv`** - FantasyPros scraped projections by position
- **`data/CSG Fantasy Football Sheet - 2025 v13.01.csv`** - Master player database with ADP
- **External APIs**: ESPN (when available) for live draft data

#### Processing Stages
- **`data/processed/`** - Intermediate calculations and transformations
- **`data/output/`** - Final rankings and analysis outputs
- **`data/draft/`** - Live draft state and pick history

#### Output Files
**Primary Rankings:**
- **`vbd_rankings_top300_YYYYMMDD.csv`** - Main blended VBD rankings
- **`rankings_vbd_*_top300_YYYYMMDD.csv`** - Individual method rankings
- **`rankings_statistical_vbd_top300_YYYYMMDD.csv`** - Advanced statistical VBD

**Draft Tools:**
- **`draft_cheat_sheet.csv`** - Formatted draft preparation sheet
- **`draft_picks_latest.csv`** - Current draft state (ESPN-compatible format)

## Key Development Patterns

### Configuration Loading
```python
from src.scoring import load_league_config
config = load_league_config()  # Loads config/league-config.yaml
```

### VBD Calculation Flow
```python
# Standard VBD calculation
df = calculate_fantasy_points_vectorized(df, config)
df_vbd = calculate_all_vbd_methods(df, config)

# Dynamic VBD with live adjustments
from src.dynamic_vbd import DynamicVBDTransformer
transformer = DynamicVBDTransformer(config)
baseline_overrides = transformer.calculate_draft_based_overrides(df, draft_probabilities)
df_vbd = calculate_all_vbd_methods(df, config, baseline_overrides)
```

### Draft State Management
```python
from src.draft_engine import DraftEngine
from backup_draft import BackupDraftTracker

# AI-powered draft recommendations
engine = DraftEngine(config)
recommendations = engine.get_recommendations(available_players, team_needs)

# Emergency backup tracking (full-featured)
tracker = BackupDraftTracker(force_dynamic_vbd=True)
tracker.run_interactive()

# Simplified version for faster startup
# python backup_draft_simplified.py --dynamic-vbd
```

### Error Handling & Validation
- **Comprehensive try/catch** with structured logging across all modules
- **Data quality validation** before expensive VBD calculations
- **Graceful degradation** when external APIs fail
- **Resume capability** for interrupted draft sessions

## Development Workflows

### Adding New VBD Methods
1. Implement calculation in `src/vbd.py`
2. Add configuration options to `config/league-config.yaml`
3. Update blend weights calculation
4. Add validation in `src/utils.py`
5. Create test cases in `tests/`

### Extending Draft Intelligence
1. Enhance recommendation logic in `src/draft_engine.py`
2. Add new factors to multi-criteria decision matrix
3. Update UI components in relevant notebooks
4. Test with historical draft data

### Data Source Integration
1. Add scraping logic to `src/scraping.py`
2. Implement data transformation in `src/data_processor.py`
3. Update configuration schema if needed
4. Add error handling and fallback mechanisms

### Notebook Development
1. Start with prototype in `notebooks/` appropriate directory
2. Extract reusable functions to `src/` modules
3. Add clear documentation and error handling
4. Archive experimental notebooks to `99_archive/`

## Testing Strategy

### Core Functionality Tests
```bash
# VBD calculations
PYTHONPATH=. python3 -m pytest tests/test_dynamic_vbd.py

# Draft tracking
PYTHONPATH=. python3 tests/test_backup_draft.py

# Integration tests
python3 test_backup_draft_integration.py
```

### Manual Validation
```bash
# Test full pipeline
python scripts/daily_update.py

# Verify data quality
python3 -c "
import sys; sys.path.insert(0, '.')
from src.utils import validate_data_quality
import pandas as pd
df = pd.read_csv('data/output/vbd_rankings_top300_*.csv')
print(validate_data_quality(df))
"
```

### Environment Issues Resolution
- **ModuleNotFoundError**: Ensure `export PATH="$(uv python find | head -n1 | xargs dirname):$PATH"`
- **Import failures**: Use `PYTHONPATH=.` prefix for all Python commands
- **Package conflicts**: Re-run `uv sync` or fallback to `uv pip install -r requirements_draft_board.txt`

## Performance Considerations

### Optimization Patterns
- **Vectorized pandas operations** for all statistical calculations
- **Position-based parallel processing** for VBD calculations
- **Caching mechanisms** in Dynamic VBD to avoid recalculation
- **Top 300 focus** to limit memory usage for draft-relevant players

### Memory Management
- **Lazy loading** of large datasets
- **Incremental processing** for live draft updates
- **Garbage collection** after expensive operations

### Scalability
- **Configurable league sizes** through YAML configuration
- **Modular architecture** allowing selective feature enabling
- **API rate limiting** with respectful delays (2-second intervals)

## Security & Data Handling

### Data Privacy
- **No personal information** stored in player databases
- **Local file storage** only - no external data transmission
- **Anonymized draft tracking** using team numbers

### API Usage
- **Rate-limited requests** to FantasyPros (2-second delays)
- **Graceful failure handling** when external services are unavailable
- **No API keys required** for core functionality

## Advanced Features

### Dynamic VBD Implementation
- **Real-time draft flow analysis** using smoothed probability distributions
- **Position scarcity modeling** with mathematical rigor
- **Adaptive baseline calculation** based on draft stage and team behaviors
- **Cache optimization** for live draft performance

### AI Draft Recommendations
- **Multi-factor scoring** considering value, need, scarcity, and timing
- **Tier break detection** for strategic positional runs
- **Roster construction optimization** based on remaining draft capital
- **Historical pattern analysis** for draft flow prediction

### Statistical Analysis
- **Advanced modeling** in `src/statistical_analysis.py`
- **Predictive analytics** for player performance
- **Monte Carlo simulations** for draft outcome modeling
- **Bayesian inference** for updated player valuations

## Troubleshooting Common Issues

### Environment Setup
- **UV sync failures**: Use `uv pip install -r requirements_draft_board.txt`
- **Python path issues**: Always use `PYTHONPATH=.` prefix
- **Module import errors**: Verify UV environment activation

### Data Issues
- **Scraping failures**: Check FantasyPros site availability, inspect network logs
- **VBD calculation errors**: Validate input data quality, check for missing columns
- **Draft tracking issues**: Verify CSV format compatibility, check file permissions

### Performance Problems
- **Slow calculations**: Profile code, consider reducing player scope
- **Memory issues**: Monitor DataFrame sizes, implement lazy loading
- **UI responsiveness**: Optimize real-time update frequency

## League Customization

### Scoring Configuration
- Modify `config/league-config.yaml` scoring section
- All major scoring systems supported (PPR, Half-PPR, Standard)
- Complex scoring rules (defensive TDs, return yards) handled

### Roster Configuration
- Flexible position requirements (supports FLEX, SUPERFLEX)
- Configurable bench sizes and roster maximums
- Custom position eligibility rules

### VBD Customization
- Adjustable replacement level calculations
- Configurable blend weights for different draft strategies
- Dynamic VBD parameters tunable for league tendencies

## Integration Points

### External Systems
- **ESPN API**: Live draft monitoring (when available)
- **FantasyPros**: Primary data source for projections
- **CSV Import/Export**: Compatible with popular fantasy platforms

### Extensibility
- **Plugin architecture** for new data sources
- **Configurable UI components** in Jupyter notebooks
- **Modular calculation engine** allowing custom VBD methods

This system represents a production-ready fantasy football draft assistance platform with both analytical depth and practical usability for live draft scenarios.