# Fantasy Football VBD Ranking Dashboard

A clean, interactive, sortable table for fantasy football draft decisions using VBD (Value Based Drafting) rankings with dynamic probability intelligence.

## Key Features

### ✅ **Dynamic Draft Position Logic**
- No hard-coded draft picks
- Calculates current pick, next pick, and picks remaining automatically
- Updates probabilities in real-time as draft progresses
- Auto-advances pick when players are marked as drafted

### ✅ **Interactive Sortable Table**
- Sort by any column (VBD Rank, VBD Score, Probabilities, etc.)
- Clean, minimal design with all data columns visible
- Dynamic column headers that update based on current draft position
- 50+ players displayed with key metrics

### ✅ **VBD-Focused Rankings**
- Pure VBD rankings (not probability-weighted)
- Shows Custom VBD values as primary scoring metric
- Probability intelligence for strategic decision support
- Decision notes with actionable guidance

## Quick Start

```python
# Initialize dashboard
dashboard = DraftDashboard(df)

# Show table (default: sorted by VBD rank)
dashboard.show_table()

# Sort by different columns
dashboard.show_table(sort_by='salary_value', ascending=False)  # Highest VBD first
dashboard.show_table(sort_by='prob_at_next_pick', ascending=False)  # Best availability
dashboard.show_table(sort_by='decision_score', ascending=False)  # Highest decision score

# Manage draft progress
dashboard.mark_drafted("Saquon Barkley")  # Auto-advances pick
dashboard.set_current_pick(15)  # Jump to specific pick
dashboard.advance_pick(3)  # Skip ahead

# Set your draft positions
dashboard.set_my_picks([8, 17, 32, 41, 56, 65, 80, 89])  # 8-team league, pick 8
```

## Table Columns

| Column | Description |
|--------|-------------|
| **VBD Rank** | Your custom VBD ranking (1, 2, 3...) |
| **Player** | Player name |
| **Position** | Position (QB, RB, WR, TE) |
| **Team** | NFL team |
| **VBD Score** | Your custom VBD value |
| **P(Pick X)%** | Probability available at your next pick (dynamic) |
| **P(Pick Y)%** | Probability available at your pick after next (dynamic) |
| **Decision Score** | VBD Score × Probability at next pick |
| **Opportunity Cost** | Value lost by waiting vs. drafting now |
| **Median Pick** | Expected draft position |
| **Pick Range** | 10th-90th percentile range |
| **Bye Week** | Bye week number |
| **Decision Notes** | Strategic guidance (ELITE - DRAFT NOW, etc.) |

## Sortable Columns

Sort by any of these columns using `dashboard.show_table(sort_by='column_name')`:

- `'overall_rank'` - VBD Rank
- `'salary_value'` - VBD Score  
- `'prob_at_next_pick'` - Probability at Next Pick
- `'prob_at_pick_after'` - Probability at Pick After
- `'decision_score'` - Decision Score
- `'opportunity_cost'` - Opportunity Cost
- `'median_pick'` - Median Pick
- `'bye_week'` - Bye Week
- `'player_name'` - Player Name

## Dynamic Draft Management

### Current Draft Status
```python
status = dashboard.get_current_status()
print(f"Current pick: {status['current_pick']}")
print(f"Your next pick: {status['next_pick']} (in {status['picks_to_next']} picks)")
```

### Draft Progression
```python
# Mark players as drafted (auto-advances pick)
dashboard.mark_drafted("Bijan Robinson")
dashboard.mark_drafted("Ja'Marr Chase")

# Manual pick management
dashboard.set_current_pick(25)  # Jump to pick 25
dashboard.advance_pick(5)       # Skip 5 picks ahead

# Undo drafted players
dashboard.unmark_drafted("Bijan Robinson")
```

### Real-Time Updates
- Table title shows: "Current Pick: 15 | Your Next Pick: 17 (in 2 picks)"
- Column headers update: "P(Pick 8)%" becomes "P(Pick 17)%" dynamically
- Probabilities recalculate based on current draft position

## Data Export

```python
# Get all enhanced data as DataFrame
data = dashboard.get_data_export()

# Get list of available players
players = dashboard.get_player_list()
```

## Example Workflow

```python
# 1. Set up your draft
dashboard.set_my_picks([8, 17, 32, 41, 56, 65, 80, 89])

# 2. View top VBD players
dashboard.show_table()

# 3. Sort by probability at your next pick
dashboard.show_table(sort_by='prob_at_next_pick', ascending=False)

# 4. Mark first overall pick
dashboard.mark_drafted("First Overall Pick")  # Advances to pick 2

# 5. Continue tracking draft
dashboard.mark_drafted("Second Pick")  # Advances to pick 3

# 6. Check decision scores as your pick approaches
dashboard.show_table(sort_by='decision_score', ascending=False)

# 7. Make your pick and continue
dashboard.mark_drafted("Your Pick")  # Auto-advances
```

## Technical Notes

- Uses VBD rankings from `draft_cheat_sheet.csv`
- Probability calculations use normal distribution (μ=player_rank, σ=3)
- Decision Score = VBD Score × Probability at next pick
- Auto-advancing picks streamlines draft tracking
- All data updates in real-time based on current draft state