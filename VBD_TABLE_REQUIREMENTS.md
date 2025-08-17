# VBD Ranking Table with Probability Intelligence

## Objective
Create a fantasy football draft table that shows players ranked by pure VBD value with probability data as decision support - NOT probability-weighted rankings.

## Current Problem
Table currently sorts by "Decision Score" (VBD × Probability) which buries the best players. User wants pure VBD rankings with probability as additional context.

## Required Table Structure

| VBD Rank | Player | VBD Score | P(at pick 8) | P(at pick 17) | Scarcity | Decision Notes |
|----------|---------|-----------|--------------|---------------|----------|----------------|
| 1 | Saquon | 131.3 | 5% | 0% | RB1 | Take if miracle |
| 2 | Bijan | 127.3 | 10% | 0% | RB2 | Take if available |
| 3 | Jahmyr | 125.0 | 15% | 2% | RB3 | Consider now? |
| 14 | CeeDee | 74.4 | 95% | 60% | WR8 | Safe to wait |

## Implementation Requirements

1. **Primary Sort**: VBD ranking (1, 2, 3, 4...) from `draft_cheat_sheet.csv`
2. **Remove Decision Score**: No probability weighting in sort order
3. **Add Multiple Pick Probabilities**: Show P(available) for user's next 2-3 picks
4. **Keep Existing Features**: Sparklines, scarcity badges, visual styling
5. **Strategic Context**: Help user decide "take now vs wait for next round"

## Data Sources
- `draft_cheat_sheet.csv`: Draft_Rank, Custom_VBD (primary ranking)
- User's pick positions: [8, 17, 32, 41, 56, 65, 80, 89]

## Files to Modify
- **`espn_probability_matrix.ipynb`**: Main notebook containing the enhanced decision table
- Specifically the `create_enhanced_decision_table()` function and sorting logic

## Expected Behavior
- **Saquon Barkley** appears at top (VBD Rank #1) regardless of low probability
- **CeeDee Lamb** appears at rank #14 with high probability data
- User can see: "My #3 VBD has 15% chance now, 2% next round → take now"

## Key Change
**Current**: Sort by `Custom_VBD × P(at my pick)` (Decision Score)  
**Required**: Sort by `Draft_Rank` (pure VBD order)