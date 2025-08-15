# Emergency Backup Draft Tracker - Quick Guide

## 🚨 When ESPN API Fails

### Start the Tracker
```bash
python backup_draft.py
```

### Basic Usage

1. **Enter Player Names** - Type partial names:
   - `mccaf` → finds "Christian McCaffrey"
   - `lamb` → finds "CeeDee Lamb"
   - System shows position/team for confirmation

2. **Commands** (type at any pick prompt):
   - `UNDO` - Remove last pick if you make a mistake
   - `STATUS` - Show current draft status and recent picks
   - `QUIT` - Save and exit cleanly

### Auto Features
- **Snake Draft** - Automatically assigns correct team (1→14, then 14→1)
- **Auto-Save** - Saves after every pick (crash-proof)
- **Resume** - If script crashes, just restart - picks up where you left off

### Output Files
- `data/draft/draft_picks_latest.csv` - Main file (notebooks read this)
- `data/draft/draft_picks_TIMESTAMP.csv` - Backup with timestamp

### Example Session
```
📍 Pick #47 (Round 4, Team 8)
Enter player name (or command): lamb

📋 Found: CeeDee Lamb (WR, DAL)
Confirm? (y/n): y
✅ Pick #47: Team 8 selects CeeDee Lamb (WR, DAL)
💾 Saved 47 picks
```

### Emergency Tips
- **Can't find player?** Try last name only or first few letters
- **Wrong player selected?** Type `UNDO` immediately
- **System crashed?** Just restart - all picks are saved
- **Multiple matches?** System shows numbered list to choose from

### File Compatibility
Output matches ESPN tracker format exactly:
- `overall_pick` - Pick number (1-224)
- `player_name` - Full player name
- `position` - QB/RB/WR/TE/K/DST
- `team_name` - "Team 1" through "Team 14"
- `pro_team` - NFL team abbreviation

### Critical Notes
- Works offline - no internet needed after player database loaded
- Zero external dependencies beyond pandas
- Tested for 14-team, 16-round snake drafts
- All analysis notebooks work normally with backup data