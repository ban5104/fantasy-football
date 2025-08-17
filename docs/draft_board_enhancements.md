# Draft Board Enhancements

Optional features you can add to the minimal draft board if needed.

## 1. Keyboard Shortcuts (5 minutes)

```python
# Add to MinimalDraftManager.__init__
from IPython.display import Javascript

def setup_keyboard_shortcuts(self):
    js_code = """
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            // Ctrl+Enter to draft
            document.querySelector('button:contains("DRAFT")').click();
        } else if (e.key === 'a' && e.ctrlKey) {
            // Ctrl+A for auto pick
            document.querySelector('button:contains("AUTO")').click();
        }
    });
    """
    display(Javascript(js_code))
```

## 2. Pick Timer (10 minutes)

```python
import threading
import time

class PickTimer:
    def __init__(self, seconds=30):
        self.seconds = seconds
        self.timer_output = widgets.Output()
        self.remaining = seconds
        self.active = False
        
    def start(self):
        self.active = True
        self.remaining = self.seconds
        
        def countdown():
            while self.remaining > 0 and self.active:
                with self.timer_output:
                    clear_output(wait=True)
                    if self.remaining <= 10:
                        print(f"⏰ TIME: {self.remaining}s 🔴")
                    else:
                        print(f"⏰ TIME: {self.remaining}s")
                time.sleep(1)
                self.remaining -= 1
            
            if self.active and self.remaining == 0:
                with self.timer_output:
                    clear_output()
                    print("⏰ TIME'S UP! AUTO-PICKING...")
                # Trigger auto-pick
        
        threading.Thread(target=countdown).start()
    
    def stop(self):
        self.active = False
```

## 3. Position Run Detection (5 minutes)

```python
def detect_position_runs(draft_state, window=5):
    """Detect if a position run is happening"""
    if len(draft_state.picks) < window:
        return None
    
    recent_picks = draft_state.picks[-window:]
    position_counts = {}
    
    for pick in recent_picks:
        pos = pick.player.position
        position_counts[pos] = position_counts.get(pos, 0) + 1
    
    # Check for runs (3+ of same position in last 5 picks)
    for pos, count in position_counts.items():
        if count >= 3:
            return f"🏃 {pos} RUN DETECTED! ({count} in last {window} picks)"
    
    return None
```

## 4. Trade Value Chart (10 minutes)

```python
def calculate_trade_value(player):
    """Calculate trade value based on VBD and ADP"""
    # Simple trade value formula
    base_value = player.vbd * 10
    
    # Adjust for position scarcity
    position_multipliers = {
        'RB': 1.2,  # RBs are scarce
        'WR': 1.0,
        'QB': 0.9,  # QBs are deep
        'TE': 1.1,  # Top TEs are valuable
    }
    
    multiplier = position_multipliers.get(player.position, 1.0)
    
    # Adjust for tier
    tier_bonus = max(0, (6 - player.tier) * 20)
    
    return int(base_value * multiplier + tier_bonus)

def show_trade_analyzer(team1_players, team2_players):
    """Compare trade values"""
    team1_value = sum(calculate_trade_value(p) for p in team1_players)
    team2_value = sum(calculate_trade_value(p) for p in team2_players)
    
    diff = abs(team1_value - team2_value)
    fair = "✅ FAIR" if diff < 50 else "⚠️ UNEVEN"
    
    return f"Team 1: {team1_value} | Team 2: {team2_value} | {fair}"
```

## 5. Bye Week Analysis (5 minutes)

```python
def analyze_bye_weeks(roster):
    """Check for bye week conflicts"""
    bye_weeks = {}
    
    for pos, players in roster.items():
        if pos == 'BENCH':
            continue
        for player in players:
            if hasattr(player, 'bye_week') and player.bye_week:
                week = player.bye_week
                if week not in bye_weeks:
                    bye_weeks[week] = []
                bye_weeks[week].append(f"{player.name} ({player.position})")
    
    # Find problematic weeks (3+ starters on bye)
    problems = []
    for week, players in bye_weeks.items():
        if len(players) >= 3:
            problems.append(f"Week {week}: {', '.join(players)}")
    
    return problems
```

## 6. Export Draft Results (5 minutes)

```python
def export_draft_results(draft_state, filename="draft_results.csv"):
    \"\"\"Export draft results to CSV\"\"\"\n",
    results = []
    
    for pick in draft_state.picks:
        results.append({
            'Pick': pick.pick_number,
            'Round': pick.round,
            'Team': draft_state.teams[pick.team_id].name,
            'Player': pick.player.name,
            'Position': pick.player.position,
            'VBD': pick.player.vbd,
            'Tier': pick.player.tier\n",
        })
    
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"✅ Draft exported to {filename}")
```

## 7. Mock Draft Mode (10 minutes)

```python
class MockDraftAI:
    \"\"\"Simple AI for mock drafts\"\"\"\n",
    
    def make_pick_for_team(self, team_id, draft_state, intelligence):\n",
        \"\"\"Make an intelligent pick for a team\"\"\"\n",
        team = draft_state.teams[team_id]\n",
        \n",
        # Get team needs\n",
        needs = []\n",
        for pos in ['QB', 'RB', 'WR', 'TE']:\n",
            if team.needs_position(pos):\n",
                needs.append(pos)\n",
        \n",
        # Get best available with position weight\n",
        candidates = intelligence.get_recommendations(draft_state, 10)\n",
        \n",
        # Pick based on value and need\n",
        for player, score, _ in candidates:\n",
            if player.position in needs:\n",
                score *= 1.5  # Boost needed positions\n",
            \n",
            # Add some randomness for realism\n",
            if np.random.random() > 0.3:  # 70% chance to take best\n",
                return player\n",
        \n",
        # Default to best available\n",
        return candidates[0][0] if candidates else None
```

## 8. Advanced Visualizations (15 minutes)

```python
import plotly.graph_objects as go

def create_value_chart(draft_state, intelligence):\n",
    \"\"\"Interactive value chart with Plotly\"\"\"\n",
    \n",
    fig = go.Figure()\n",
    \n",
    # Add traces for each position\n",
    for pos in ['QB', 'RB', 'WR', 'TE']:\n",
        drafted = []\n",
        available = []\n",
        \n",
        for pick in draft_state.picks:\n",
            if pick.player.position == pos:\n",
                drafted.append({\n",
                    'pick': pick.pick_number,\n",
                    'vbd': pick.player.vbd,\n",
                    'name': pick.player.name\n",
                })\n",
        \n",
        # Plot drafted players\n",
        if drafted:\n",
            fig.add_trace(go.Scatter(\n",
                x=[d['pick'] for d in drafted],\n",
                y=[d['vbd'] for d in drafted],\n",
                mode='markers',\n",
                name=f'{pos} (Drafted)',\n",
                text=[d['name'] for d in drafted],\n",
                marker=dict(size=10)\n",
            ))\n",
    \n",
    fig.update_layout(\n",
        title='Value Over Replacement by Pick',\n",
        xaxis_title='Pick Number',\n",
        yaxis_title='VBD Score',\n",
        hovermode='closest'\n",
    )\n",
    \n",
    return fig
```

## 9. Stack Ranking Display (5 minutes)

```python
def show_stack_rankings(draft_state):\n",
    \"\"\"Show how each team ranks by total VBD\"\"\"\n",
    team_values = []\n",
    \n",
    for team_id, team in draft_state.teams.items():\n",
        total_vbd = sum(p.vbd for p in team.players)\n",
        avg_vbd = total_vbd / len(team.players) if team.players else 0\n",
        \n",
        team_values.append({\n",
            'Team': team.name,\n",
            'Total VBD': total_vbd,\n",
            'Avg VBD': avg_vbd,\n",
            'Picks': len(team.players)\n",
        })\n",
    \n",
    df = pd.DataFrame(team_values)\n",
    df = df.sort_values('Total VBD', ascending=False)\n",
    \n",
    # Highlight user's team\n",
    def highlight_user(row):\n",
        if row['Team'] == draft_state.teams[draft_state.user_team_id].name:\n",
            return ['background-color: #27AE60'] * len(row)\n",
        return [''] * len(row)\n",
    \n",
    return df.style.apply(highlight_user, axis=1)\n",
```

## 10. Save/Load Draft State (5 minutes)

```python
import pickle

def save_draft(draft_state, filename='draft_save.pkl'):\n",
    \"\"\"Save draft to file\"\"\"\n",
    with open(filename, 'wb') as f:\n",
        pickle.dump({\n",
            'picks': draft_state.picks,\n",
            'current_pick': draft_state.current_pick,\n",
            'teams': draft_state.teams\n",
        }, f)\n",
    print(f\"💾 Draft saved to {filename}\")\n",

def load_draft(draft_state, filename='draft_save.pkl'):\n",
    \"\"\"Load draft from file\"\"\"\n",
    try:\n",
        with open(filename, 'rb') as f:\n",
            data = pickle.load(f)\n",
            draft_state.picks = data['picks']\n",
            draft_state.current_pick = data['current_pick']\n",
            draft_state.teams = data['teams']\n",
        print(f\"📂 Draft loaded from {filename}\")\n",
        return True\n",
    except FileNotFoundError:\n",
        print(f\"❌ File {filename} not found\")\n",
        return False\n",
```

## Usage Tips

### Adding Features:\n",
1. Copy the code snippet\n",
2. Add to appropriate cell in notebook\n",
3. Call from main manager class\n",

### Performance:\n",
- Keep visualizations simple during draft\n",
- Save complex analysis for post-draft\n",
- Cache calculations when possible\n",

### Testing:\n",
- Use mock draft mode to test features\n",
- Save state before experimenting\n",
- Keep backup of working version\n",

Remember: **Only add what you'll actually use on draft day!**