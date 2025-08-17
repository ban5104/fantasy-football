# Fantasy Football Draft Interface - Quick Start Guide

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ and npm
- Python 3.11+
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Option 1: Development Mode (Recommended for Draft Day)

#### 1. Start the Backend API
```bash
cd api
pip install fastapi uvicorn pandas numpy
python main.py
```
The API will start at http://localhost:8000

#### 2. Start the Frontend
```bash
cd frontend
npm install
npm run dev
```
The app will open at http://localhost:5173

### Option 2: Docker Deployment
```bash
docker-compose up --build
```
Access the app at http://localhost:8000

### Option 3: Production Build
```bash
# Build frontend
cd frontend
npm install
npm run build

# Serve with API
cd ..
python api/main.py
```

## 📊 Using the Interface

### Initial Setup
1. **Configure Your Draft Position**: The app defaults to pick #8 in an 8-team league
2. **Load Your Data**: Ensure these files are in place:
   - `data/espn_projections_20250814.csv` - ESPN rankings
   - `data/fantasypros_adp_20250815.csv` - ADP data
   - `draft_cheat_sheet.csv` - VBD scores

### During the Draft

#### Main Interface Components
- **Left Panel**: Shows current pick, next pick, and recommendations
- **Center Table**: Interactive player list with sorting and filtering
- **Right Panel**: Analytics, scatter plots, and position scarcity

#### Key Features
1. **Sort Columns**: Click any column header to sort
2. **Filter by Position**: Use position buttons above the table
3. **Draft a Player**: Click "Draft" button or use keyboard shortcut
4. **Expand Details**: Click player name for more info
5. **View Probabilities**: Check availability at your next picks

#### Decision Guide
- **Green (>70%)**: Safe to wait until next pick
- **Yellow (30-70%)**: Monitor closely, moderate risk
- **Red (<30%)**: Draft now or likely gone

### Keyboard Shortcuts
- `D` - Draft selected player
- `U` - Undo last draft
- `Arrow Keys` - Navigate table
- `1-5` - Filter by position (1=ALL, 2=QB, 3=RB, 4=WR, 5=TE)
- `Space` - Expand/collapse player details

## 🔧 Customization

### Adjust Draft Positions
Edit `api/main.py`:
```python
self.my_picks = [8, 17, 32, 41, 56, 65, 80, 89]  # Your pick numbers
```

### Change Probability Weights
Edit `api/main.py`:
```python
compute_pick_probabilities(df, espn_weight=0.8, adp_weight=0.2)
```

### Modify Visual Thresholds
Edit component files in `frontend/src/components/` to adjust:
- Color thresholds for availability
- Decision score calculations
- VBD bar scaling

## 📈 Data Updates

### Update ESPN Rankings
```bash
python scripts/extract_espn_projections.py
```

### Update ADP Data
```bash
python scripts/scrape_fantasypros_adp.py
```

### Update VBD Scores
Replace `draft_cheat_sheet.csv` with your updated values

## 🎯 Draft Day Checklist

- [ ] Update all data sources (ESPN, ADP, VBD)
- [ ] Test the interface with mock picks
- [ ] Set your correct draft positions
- [ ] Have backup laptop/device ready
- [ ] Test internet connection stability
- [ ] Keep paper backup of top 50 players

## 🐛 Troubleshooting

### API Not Connecting
- Check if port 8000 is available
- Verify Python dependencies installed
- Check data files exist in correct locations

### Frontend Not Loading
- Clear browser cache
- Check console for errors (F12)
- Verify npm packages installed correctly

### Real-time Updates Not Working
- Check WebSocket connection in browser console
- Ensure no firewall blocking WebSocket
- Try refreshing the page

## 📱 Mobile/Tablet Usage

The interface is responsive and works on tablets:
- Landscape mode recommended for tablets
- Touch to select players
- Swipe to scroll table
- Pinch to zoom charts

## 💾 Data Backup

The app automatically saves:
- Drafted players list
- Current pick position
- Your selections

Export draft results:
1. Open browser console (F12)
2. Type: `JSON.stringify(localStorage.draftState)`
3. Copy and save the output

## 🚨 Emergency Fallback

If the app fails during draft:
1. Note current pick number
2. List of drafted players (visible in left panel)
3. Restart the app
4. Use `/api/set-pick` endpoint to restore position
5. Mark drafted players using the interface

## 📞 Support

For issues during draft day:
1. Check browser console for errors
2. Try refreshing the page
3. Restart backend if needed
4. Use paper backup as last resort

---

**Remember**: The interface is a tool to assist decisions, not replace judgment. Trust your research and instincts!