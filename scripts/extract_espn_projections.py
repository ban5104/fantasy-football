#!/usr/bin/env python3
"""
Extract ESPN Fantasy Football projections from PDF to CSV format.

This script parses the ESPN projections PDF and converts it to a clean CSV
format suitable for fantasy football analysis.
"""

import re
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any

import pdfplumber


def extract_player_data(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract player data from ESPN projections PDF."""
    players = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
                
            # Split into lines and process each line
            lines = text.split('\n')
            for line in lines:
                # Skip header lines and empty lines
                if not line.strip() or 'RANKINGS' in line or 'ESPN' in line or 'Position' in line:
                    continue
                    
                # Each line contains multiple players separated by spaces
                # Pattern: rank. (position) player_name, team $value bye_week
                pattern = r'(\d+)\.\s+\(([A-Z]+\d*)\)\s+([^,]+?),\s+([A-Z]{2,3}(?:/[A-Z]{3})?)\s+\$(\d+)\s+(\d+)'
                
                # Find all matches in the line (multiple players per line)
                matches = re.findall(pattern, line)
                
                for match in matches:
                    overall_rank, position_rank, player_name, team, salary, bye_week = match
                    
                    # Clean up team name (handle cases like "FA" or multi-team)
                    team = team.split('/')[0] if '/' in team else team
                    
                    # Extract position from position_rank (e.g., "WR1" -> "WR")
                    position_match = re.match(r'([A-Z]+)', position_rank)
                    if position_match:
                        position = position_match.group(1)
                        
                        players.append({
                            'overall_rank': int(overall_rank),
                            'position': position,
                            'position_rank': position_rank,
                            'player_name': player_name.strip(),
                            'team': team,
                            'salary_value': int(salary),
                            'bye_week': int(bye_week)
                        })
    
    return players


def save_to_csv(players: List[Dict[str, Any]], output_path: str) -> None:
    """Save player data to CSV file."""
    fieldnames = ['overall_rank', 'position', 'position_rank', 'player_name', 'team', 'salary_value', 'bye_week']
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(players)


def main():
    """Main function to extract ESPN projections and save to CSV."""
    parser = argparse.ArgumentParser(description='Extract ESPN projections from PDF to CSV')
    parser.add_argument('--pdf', default='espn-projections-non-ppr.pdf', 
                       help='Path to ESPN projections PDF')
    parser.add_argument('--output', default='data/espn_projections_20250814.csv',
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting data from {args.pdf}...")
    players = extract_player_data(args.pdf)
    
    if not players:
        print("No player data found. Check PDF format or parsing logic.")
        return
    
    print(f"Extracted {len(players)} players (including duplicates)")
    
    # Remove duplicates and sort by overall rank
    seen_players = set()
    unique_players = []
    for player in players:
        player_key = (player['overall_rank'], player['player_name'], player['team'])
        if player_key not in seen_players:
            seen_players.add(player_key)
            unique_players.append(player)
    
    unique_players.sort(key=lambda x: x['overall_rank'])
    
    save_to_csv(unique_players, args.output)
    print(f"Saved {len(unique_players)} unique players to {args.output}")
    
    # Print sample of extracted data
    print("\nSample data:")
    for player in unique_players[:5]:
        print(f"{player['overall_rank']}. ({player['position_rank']}) {player['player_name']}, {player['team']} ${player['salary_value']} Bye:{player['bye_week']}")


if __name__ == "__main__":
    main()