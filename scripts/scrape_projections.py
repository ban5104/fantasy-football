#!/usr/bin/env python3
"""
FantasyPros Season Projections Scraper
Simple, reliable scraper for NFL fantasy football projections

Run with: uv run python scrape_projections.py
"""

import time
import csv
import os
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Configuration
BASE_URL = "https://www.fantasypros.com/nfl/projections/{position}.php?week=draft"
OUTPUT_DIR = "data/projections"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
DELAY_SECONDS = 2  # Be respectful to the server

# Position-specific stat columns we care about
POSITION_STATS = {
    'qb': ['Player', 'Team', 'PASSING_ATT', 'PASSING_CMP', 'PASSING_YDS', 'PASSING_TDS', 
           'PASSING_INT', 'RUSHING_ATT', 'RUSHING_YDS', 'RUSHING_TDS', 'FL'],
    'rb': ['Player', 'Team', 'RUSHING_ATT', 'RUSHING_YDS', 'RUSHING_TDS', 
           'RECEIVING_REC', 'RECEIVING_YDS', 'RECEIVING_TDS', 'FL'],
    'wr': ['Player', 'Team', 'RECEIVING_REC', 'RECEIVING_TGT', 'RECEIVING_YDS', 
           'RECEIVING_TDS', 'RUSHING_ATT', 'RUSHING_YDS', 'RUSHING_TDS', 'FL'],
    'te': ['Player', 'Team', 'RECEIVING_REC', 'RECEIVING_TGT', 'RECEIVING_YDS', 
           'RECEIVING_TDS', 'FL'],
    'k': ['Player', 'Team', 'FG', 'FGA', 'XPT', 'FPTS'],
    'dst': ['Player', 'SACK', 'INT', 'FR', 'FF', 'TD', 'SAFETY', 'PA', 'YDS_AGN']
}


def scrape_position(position):
    """
    Scrape projections for a specific position from FantasyPros
    Returns a pandas DataFrame with the projection data
    """
    print(f"Scraping {position.upper()} projections...")
    
    url = BASE_URL.format(position=position)
    headers = {'User-Agent': USER_AGENT}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching {position}: {e}")
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the main data table
    # FantasyPros uses a table with id="data" for projections
    table = soup.find('table', {'id': 'data'})
    
    if not table:
        print(f"Could not find projection table for {position}")
        return None
    
    # Parse the table using pandas
    try:
        # Read the HTML table directly
        dfs = pd.read_html(str(table))
        if not dfs:
            print(f"No data found in table for {position}")
            return None
        
        df = dfs[0]
        
        # Clean column names - FantasyPros uses multi-level headers
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten multi-level columns
            df.columns = ['_'.join(col).strip('_') for col in df.columns.values]
        
        # Standardize column names
        df.columns = [col.upper().replace(' ', '_') for col in df.columns]
        
        # Fix specific column naming issues
        if any('PLAYER' in col for col in df.columns):
            # Find the player column and rename it to just 'PLAYER'
            player_cols = [col for col in df.columns if 'PLAYER' in col]
            if player_cols:
                df = df.rename(columns={player_cols[0]: 'PLAYER'})
        
        # Add position column
        df['POSITION'] = position.upper()
        
        # Add scrape timestamp
        df['SCRAPE_DATE'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"Successfully scraped {len(df)} {position.upper()} players")
        return df
        
    except Exception as e:
        print(f"Error parsing table for {position}: {e}")
        return None


def save_projections(df, position):
    """
    Save projections DataFrame to CSV
    """
    if df is None or df.empty:
        print(f"No data to save for {position}")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d')
    filename = f"{OUTPUT_DIR}/projections_{position}_{timestamp}.csv"
    
    try:
        df.to_csv(filename, index=False)
        print(f"Saved {position.upper()} projections to {filename}")
        return True
    except Exception as e:
        print(f"Error saving {position} data: {e}")
        return False


def scrape_all_positions():
    """
    Scrape projections for all positions
    """
    positions = ['qb', 'rb', 'wr', 'te', 'k', 'dst']
    all_data = {}
    
    print("Starting FantasyPros projection scraping...")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Delay between requests: {DELAY_SECONDS} seconds")
    print("-" * 50)
    
    for i, position in enumerate(positions):
        # Scrape the position
        df = scrape_position(position)
        
        if df is not None:
            all_data[position] = df
            save_projections(df, position)
        
        # Delay between requests (except for last one)
        if i < len(positions) - 1:
            print(f"Waiting {DELAY_SECONDS} seconds before next request...")
            time.sleep(DELAY_SECONDS)
    
    # Create combined dataset
    if all_data:
        print("\nCreating combined projections file...")
        combined_df = pd.concat(all_data.values(), ignore_index=True, sort=False)
        save_projections(combined_df, 'all_positions')
    
    print("\nScraping complete!")
    return all_data


def validate_data(df, position):
    """
    Basic validation to ensure we got meaningful data
    """
    if df is None or df.empty:
        return False
    
    # Check for required columns
    if 'PLAYER' not in df.columns and not any('PLAYER' in col for col in df.columns):
        print(f"Warning: No player column found for {position}")
        return False
    
    # Check we have numeric projection columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) < 3:
        print(f"Warning: Too few numeric columns for {position}")
        return False
    
    return True


if __name__ == "__main__":
    # Run the scraper
    data = scrape_all_positions()
    
    # Summary
    print("\n" + "="*50)
    print("SCRAPING SUMMARY")
    print("="*50)
    
    for position, df in data.items():
        if df is not None:
            print(f"{position.upper()}: {len(df)} players, {len(df.columns)} columns")
    
    print(f"\nAll projection files saved to: {os.path.abspath(OUTPUT_DIR)}")