#!/usr/bin/env python3
"""
FantasyPros ADP (Average Draft Position) Scraper
Scrapes overall ADP data from FantasyPros for fantasy football analysis
"""

import time
import csv
import os
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd
import argparse
from pathlib import Path
from io import StringIO

# Configuration
ADP_URL = "https://www.fantasypros.com/nfl/adp/overall.php"
OUTPUT_DIR = "data"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
DELAY_SECONDS = 2  # Be respectful to the server


def scrape_adp_data():
    """
    Scrape ADP data from FantasyPros
    Returns a pandas DataFrame with the ADP data
    """
    print("Scraping FantasyPros ADP data...")
    
    headers = {'User-Agent': USER_AGENT}
    
    try:
        response = requests.get(ADP_URL, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching ADP data: {e}")
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Look for the main data table - FantasyPros typically uses table with id="data"
    table = soup.find('table', {'id': 'data'})
    
    if not table:
        # Try alternative selectors
        table = soup.find('table', class_='table')
        if not table:
            print("Could not find ADP data table")
            return None
    
    try:
        # Parse the table using pandas
        dfs = pd.read_html(StringIO(str(table)))
        if not dfs:
            print("No data found in ADP table")
            return None
        
        df = dfs[0]
        
        # Clean column names
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten multi-level columns
            df.columns = ['_'.join(str(col).strip() for col in cols if str(col) != 'nan') for cols in df.columns.values]
        
        # Standardize column names
        df.columns = [str(col).upper().replace(' ', '_').replace('(', '').replace(')', '') for col in df.columns]
        
        # Add scrape metadata
        df['SCRAPE_DATE'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df['SOURCE'] = 'FantasyPros'
        
        # Clean up common column name variations
        column_mapping = {
            'PLAYER_NAME': 'PLAYER',
            'PLAYER_TEAM_BYE': 'PLAYER_TEAM_BYE',
            'POS': 'POSITION',
            'AVG_PICK': 'ADP',
            'AVG': 'ADP',
            'AVERAGE_DRAFT_POSITION': 'ADP'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Parse PLAYER_TEAM_BYE column to extract individual components
        if 'PLAYER_TEAM_BYE' in df.columns:
            # Split the combined player/team/bye column
            # Format is typically: "Player Name Team (Bye)"
            import re
            
            def parse_player_info(player_str):
                if pd.isna(player_str):
                    return pd.Series(['', '', ''])
                
                # Pattern to match: "Player Name TEAM (BYE)" or "Player Name TEAM BYE"
                match = re.match(r'(.+?)\s+([A-Z]{2,3})\s*\(?(\d+)?\)?', str(player_str))
                if match:
                    player_name = match.group(1).strip()
                    team = match.group(2).strip()
                    bye_week = match.group(3) if match.group(3) else ''
                    return pd.Series([player_name, team, bye_week])
                else:
                    return pd.Series([str(player_str), '', ''])
            
            # Apply the parsing function
            df[['PLAYER', 'TEAM', 'BYE_WEEK']] = df['PLAYER_TEAM_BYE'].apply(parse_player_info)
            
            # Remove the original combined column
            df = df.drop('PLAYER_TEAM_BYE', axis=1)
        
        print(f"Successfully scraped {len(df)} players from ADP data")
        print(f"Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"Error parsing ADP table: {e}")
        return None


def save_adp_data(df, output_path=None):
    """
    Save ADP DataFrame to CSV
    """
    if df is None or df.empty:
        print("No ADP data to save")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate filename with timestamp if not provided
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d')
        output_path = f"{OUTPUT_DIR}/fantasypros_adp_{timestamp}.csv"
    
    try:
        df.to_csv(output_path, index=False)
        print(f"Saved ADP data to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving ADP data: {e}")
        return False


def validate_adp_data(df):
    """
    Basic validation to ensure we got meaningful ADP data
    """
    if df is None or df.empty:
        print("No data to validate")
        return False
    
    # Check for required columns
    required_cols = ['PLAYER']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return False
    
    # Check for ADP-related columns
    adp_cols = [col for col in df.columns if 'ADP' in col or 'PICK' in col or 'AVG' in col]
    if not adp_cols:
        print("Warning: No ADP-related columns found")
        print(f"Available columns: {list(df.columns)}")
        return False
    
    print(f"Validation passed. Found ADP columns: {adp_cols}")
    return True


def main():
    """Main function to scrape FantasyPros ADP data and save to CSV."""
    parser = argparse.ArgumentParser(description='Scrape FantasyPros ADP data to CSV')
    parser.add_argument('--output', 
                       help='Output CSV file path (default: data/fantasypros_adp_YYYYMMDD.csv)')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation on scraped data')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None
    
    print("="*50)
    print("FANTASYPROS ADP SCRAPER")
    print("="*50)
    print(f"Target URL: {ADP_URL}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)
    
    # Scrape the data
    df = scrape_adp_data()
    
    if df is None:
        print("Failed to scrape ADP data")
        return 1
    
    # Validate if requested
    if args.validate:
        if not validate_adp_data(df):
            print("Data validation failed")
            return 1
    
    # Save the data
    if not save_adp_data(df, args.output):
        print("Failed to save ADP data")
        return 1
    
    # Summary
    print("\n" + "="*50)
    print("SCRAPING SUMMARY")
    print("="*50)
    print(f"Players scraped: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Sample columns: {list(df.columns)[:5]}")
    
    if len(df) > 0:
        print("\nSample data (first 3 rows):")
        print(df.head(3).to_string())
    
    return 0


if __name__ == "__main__":
    exit(main())