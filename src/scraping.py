"""
FantasyPros scraping utilities
"""
import requests
import pandas as pd
import time
from datetime import datetime
import logging

def scrape_position_projections(position, delay=2):
    """
    Scrape projections for a single position from FantasyPros
    
    Args:
        position (str): Position to scrape (qb, rb, wr, te, k, dst)
        delay (int): Delay between requests in seconds
        
    Returns:
        pd.DataFrame: Player projections for the position
    """
    url = f"https://www.fantasypros.com/nfl/projections/{position}.php?week=draft"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    try:
        logging.info(f"Scraping {position.upper()} projections...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML tables with pandas
        tables = pd.read_html(response.content)
        if not tables:
            logging.warning(f"No tables found for {position}")
            return pd.DataFrame()
            
        df = tables[0]
        
        # Clean up column names
        df.columns = [str(col).replace('Unnamed: 0_level_0_', '').strip() for col in df.columns]
        
        # Add position column
        df['Position'] = position.upper()
        
        logging.info(f"Successfully scraped {len(df)} {position.upper()} players")
        time.sleep(delay)
        
        return df
        
    except Exception as e:
        logging.error(f"Failed to scrape {position}: {str(e)}")
        return pd.DataFrame()

def scrape_all_positions(positions=['qb', 'rb', 'wr', 'te', 'k', 'dst'], delay=2):
    """
    Scrape projections for all specified positions
    
    Args:
        positions (list): List of positions to scrape
        delay (int): Delay between requests in seconds
        
    Returns:
        pd.DataFrame: Combined projections for all positions
    """
    all_data = []
    
    for position in positions:
        df = scrape_position_projections(position, delay)
        if not df.empty:
            all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        logging.info(f"Total players scraped: {len(combined_df)}")
        return combined_df
    else:
        logging.error("No data scraped successfully")
        return pd.DataFrame()

def save_projection_data(df, base_path="data/raw"):
    """
    Save projection data to CSV files
    
    Args:
        df (pd.DataFrame): Combined projection data
        base_path (str): Base directory for saving files
    """
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Save individual position files
    for position in df['Position'].unique():
        pos_df = df[df['Position'] == position]
        filename = f"{base_path}/projections_{position.lower()}_{timestamp}.csv"
        pos_df.to_csv(filename, index=False)
        logging.info(f"Saved {len(pos_df)} {position} players to {filename}")
    
    # Save combined file
    combined_filename = f"{base_path}/projections_all_positions_{timestamp}.csv"
    df.to_csv(combined_filename, index=False)
    logging.info(f"Saved {len(df)} total players to {combined_filename}")