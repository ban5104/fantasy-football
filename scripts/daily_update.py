#!/usr/bin/env python3
"""
Daily update script for fantasy football projections

This script:
1. Scrapes latest projections from FantasyPros
2. Calculates fantasy points based on league settings
3. Computes VBD rankings
4. Saves updated rankings to output directory

Usage:
    python scripts/daily_update.py

Can be run via cron for automated updates:
    0 6 * * * cd /path/to/project && python scripts/daily_update.py
"""

import sys
import os
from datetime import datetime

# Add src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from scraping import scrape_all_positions, save_projection_data
from scoring import load_league_config, prepare_stats_columns, calculate_fantasy_points_vectorized, rank_players_by_position
from vbd import calculate_all_vbd_methods, get_top_players_by_vbd
from utils import setup_logging, save_rankings, validate_data_quality
import logging

def main():
    """Main update pipeline"""
    try:
        # Setup logging
        setup_logging()
        logging.info("Starting daily fantasy football update")
        
        # 1. Scrape latest projections
        logging.info("Step 1: Scraping projections from FantasyPros")
        positions = ['qb', 'rb', 'wr', 'te', 'k', 'dst']
        df = scrape_all_positions(positions, delay=2)
        
        if df.empty:
            logging.error("No data scraped - aborting update")
            return False
        
        # Save raw projections
        save_projection_data(df, base_path=os.path.join(project_root, "data/raw"))
        
        # 2. Load league configuration
        logging.info("Step 2: Loading league configuration")
        config_path = os.path.join(project_root, "config/league-config.yaml")
        config = load_league_config(config_path)
        
        if not config:
            logging.error("Failed to load league configuration")
            return False
        
        # 3. Calculate fantasy points
        logging.info("Step 3: Calculating fantasy points")
        df = prepare_stats_columns(df)
        df = calculate_fantasy_points_vectorized(df, config['scoring'])
        df = rank_players_by_position(df)
        
        # Validate calculations
        validation = validate_data_quality(df)
        logging.info(f"Data validation: {validation}")
        
        # 4. Calculate VBD rankings
        logging.info("Step 4: Calculating VBD rankings")
        df_vbd = calculate_all_vbd_methods(df, config)
        
        # 5. Filter to top 300 and save
        logging.info("Step 5: Saving rankings")
        top_300 = get_top_players_by_vbd(df_vbd, method='VBD_BLENDED', top_n=300)
        top_300['VBD_RANK'] = range(1, len(top_300) + 1)
        
        output_path = os.path.join(project_root, "data/output")
        
        # Save main rankings
        filename = save_rankings(top_300, filename_prefix="vbd_rankings_top300", output_path=output_path)
        
        # Save method-specific rankings
        for method in ['VBD_VOLS', 'VBD_VORP', 'VBD_BEER']:
            method_top300 = get_top_players_by_vbd(df_vbd, method=method, top_n=300)
            save_rankings(
                method_top300, 
                filename_prefix=f"rankings_{method.lower()}_top300", 
                output_path=output_path
            )
        
        logging.info(f"Update completed successfully. Rankings saved to {filename}")
        
        # Print summary
        print(f"✅ Daily update completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Processed {len(df)} total players")
        print(f"🏆 Top 300 rankings saved to {filename}")
        print(f"📈 VBD range: {top_300['VBD_BLENDED'].max():.2f} to {top_300['VBD_BLENDED'].min():.2f}")
        
        # Show position distribution
        dist = top_300['Position'].value_counts()
        print(f"📍 Position distribution: {dict(dist)}")
        
        return True
        
    except Exception as e:
        logging.error(f"Update failed: {str(e)}")
        print(f"❌ Update failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)