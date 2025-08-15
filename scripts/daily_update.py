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

def validate_vbd_data_quality(df, config):
    """
    Validate data quality specifically for VBD calculations
    
    Args:
        df (pd.DataFrame): Player projections dataframe
        config (dict): League configuration
        
    Returns:
        dict: Validation results with warnings and critical issues
    """
    validation = {
        'is_valid': True,
        'is_critical': False,
        'warnings': [],
        'position_counts': {}
    }
    
    try:
        # Check required columns
        required_columns = ['POSITION', 'FANTASY_PTS', 'PLAYER']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation['warnings'].append(f"Missing required columns: {missing_columns}")
            validation['is_critical'] = True
            validation['is_valid'] = False
            return validation
        
        # Check for empty dataframe
        if df.empty:
            validation['warnings'].append("DataFrame is empty")
            validation['is_critical'] = True
            validation['is_valid'] = False
            return validation
        
        # Check minimum players per position
        min_players = config.get('vbd', {}).get('min_players_per_position', 3)
        roster_slots = config.get('roster', {}).get('roster_slots', {})
        
        for position in df['POSITION'].unique():
            pos_count = len(df[df['POSITION'] == position])
            validation['position_counts'][position] = pos_count
            
            if pos_count < min_players:
                validation['warnings'].append(f"Position {position} has only {pos_count} players, minimum {min_players} recommended")
                validation['is_valid'] = False
                
                # Critical if we have fewer players than roster requirements
                config_position = {'DST': 'DEF', 'DEF': 'DEF'}.get(position, position)
                required_starters = roster_slots.get(config_position, 1)
                teams = config.get('basic_settings', {}).get('teams', 12)
                
                if pos_count < (teams * required_starters):
                    validation['warnings'].append(f"Position {position} has insufficient players for league requirements")
                    validation['is_critical'] = True
        
        # Check for NaN values in critical columns
        nan_fantasy_pts = df['FANTASY_PTS'].isna().sum()
        if nan_fantasy_pts > 0:
            validation['warnings'].append(f"{nan_fantasy_pts} players have NaN fantasy points")
            validation['is_valid'] = False
        
        # Check for negative fantasy points (unusual but possible)
        negative_pts = (df['FANTASY_PTS'] < 0).sum()
        if negative_pts > 0:
            validation['warnings'].append(f"{negative_pts} players have negative fantasy points")
        
        # Check for duplicate players
        duplicates = df.duplicated(subset=['PLAYER']).sum()
        if duplicates > 0:
            validation['warnings'].append(f"{duplicates} duplicate players found")
            validation['is_valid'] = False
        
        logging.info(f"VBD data validation completed. Position counts: {validation['position_counts']}")
        
    except Exception as e:
        validation['warnings'].append(f"Validation error: {str(e)}")
        validation['is_critical'] = True
        validation['is_valid'] = False
    
    return validation

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
        
        # Additional VBD-specific data quality checks
        vbd_validation = validate_vbd_data_quality(df, config)
        if not vbd_validation['is_valid']:
            logging.warning(f"VBD data quality issues detected: {vbd_validation['warnings']}")
            if vbd_validation['is_critical']:
                logging.error("Critical VBD data quality issues - aborting update")
                return False
        
        # 4. Calculate VBD rankings
        logging.info("Step 4: Calculating VBD rankings")
        try:
            df_vbd = calculate_all_vbd_methods(df, config)
        except Exception as e:
            logging.error(f"VBD calculation failed: {str(e)}")
            return False
        
        # 5. Filter to top 300 and save
        logging.info("Step 5: Saving rankings")
        try:
            top_300 = get_top_players_by_vbd(df_vbd, method='VBD_BLENDED', top_n=300)
            if top_300.empty:
                logging.error("No players returned from VBD calculation")
                return False
            
            top_300['VBD_RANK'] = range(1, len(top_300) + 1)
        except Exception as e:
            logging.error(f"Error getting top players: {str(e)}")
            return False
        
        output_path = os.path.join(project_root, "data/output")
        
        # Save main rankings
        filename = save_rankings(top_300, filename_prefix="vbd_rankings_top300", output_path=output_path)
        
        # Save method-specific rankings
        for method in ['VBD_VOLS', 'VBD_VORP', 'VBD_BEER']:
            try:
                method_top300 = get_top_players_by_vbd(df_vbd, method=method, top_n=300)
                if not method_top300.empty:
                    save_rankings(
                        method_top300, 
                        filename_prefix=f"rankings_{method.lower()}_top300", 
                        output_path=output_path
                    )
                else:
                    logging.warning(f"No players found for method {method}")
            except Exception as e:
                logging.error(f"Error saving rankings for method {method}: {str(e)}")
                # Continue with other methods rather than failing
        
        logging.info(f"Update completed successfully. Rankings saved to {filename}")
        
        # Print summary
        print(f"✅ Daily update completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Processed {len(df)} total players")
        print(f"🏆 Top 300 rankings saved to {filename}")
        print(f"📈 VBD range: {top_300['VBD_BLENDED'].max():.2f} to {top_300['VBD_BLENDED'].min():.2f}")
        
        # Show position distribution
        dist = top_300['POSITION'].value_counts()
        print(f"📍 Position distribution: {dict(dist)}")
        
        return True
        
    except Exception as e:
        logging.error(f"Update failed: {str(e)}")
        print(f"❌ Update failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)