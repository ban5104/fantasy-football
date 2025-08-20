#!/usr/bin/env python3
"""
Data Cleanup Utility for Monte Carlo Simulation Files

This script helps manage simulation data files to ensure you're always 
working with the latest data in your Jupyter notebooks.
"""

import os
from pathlib import Path
import datetime
import argparse

def list_simulation_files(data_dir='data/cache', strategy=None, pick=None):
    """List all simulation files with timestamps"""
    cache_path = Path(data_dir)
    
    if not cache_path.exists():
        print(f"❌ Cache directory {cache_path} not found")
        return
    
    # Find all .parquet files
    pattern = "*.parquet"
    if strategy:
        pattern = f"{strategy}_*.parquet"
    
    files = list(cache_path.glob(pattern))
    
    if not files:
        print(f"❌ No simulation files found in {cache_path}")
        return
    
    # Group files by parameters
    file_groups = {}
    
    for file in files:
        # Skip processed/cache files
        if any(x in file.stem for x in ['processed', 'replacement', 'summary', 'comparison', 'metrics', 'probs']):
            continue
            
        # Parse filename: strategy_pick5_n200_r14.parquet
        if '_pick' not in file.stem:
            continue
            
        try:
            parts = file.stem.split('_pick')
            file_strategy = parts[0]
            rest = parts[1].split('_')
            file_pick = int(rest[0])
            file_n_sims = int(rest[1][1:])  # Remove 'n' prefix
            
            # Filter if requested
            if strategy and file_strategy != strategy:
                continue
            if pick and file_pick != pick:
                continue
                
            key = (file_strategy, file_pick, file_n_sims)
            if key not in file_groups:
                file_groups[key] = []
            file_groups[key].append(file)
            
        except (ValueError, IndexError):
            continue
    
    # Display grouped files
    print(f"📊 Simulation Data Files in {cache_path}")
    print("=" * 80)
    
    for (strategy, pick, n_sims), files in sorted(file_groups.items()):
        print(f"\n{strategy.upper()} | Pick #{pick} | {n_sims} sims:")
        print("-" * 50)
        
        # Sort by timestamp (newest first)
        files_with_time = [(f, f.stat().st_mtime) for f in files]
        files_with_time.sort(key=lambda x: x[1], reverse=True)
        
        for i, (file, mtime) in enumerate(files_with_time):
            timestamp = datetime.datetime.fromtimestamp(mtime)
            age = datetime.datetime.now() - timestamp
            marker = "✅ LATEST" if i == 0 else f"   {age.days}d {age.seconds//3600}h old"
            print(f"   {marker}: {file.name} ({timestamp.strftime('%m/%d %H:%M')})")

def clean_old_files(data_dir='data/cache', strategy=None, pick=None, keep_latest=2, dry_run=True):
    """Clean up old simulation files, keeping only the most recent ones"""
    cache_path = Path(data_dir)
    
    if not cache_path.exists():
        print(f"❌ Cache directory {cache_path} not found")
        return
    
    # Find all .parquet files
    files = list(cache_path.glob("*.parquet"))
    
    # Group files by parameters
    file_groups = {}
    
    for file in files:
        # Skip processed/cache files
        if any(x in file.stem for x in ['processed', 'replacement', 'summary', 'comparison', 'metrics', 'probs']):
            continue
            
        if '_pick' not in file.stem:
            continue
            
        try:
            parts = file.stem.split('_pick')
            file_strategy = parts[0]
            rest = parts[1].split('_')
            file_pick = int(rest[0])
            file_n_sims = int(rest[1][1:])
            
            # Filter if requested
            if strategy and file_strategy != strategy:
                continue
            if pick and file_pick != pick:
                continue
                
            key = (file_strategy, file_pick, file_n_sims)
            if key not in file_groups:
                file_groups[key] = []
            file_groups[key].append(file)
            
        except (ValueError, IndexError):
            continue
    
    # Clean each group
    total_deleted = 0
    total_saved = 0
    
    for (strategy, pick, n_sims), files in file_groups.items():
        if len(files) <= keep_latest:
            continue
            
        # Sort by timestamp (newest first)
        files_with_time = [(f, f.stat().st_mtime) for f in files]
        files_with_time.sort(key=lambda x: x[1], reverse=True)
        
        files_to_keep = files_with_time[:keep_latest]
        files_to_delete = files_with_time[keep_latest:]
        
        if files_to_delete:
            print(f"\n{strategy.upper()} | Pick #{pick} | {n_sims} sims:")
            print(f"   Keeping {len(files_to_keep)} latest files")
            
            for file, mtime in files_to_delete:
                timestamp = datetime.datetime.fromtimestamp(mtime)
                if dry_run:
                    print(f"   🗑️  Would delete: {file.name} ({timestamp.strftime('%m/%d %H:%M')})")
                else:
                    print(f"   🗑️  Deleting: {file.name} ({timestamp.strftime('%m/%d %H:%M')})")
                    file.unlink()
                total_deleted += 1
            
            total_saved += len(files_to_keep)
    
    if dry_run:
        print(f"\n📊 DRY RUN Summary:")
        print(f"   Would delete: {total_deleted} files")
        print(f"   Would keep: {total_saved} files")
        print(f"\n   Run with --execute to actually delete files")
    else:
        print(f"\n📊 Cleanup Summary:")
        print(f"   Deleted: {total_deleted} files")
        print(f"   Kept: {total_saved} latest files")

def main():
    parser = argparse.ArgumentParser(description='Manage Monte Carlo simulation data files')
    parser.add_argument('command', choices=['list', 'clean'], help='Command to run')
    parser.add_argument('--strategy', help='Filter by strategy (e.g., balanced, zero_rb)')
    parser.add_argument('--pick', type=int, help='Filter by draft pick number')
    parser.add_argument('--keep', type=int, default=2, help='Number of latest files to keep (default: 2)')
    parser.add_argument('--execute', action='store_true', help='Actually delete files (default is dry run)')
    parser.add_argument('--data-dir', default='data/cache', help='Data directory path')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_simulation_files(args.data_dir, args.strategy, args.pick)
    elif args.command == 'clean':
        clean_old_files(args.data_dir, args.strategy, args.pick, args.keep, not args.execute)

if __name__ == '__main__':
    main()