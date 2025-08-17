#!/usr/bin/env python3
"""
Test script for Dynamic VBD integration with backup_draft.py
"""

import pandas as pd
import os
import sys

def test_vbd_file_detection():
    """Test that VBD files are properly detected."""
    print("🧪 Testing VBD file detection...")
    
    # Check for VBD ranking files
    vbd_sources = []
    if os.path.exists("data/output"):
        import glob
        vbd_patterns = [
            "data/output/vbd_rankings_top300_*.csv",
            "data/output/rankings_vbd_*_top300_*.csv", 
            "data/output/rankings_statistical_vbd_top300_*.csv"
        ]
        for pattern in vbd_patterns:
            vbd_sources.extend(sorted(glob.glob(pattern), reverse=True))
    
    print(f"📂 Found {len(vbd_sources)} VBD ranking files:")
    for source in vbd_sources[:3]:  # Show first 3
        print(f"   - {source}")
    
    return len(vbd_sources) > 0

def test_vbd_file_structure():
    """Test VBD file structure and columns."""
    print("\n🧪 Testing VBD file structure...")
    
    # Find most recent VBD file
    import glob
    vbd_files = sorted(glob.glob("data/output/vbd_rankings_top300_*.csv"), reverse=True)
    
    if not vbd_files:
        print("❌ No VBD ranking files found")
        return False
    
    test_file = vbd_files[0]
    print(f"📂 Testing file: {test_file}")
    
    try:
        df = pd.read_csv(test_file)
        print(f"✅ Loaded {len(df)} rows")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Check for required columns
        required_cols = ['Player', 'POSITION']
        vbd_cols = [col for col in df.columns if 'VBD' in col.upper()]
        fantasy_pts = 'FANTASY_PTS' in df.columns
        
        print(f"🔍 Required columns present: {all(col in df.columns for col in required_cols)}")
        print(f"📈 VBD columns: {vbd_cols}")
        print(f"⚡ FANTASY_PTS available: {fantasy_pts}")
        
        return len(vbd_cols) > 0 or fantasy_pts
        
    except Exception as e:
        print(f"❌ Error reading VBD file: {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    print("\n🧪 Testing configuration loading...")
    
    config_path = "config/league-config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        dynamic_vbd = config.get('dynamic_vbd', {})
        enabled = dynamic_vbd.get('enabled', False)
        
        print(f"✅ Config loaded successfully")
        print(f"🚀 Dynamic VBD enabled in config: {enabled}")
        print(f"📊 Dynamic VBD params: {dynamic_vbd.get('params', {})}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False

def test_import_structure():
    """Test that required modules can be imported."""
    print("\n🧪 Testing import structure...")
    
    try:
        # Test if src directory exists and has required modules
        sys.path.append('src')
        
        # Test basic imports (these might fail due to missing dependencies)
        test_imports = [
            ('yaml', 'PyYAML'),
            ('pandas', 'pandas'), 
            ('numpy', 'numpy')
        ]
        
        missing_deps = []
        for module, package in test_imports:
            try:
                __import__(module)
                print(f"✅ {module} available")
            except ImportError:
                print(f"❌ {module} missing (install {package})")
                missing_deps.append(package)
        
        # Test Dynamic VBD module
        if os.path.exists('src/dynamic_vbd.py'):
            print("✅ Dynamic VBD module found")
        else:
            print("❌ Dynamic VBD module not found")
            
        return len(missing_deps) == 0
        
    except Exception as e:
        print(f"❌ Import test error: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🏈 BACKUP DRAFT INTEGRATION TESTS")
    print("=" * 50)
    
    tests = [
        ("VBD File Detection", test_vbd_file_detection),
        ("VBD File Structure", test_vbd_file_structure),
        ("Configuration Loading", test_config_loading),
        ("Import Structure", test_import_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Summary: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🚀 All tests passed! Dynamic VBD integration should work properly.")
        print("\nTo use the enhanced backup draft:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run with Dynamic VBD: python backup_draft.py --dynamic-vbd")
        print("3. Use RANKINGS command during draft to see live VBD updates")
    else:
        print("\n⚠️  Some tests failed. Check missing dependencies or file paths.")
        print("The backup draft will still work but may fall back to static rankings.")

if __name__ == "__main__":
    main()