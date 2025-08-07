#!/usr/bin/env python3
"""
Minimal COE Demo - Only Basic Functions
Handles errors gracefully and provides clear feedback
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

def safe_import(module_name, package_name=None):
    """Safely import a module and return None if it fails"""
    try:
        if package_name:
            return __import__(package_name)
        else:
            return __import__(module_name)
    except ImportError:
        return None

def main():
    print("🚗 SINGAPORE COE SYSTEM - MINIMAL DEMO")
    print("=" * 60)
    print("This demo tests basic functionality with error handling")
    print("Started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Test 1: Basic imports
    print("\n📦 TESTING BASIC IMPORTS")
    print("-" * 40)
    
    required_modules = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'requests': 'requests',
        'yaml': 'yaml'
    }
    
    missing_modules = []
    for name, module in required_modules.items():
        if safe_import(module):
            print(f"✅ {name}")
        else:
            print(f"❌ {name} - NOT INSTALLED")
            missing_modules.append(name)
    
    if missing_modules:
        print(f"\n⚠️ Missing modules: {', '.join(missing_modules)}")
        print("Please install with: pip install -r requirements.txt")
        return False
    
    # Test 2: Configuration
    print("\n📋 TESTING CONFIGURATION")
    print("-" * 40)
    
    try:
        import yaml
        config_path = "config/config.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            print("✅ Configuration file loaded")
        else:
            print(f"❌ Configuration file not found: {config_path}")
            return False
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False
    
    # Test 3: Data Collection (with fallback)
    print("\n📊 TESTING DATA COLLECTION")
    print("-" * 40)
    
    try:
        # Try to import the data collector
        sys.path.append('src')
        from data_collection.coe_data_collector import COEDataCollector
        
        collector = COEDataCollector()
        print("✅ Data collector initialized")
        
        # Try to collect data
        data = collector.collect_and_process_data()
        print(f"✅ Successfully collected {len(data):,} records")
        
        # Show basic info
        if len(data) > 0:
            print(f"📅 Date range: {data['Date'].min()} to {data['Date'].max()}")
            categories = data['Category'].unique()
            print(f"🏷️ Categories: {', '.join(categories)}")
            
            # Show recent prices
            quota_data = data[data['Metric_Type'] == 'Quota Premium']
            if len(quota_data) > 0:
                print("\n💰 Recent COE Prices:")
                for category in ['Cat A', 'Cat B', 'Cat C', 'Cat D']:
                    cat_data = quota_data[quota_data['Category'] == category]
                    if len(cat_data) > 0:
                        latest = cat_data.iloc[-1]
                        print(f"   • {category}: ${latest['Value']:,.0f} ({latest['Date'].strftime('%Y-%m')})")
        
        return True
        
    except Exception as e:
        print(f"❌ Data collection failed: {e}")
        print("\n🔍 Error details:")
        traceback.print_exc()
        
        # Provide fallback information
        print("\n📋 FALLBACK INFORMATION:")
        print("The COE system collects data from Singapore's data.gov.sg API")
        print("Dataset: Motor Vehicle Quota, Quota Premium And Prevailing Quota Premium")
        print("API URL: https://data.gov.sg/api/action/datastore_search")
        print("Dataset ID: d_22094bf608253d36c0c63b52d852dd6e")
        
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🎉 MINIMAL DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("✅ Basic system components are working")
        print("✅ You can now try the full demo: python demo_simple.py")
    else:
        print(f"\n❌ MINIMAL DEMO FAILED")
        print("=" * 60)
        print("🔧 Troubleshooting steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Check internet connection (for API access)")
        print("3. Verify configuration file exists")
        print("4. Check file permissions")
    
    sys.exit(0 if success else 1) 