#!/usr/bin/env python3
"""
Simple COE Demo - Only Core Functions That Work
Avoids all complex modeling and optimization
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

def main():
    print("🚗 SINGAPORE COE SYSTEM - SIMPLE DEMO")
    print("=" * 60)
    print("This demo only tests the working components")
    print("Started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    try:
        # Test 1: Data Collection (we know this works)
        print("\n📊 TESTING DATA COLLECTION")
        print("-" * 40)
        
        from src.data_collection.coe_data_collector import COEDataCollector
        
        collector = COEDataCollector()
        print("✅ Data collector initialized")
        
        data = collector.collect_and_process_data()
        print(f"✅ Successfully collected {len(data):,} records")
        print(f"📅 Date range: {data['Date'].min()} to {data['Date'].max()}")
        
        # Show categories
        categories = data['Category'].unique()
        print(f"🏷️ Categories found: {', '.join(categories)}")
        
        # Test 2: Basic Data Analysis (safe operations)
        print("\n📈 BASIC DATA ANALYSIS")
        print("-" * 40)
        
        # Get quota premium data only
        quota_data = data[data['Metric_Type'] == 'Quota Premium'].copy()
        print(f"✅ Found {len(quota_data)} quota premium records")
        
        if len(quota_data) > 0:
            # Show recent prices by category
            print("\n💰 Recent COE Prices (Last Available):")
            for category in ['Cat A', 'Cat B', 'Cat C', 'Cat D']:
                cat_data = quota_data[quota_data['Category'] == category]
                if len(cat_data) > 0:
                    latest_price = cat_data.iloc[-1]['Value']
                    latest_date = cat_data.iloc[-1]['Date']
                    print(f"   • {category}: ${latest_price:,.0f} ({latest_date.strftime('%Y-%m')})")
            
            # Simple statistics
            print(f"\n📊 PRICE STATISTICS:")
            print(f"   • Average price: ${quota_data['Value'].mean():,.0f}")
            print(f"   • Highest price: ${quota_data['Value'].max():,.0f}")
            print(f"   • Lowest price: ${quota_data['Value'].min():,.0f}")
            print(f"   • Price volatility: {quota_data['Value'].std():.0f}")
        
        # Test 3: Simple Trend Analysis (no ML)
        print("\n📈 SIMPLE TREND ANALYSIS")
        print("-" * 40)
        
        # Calculate simple moving averages for each category
        for category in ['Cat A', 'Cat B']:
            cat_data = quota_data[quota_data['Category'] == category].copy()
            if len(cat_data) >= 12:
                cat_data = cat_data.sort_values('Date')
                cat_data['MA_6'] = cat_data['Value'].rolling(6).mean()
                cat_data['MA_12'] = cat_data['Value'].rolling(12).mean()
                
                recent_price = cat_data.iloc[-1]['Value']
                ma_6 = cat_data.iloc[-1]['MA_6']
                ma_12 = cat_data.iloc[-1]['MA_12']
                
                print(f"✅ {category} Trend Analysis:")
                print(f"   • Current: ${recent_price:,.0f}")
                print(f"   • 6-month avg: ${ma_6:,.0f}")
                print(f"   • 12-month avg: ${ma_12:,.0f}")
                
                if recent_price > ma_6:
                    print(f"   • Trend: Above 6-month average (📈 Rising)")
                else:
                    print(f"   • Trend: Below 6-month average (📉 Falling)")
        
        # Test 4: Save processed data
        print("\n💾 SAVING PROCESSED DATA")
        print("-" * 40)
        
        output_file = f"data/processed/coe_analysis_{datetime.now().strftime('%Y%m%d')}.csv"
        os.makedirs('data/processed', exist_ok=True)
        quota_data.to_csv(output_file, index=False)
        print(f"✅ Data saved to: {output_file}")
        
        # Success summary
        print(f"\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("✅ Core system components are working properly")
        print("✅ Data collection: PASSED")
        print("✅ Data processing: PASSED") 
        print("✅ Basic analysis: PASSED")
        print("✅ File operations: PASSED")
        
        print(f"\n📊 Summary: Processed {len(data):,} total records")
        print(f"📅 Time period: {data['Date'].min().strftime('%Y-%m')} to {data['Date'].max().strftime('%Y-%m')}")
        
        print(f"\n🚀 Next Steps:")
        print("   • For full ML models: streamlit run app/coe_dashboard.py")
        print("   • For analysis: jupyter notebook notebooks/coe_analysis.ipynb")
        print("   • This simple demo took:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("🔍 This suggests an issue with the basic system setup")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 