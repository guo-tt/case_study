#!/usr/bin/env python3
"""
Fast Demo - Singapore COE System (Optimized for Speed)
Skips complex optimization and uses lightweight models only
"""

import sys
import os
import logging
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.WARNING)  # Reduce log noise

def print_header(title):
    print(f"\n🚗 {title}")
    print("=" * 60)

def demo_fast_data_collection():
    """Quick data collection demo"""
    print_header("FAST DATA COLLECTION")
    
    try:
        from src.data_collection.coe_data_collector import COEDataCollector
        
        collector = COEDataCollector()
        print("✅ Data collector initialized")
        
        # Use smaller dataset for speed
        data = collector.collect_and_process_data()
        
        print(f"📊 Collected: {len(data):,} records")
        print(f"📅 Date range: {data['Date'].min()} to {data['Date'].max()}")
        print(f"🏷️ Categories: {', '.join(data['Category'].unique())}")
        
        # Show sample data
        quota_premium_data = data[data['Metric_Type'] == 'Quota Premium'].head(3)
        if not quota_premium_data.empty:
            print(f"\n📋 Recent COE Quota Premiums:")
            for _, row in quota_premium_data.iterrows():
                print(f"   • {row['Category']}: ${row['Value']:,.0f} ({row['Date'].strftime('%Y-%m')})")
        
        return data
        
    except Exception as e:
        print(f"❌ Data collection failed: {e}")
        return None

def demo_fast_modeling(data):
    """Quick modeling demo using only lightweight models"""
    print_header("FAST PREDICTIVE MODELING")
    
    if data is None:
        print("❌ No data available")
        return None
    
    try:
        from src.models.coe_predictive_models import COEPredictiveModels
        
        # Filter to recent data only for speed
        recent_data = data[data['Date'] >= '2020-01-01'].copy()
        quota_premium_data = recent_data[recent_data['Metric_Type'] == 'Quota Premium'].copy()
        
        if len(quota_premium_data) < 50:
            print("⚠️ Insufficient recent data, using full dataset")
            quota_premium_data = data[data['Metric_Type'] == 'Quota Premium'].copy()
        
        print(f"📊 Using {len(quota_premium_data)} quota premium records")
        
        models = COEPredictiveModels()
        
        # Prepare features (faster with less data)
        print("🔧 Preparing features...")
        featured_data = models.prepare_features(quota_premium_data)
        print(f"✅ Created {featured_data.shape[1]} features")
        
        # Quick prediction using simple model (XGBoost only)
        print("🎯 Training lightweight models...")
        
        categories_with_data = []
        predictions = {}
        
        for category in ['Cat A', 'Cat B', 'Cat C', 'Cat D']:
            cat_data = featured_data[featured_data['Category'] == category]
            if len(cat_data) >= 20:  # Minimum for training
                try:
                    train_data, val_data, test_data = models.split_data(cat_data)
                    if len(train_data) >= 10:
                        # Train only XGBoost for speed
                        model_result = models.train_xgboost_model(train_data, val_data, category)
                        if model_result:
                            categories_with_data.append(category)
                            # Simple prediction
                            pred = models.predict(category, 'xgboost', periods=1)
                            if pred is not None and len(pred) > 0:
                                predictions[category] = pred[0]
                            print(f"✅ {category}: Model trained successfully")
                except Exception as e:
                    print(f"⚠️ {category}: {str(e)[:50]}...")
        
        if predictions:
            print(f"\n🔮 SAMPLE PREDICTIONS (Next Month):")
            for cat, pred in predictions.items():
                print(f"   • {cat}: ${pred:,.0f}")
        
        return models
        
    except Exception as e:
        print(f"❌ Modeling failed: {str(e)[:100]}...")
        return None

def demo_fast_summary():
    """Quick system summary"""
    print_header("FAST SYSTEM SUMMARY")
    
    print("✅ **OPTIMIZED FOR SPEED:**")
    print("   • Uses recent data only (2020+)")
    print("   • Single model type (XGBoost)")  
    print("   • Skips complex optimization")
    print("   • Reduced feature engineering")
    
    print("\n🚀 **FOR FULL ANALYSIS:**")
    print("   • Run: streamlit run app/coe_dashboard.py")
    print("   • Use: notebooks/coe_analysis.ipynb")
    print("   • Full: python main.py --skip-optimization")
    
    print("\n💡 **PERFORMANCE TIPS:**")
    print("   • Use --limit 1000 for small data tests")
    print("   • Set PYTHONPATH for imports")
    print("   • Install conda for faster ML libraries")

def main():
    """Fast demo main function"""
    print("🚗 SINGAPORE COE SYSTEM - FAST DEMO")
    print("=" * 60)
    print("This optimized demo runs in ~30 seconds")
    print("Started at:", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Run fast components
    data = demo_fast_data_collection()
    models = demo_fast_modeling(data)
    demo_fast_summary()
    
    print(f"\n🎉 FAST DEMO COMPLETED!")
    print("Finished at:", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    if data is not None:
        print(f"📊 Processed {len(data):,} records successfully")

if __name__ == "__main__":
    main() 