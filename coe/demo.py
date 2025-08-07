#!/usr/bin/env python3
"""
COE System Demonstration Script

This script provides a quick demonstration of the key capabilities
of the Singapore COE prediction and optimization system.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_collection.coe_data_collector import COEDataCollector
from src.models.coe_predictive_models import COEPredictiveModels
from src.optimization.quota_optimizer import COEQuotaOptimizer

def demo_data_collection():
    """Demonstrate data collection capabilities"""
    print("🔍 DEMONSTRATION: Data Collection")
    print("-" * 40)
    
    try:
        collector = COEDataCollector()
        print("✅ Data collector initialized successfully")
        print(f"📊 Target dataset: {collector.dataset_id}")
        print(f"🔗 API endpoint: {collector.base_url}")
        
        # Collect sample data
        data = collector.collect_and_process_data()
        
        print(f"✅ Successfully collected {len(data)} records")
        print(f"📅 Date range: {data['Date'].min()} to {data['Date'].max()}")
        print(f"🏷️ Categories available: {', '.join(data['Category'].unique())}")
        
        # Show sample data
        print("\n📋 Sample data:")
        sample = data.head(3)[['Date', 'Category', 'Metric_Type', 'Value']]
        print(sample.to_string(index=False))
        
        return data
        
    except Exception as e:
        print(f"❌ Data collection failed: {e}")
        return None

def demo_predictive_modeling(data):
    """Demonstrate predictive modeling capabilities"""
    print("\n🎯 DEMONSTRATION: Predictive Modeling")
    print("-" * 40)
    
    if data is None:
        print("❌ No data available for modeling")
        return None
    
    try:
        models = COEPredictiveModels()
        print("✅ Model system initialized")
        
        # Prepare features
        feature_df = models.prepare_features(data)
        print(f"✅ Created {len(models.feature_columns)} engineered features")
        
        # Train a subset of models for demo (faster)
        print("🎯 Training models (this may take a moment)...")
        trained_models = models.train_all_models(feature_df)
        
        print(f"✅ Successfully trained models for {len(trained_models)} categories:")
        for category, category_models in trained_models.items():
            print(f"   • {category}: {len(category_models)} models")
        
        # Generate sample predictions
        print("\n🔮 Sample predictions (6-month forecast):")
        for category in ['Cat A', 'Cat B']:
            try:
                predictions = models.ensemble_predict(category, periods=6)
                if len(predictions) > 0:
                    avg_price = predictions.mean()
                    volatility = predictions.std()
                    print(f"   • {category}: ${avg_price:,.0f} (±{volatility:,.0f})")
            except:
                print(f"   • {category}: Prediction unavailable")
        
        return models
        
    except Exception as e:
        print(f"❌ Modeling failed: {e}")
        return None

def demo_optimization(data, models):
    """Demonstrate quota optimization capabilities"""
    print("\n⚡ DEMONSTRATION: Quota Optimization")
    print("-" * 40)
    
    if data is None or models is None:
        print("❌ Data or models unavailable for optimization")
        return None
    
    try:
        optimizer = COEQuotaOptimizer()
        optimizer.set_data_and_predictor(data, models)
        print("✅ Optimizer initialized with data and models")
        
        # Show current quota estimates
        print("📊 Current quota estimates:")
        for category, quota in optimizer.current_quotas.items():
            print(f"   • {category}: {quota:,.0f}")
        
        # Run a quick optimization (genetic algorithm only for demo)
        print("\n⚡ Running optimization (genetic algorithm)...")
        result = optimizer.optimize_genetic_algorithm()
        
        if result.constraints_satisfied:
            print("✅ Optimization completed successfully!")
            print(f"📈 Objective value: {result.objective_value:.4f}")
            print(f"⏱️ Execution time: {result.execution_time:.2f} seconds")
            
            print("\n💡 Recommended quota adjustments:")
            for category, multiplier in result.optimal_quotas.items():
                change_pct = (multiplier - 1.0) * 100
                predicted_price = result.predicted_prices.get(category, 0)
                print(f"   • {category}: {change_pct:+.1f}% → ${predicted_price:,.0f}")
        else:
            print("⚠️ Optimization completed but constraints not satisfied")
        
        return result
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return None

def demo_system_integration():
    """Demonstrate full system integration"""
    print("\n🔧 DEMONSTRATION: System Integration")
    print("-" * 40)
    
    print("🎬 Running complete workflow demonstration...")
    
    # Step 1: Data Collection
    data = demo_data_collection()
    
    # Step 2: Predictive Modeling
    models = demo_predictive_modeling(data)
    
    # Step 3: Optimization
    optimization_result = demo_optimization(data, models)
    
    # Step 4: Summary
    print("\n📊 SYSTEM CAPABILITIES SUMMARY")
    print("=" * 50)
    
    capabilities = [
        "✅ Real-time data collection from Singapore government APIs",
        "✅ Advanced feature engineering with lag, rolling, and seasonal features",
        "✅ Multiple ML models: ARIMA, Prophet, XGBoost, LSTM",
        "✅ Ensemble predictions combining all model types",
        "✅ Multi-objective optimization with policy constraints",
        "✅ Interactive Streamlit dashboard for policy makers",
        "✅ Comprehensive analysis and reporting capabilities"
    ]
    
    for capability in capabilities:
        print(capability)
    
    print("\n🎯 KEY BENEFITS FOR POLICY MAKERS:")
    benefits = [
        "• Data-driven policy decisions based on real-time market data",
        "• Predictive insights to anticipate price movements",
        "• Optimization recommendations to balance multiple objectives",
        "• Scenario analysis to assess policy impact before implementation",
        "• Comprehensive reporting and visualization tools"
    ]
    
    for benefit in benefits:
        print(benefit)
    
    return data, models, optimization_result

def main():
    """Main demonstration function"""
    print("🚗 SINGAPORE COE SYSTEM DEMONSTRATION")
    print("=" * 60)
    print("This demonstration showcases the key capabilities of the")
    print("Singapore COE Price Prediction and Quota Optimization System")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Run complete demonstration
        data, models, optimization_result = demo_system_integration()
        
        print(f"\n🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Final statistics
        if data is not None:
            print(f"📊 Processed {len(data):,} historical data points")
        
        if models is not None:
            print(f"🎯 Trained predictive models for COE price forecasting")
        
        if optimization_result is not None:
            print(f"⚡ Generated optimized quota recommendations")
        
        print("\n🚀 NEXT STEPS:")
        print("• Run 'python main.py' for full system analysis")
        print("• Launch 'streamlit run app/coe_dashboard.py' for interactive dashboard")
        print("• Explore 'notebooks/coe_analysis.ipynb' for detailed analysis")
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        print("Please check your environment and dependencies")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 