#!/usr/bin/env python3
"""
Ultra-Fast COE Optimization Demo

This demo uses ONLY the fastest algorithm (simple gradient) for maximum speed.
Perfect for real-time applications and quick demonstrations.
"""

import sys
import os
import logging
import time
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_collection.coe_data_collector import COEDataCollector
from src.models.coe_predictive_models import COEPredictiveModels
from src.optimization.optimized_quota_optimizer import OptimizedCOEQuotaOptimizer

def setup_logging():
    """Setup minimal logging for speed"""
    logging.basicConfig(
        level=logging.WARNING,  # Reduce logging for speed
        format='%(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def ultra_fast_demo():
    """Ultra-fast optimization demo using only the fastest algorithm"""
    logger = setup_logging()
    
    print("🚀 ULTRA-FAST COE OPTIMIZATION DEMO")
    print("=" * 50)
    print("⚡ Using only the fastest algorithm (Simple Gradient)")
    print("⏱️  Expected time: < 1 second for optimization")
    print()
    
    start_total = time.time()
    
    try:
        # Step 1: Quick data loading
        print("📊 Loading data...")
        collector = COEDataCollector()
        
        # Use existing data if available
        latest_file = os.path.join(collector.config['paths']['raw_data'], "coe_data_latest.csv")
        if os.path.exists(latest_file):
            raw_data = collector.process_coe_data(collector.fetch_coe_data())
        else:
            raw_data = collector.collect_and_process_data()
        
        print(f"✅ Loaded {len(raw_data)} records")
        
        # Step 2: Fast model training
        print("🤖 Training models...")
        models = COEPredictiveModels()
        feature_df = models.prepare_features(raw_data)
        models.train_all_models(feature_df)
        print("✅ Models trained")
        
        # Step 3: Ultra-fast optimization (SIMPLE GRADIENT ONLY)
        print("⚡ Running ultra-fast optimization...")
        optimizer = OptimizedCOEQuotaOptimizer()
        optimizer.set_data_and_predictor(raw_data, models)
        
        # Configure for maximum speed
        optimizer.max_prediction_time = 10
        optimizer.cache_predictions = True
        
        # Run ONLY the fastest algorithm
        start_opt = time.time()
        result = optimizer.optimize_simple_gradient()
        opt_time = time.time() - start_opt
        
        total_time = time.time() - start_total
        
        # Display results
        print("\n" + "=" * 60)
        print("⚡ ULTRA-FAST OPTIMIZATION RESULTS")
        print("=" * 60)
        
        print(f"\n🏆 SIMPLE GRADIENT ALGORITHM:")
        print(f"   ⏱️  Optimization Time: {opt_time:.3f}s")
        print(f"   🎯 Objective Value: {result.objective_value:.2f}")
        print(f"   📊 Status: {result.convergence_status}")
        print(f"   🔄 Iterations: {result.iterations}")
        
        if result.convergence_status == "success":
            print("\n💡 RECOMMENDED QUOTA ADJUSTMENTS:")
            print("-" * 40)
            
            for category, multiplier in result.optimal_quotas.items():
                change_pct = (multiplier - 1.0) * 100
                predicted_price = result.predicted_prices.get(category, 0)
                current_quota = optimizer.current_quotas.get(category, 1000)
                new_quota = current_quota * multiplier
                
                change_icon = "📈" if change_pct > 0 else "📉" if change_pct < 0 else "➡️"
                print(f"{change_icon} {category}:")
                print(f"   Current: {current_quota:,.0f}")
                print(f"   Recommended: {new_quota:,.0f}")
                print(f"   Change: {change_pct:+.1f}%")
                print(f"   Predicted Price: ${predicted_price:,.0f}")
                print()
            
            # Calculate total revenue impact
            total_revenue = sum(
                result.predicted_prices.get(cat, 0) * optimizer.current_quotas.get(cat, 1000) * mult
                for cat, mult in result.optimal_quotas.items()
            )
            print(f"💰 Total Revenue Impact: ${total_revenue:,.0f}")
        
        print("\n📊 PERFORMANCE SUMMARY:")
        print("-" * 40)
        print(f"Total Demo Time: {total_time:.2f}s")
        print(f"Optimization Time: {opt_time:.3f}s")
        print(f"Data Loading & Model Training: {total_time - opt_time:.2f}s")
        print(f"Speed: {opt_time:.3f}s optimization time")
        
        if opt_time < 1.0:
            print("🚀 ULTRA-FAST: Optimization completed in under 1 second!")
        elif opt_time < 5.0:
            print("⚡ FAST: Optimization completed in under 5 seconds!")
        else:
            print("✅ GOOD: Optimization completed successfully!")
        
        print("\n" + "=" * 60)
        print("🎉 ULTRA-FAST DEMO COMPLETED SUCCESSFULLY!")
        print("💡 This algorithm is perfect for real-time applications!")
        
        return True
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        return False

if __name__ == "__main__":
    success = ultra_fast_demo()
    sys.exit(0 if success else 1) 