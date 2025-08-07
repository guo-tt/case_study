#!/usr/bin/env python3
"""
Test Fast Optimization Script

This script tests the optimized quota optimizer to demonstrate fast performance
and reliable convergence.
"""

import sys
import os
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_collection.coe_data_collector import COEDataCollector
from src.models.coe_predictive_models import COEPredictiveModels
from src.optimization.optimized_quota_optimizer import OptimizedCOEQuotaOptimizer

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_fast_optimization():
    """Test the fast optimization system"""
    logger = setup_logging()
    
    logger.info("🚀 Testing Fast COE Quota Optimization")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load existing data (faster than collecting fresh)
        logger.info("Step 1: Loading existing data...")
        collector = COEDataCollector()
        
        # Try to load existing data first
        latest_file = os.path.join(collector.config['paths']['raw_data'], "coe_data_latest.csv")
        if os.path.exists(latest_file):
            logger.info(f"Loading existing data from {latest_file}")
            raw_data = collector.process_coe_data(collector.fetch_coe_data())
        else:
            logger.info("No existing data found, collecting fresh data...")
            raw_data = collector.collect_and_process_data()
        
        if raw_data.empty:
            logger.error("No data available for testing")
            return False
        
        logger.info(f"✅ Loaded {len(raw_data)} records")
        
        # Step 2: Train models
        logger.info("Step 2: Training predictive models...")
        models = COEPredictiveModels()
        feature_df = models.prepare_features(raw_data)
        models.train_all_models(feature_df)
        
        logger.info("✅ Models trained successfully")
        
        # Step 3: Run fast optimization
        logger.info("Step 3: Running fast optimization...")
        optimizer = OptimizedCOEQuotaOptimizer()
        optimizer.set_data_and_predictor(raw_data, models)
        
        # Run optimizations with timeout protection
        results = optimizer.run_fast_optimizations()
        
        if not results:
            logger.error("No optimization results obtained")
            return False
        
        # Step 4: Generate recommendations
        logger.info("Step 4: Generating policy recommendations...")
        recommendations = optimizer.generate_policy_recommendations(results)
        
        # Step 5: Display results
        logger.info("Step 5: Displaying results...")
        display_optimization_results(results, recommendations)
        
        logger.info("✅ Fast optimization test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Fast optimization test failed: {e}")
        logger.exception("Full traceback:")
        return False

def display_optimization_results(results, recommendations):
    """Display optimization results in a readable format"""
    print("\n" + "=" * 60)
    print("📊 FAST OPTIMIZATION RESULTS")
    print("=" * 60)
    
    # Display algorithm results
    print("\n🔬 Algorithm Performance:")
    print("-" * 40)
    for algorithm, result in results.items():
        status_icon = "✅" if result.convergence_status == "success" else "⚠️"
        print(f"{status_icon} {algorithm}:")
        print(f"   Objective Value: {result.objective_value:.4f}")
        print(f"   Execution Time: {result.execution_time:.2f}s")
        print(f"   Iterations: {result.iterations}")
        print(f"   Status: {result.convergence_status}")
        print()
    
    # Display recommendations
    if recommendations['status'] == 'success':
        print("🎯 POLICY RECOMMENDATIONS:")
        print("-" * 40)
        print(f"Best Algorithm: {recommendations['best_algorithm']}")
        print(f"Objective Value: {recommendations['objective_value']:.4f}")
        print(f"Total Execution Time: {recommendations['execution_time']:.2f}s")
        print(f"Summary: {recommendations['policy_summary']}")
        
        print("\n💡 Quota Adjustments:")
        print("-" * 40)
        for category, impact in recommendations['quota_adjustments'].items():
            change = impact['change_percentage']
            price = impact['predicted_price']
            current = impact['current_quota']
            recommended = impact['recommended_quota']
            
            change_icon = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            print(f"{change_icon} {category}:")
            print(f"   Current Quota: {current:,.0f}")
            print(f"   Recommended: {recommended:,.0f}")
            print(f"   Change: {change:+.1f}%")
            print(f"   Predicted Price: ${price:,.0f}")
            print()
    else:
        print("❌ No valid recommendations generated")
    
    # Performance summary
    print("⚡ PERFORMANCE SUMMARY:")
    print("-" * 40)
    total_time = sum(result.execution_time for result in results.values())
    successful_algorithms = sum(1 for result in results.values() if result.convergence_status == "success")
    
    print(f"Total Optimization Time: {total_time:.2f}s")
    print(f"Successful Algorithms: {successful_algorithms}/{len(results)}")
    print(f"Average Time per Algorithm: {total_time/len(results):.2f}s")
    
    if successful_algorithms > 0:
        best_time = min(result.execution_time for result in results.values() if result.convergence_status == "success")
        print(f"Fastest Successful Algorithm: {best_time:.2f}s")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    success = test_fast_optimization()
    sys.exit(0 if success else 1) 