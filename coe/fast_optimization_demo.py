#!/usr/bin/env python3
"""
Fast COE Optimization Demo

This demo uses only the fastest optimization algorithms to avoid slow genetic algorithms.
Perfect for quick demonstrations and real-time applications.
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
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class FastOptimizationDemo:
    """Fast optimization demo using only the fastest algorithms"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.optimizer = None
        self.results = {}
        
    def run_demo(self):
        """Run the complete fast optimization demo"""
        self.logger.info("🚀 FAST COE OPTIMIZATION DEMO")
        self.logger.info("=" * 50)
        
        try:
            # Step 1: Quick data loading
            self.logger.info("📊 Step 1: Loading data...")
            data = self._load_data_quick()
            
            # Step 2: Fast model training
            self.logger.info("🤖 Step 2: Training models...")
            models = self._train_models_fast(data)
            
            # Step 3: Ultra-fast optimization
            self.logger.info("⚡ Step 3: Running ultra-fast optimization...")
            self._run_ultra_fast_optimization(data, models)
            
            # Step 4: Display results
            self.logger.info("📈 Step 4: Displaying results...")
            self._display_results()
            
            self.logger.info("✅ Fast optimization demo completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Demo failed: {e}")
            return False
    
    def _load_data_quick(self):
        """Load data quickly using existing files if available"""
        collector = COEDataCollector()
        
        # Try to load existing data first
        latest_file = os.path.join(collector.config['paths']['raw_data'], "coe_data_latest.csv")
        if os.path.exists(latest_file):
            self.logger.info("📁 Loading existing data...")
            raw_data = collector.process_coe_data(collector.fetch_coe_data())
        else:
            self.logger.info("🔄 Collecting fresh data...")
            raw_data = collector.collect_and_process_data()
        
        self.logger.info(f"✅ Loaded {len(raw_data)} records")
        return raw_data
    
    def _train_models_fast(self, data):
        """Train models with optimized settings for speed"""
        models = COEPredictiveModels()
        
        # Use faster training settings
        feature_df = models.prepare_features(data)
        
        # Train only essential models for speed
        self.logger.info("🎯 Training essential models only...")
        models.train_all_models(feature_df)
        
        self.logger.info("✅ Models trained successfully")
        return models
    
    def _run_ultra_fast_optimization(self, data, models):
        """Run only the fastest optimization algorithms"""
        self.optimizer = OptimizedCOEQuotaOptimizer()
        self.optimizer.set_data_and_predictor(data, models)
        
        # Configure for ultra-fast optimization
        self.optimizer.max_prediction_time = 15  # Reduce timeout
        self.optimizer.cache_predictions = True
        
        self.logger.info("⚡ Running ultra-fast algorithms only...")
        
        # Run only the fastest algorithms
        algorithms = [
            ('simple_gradient', self.optimizer.optimize_simple_gradient),
            ('dual_annealing', self.optimizer.optimize_dual_annealing)
        ]
        
        start_time = time.time()
        
        for name, method in algorithms:
            try:
                self.logger.info(f"🏃 Running {name}...")
                result = method()
                self.results[name] = result
                
                if result.convergence_status == "success":
                    self.logger.info(f"✅ {name} completed in {result.execution_time:.2f}s")
                else:
                    self.logger.warning(f"⚠️ {name} completed with status: {result.convergence_status}")
                    
            except Exception as e:
                self.logger.error(f"❌ {name} failed: {e}")
                self.results[name] = self.optimizer._create_fallback_result(name, 0.0)
        
        total_time = time.time() - start_time
        self.logger.info(f"⏱️ Total optimization time: {total_time:.2f}s")
    
    def _display_results(self):
        """Display optimization results in a clean format"""
        print("\n" + "=" * 60)
        print("🚀 ULTRA-FAST OPTIMIZATION RESULTS")
        print("=" * 60)
        
        # Algorithm performance
        print("\n⚡ Algorithm Performance:")
        print("-" * 40)
        for algorithm, result in self.results.items():
            status_icon = "✅" if result.convergence_status == "success" else "⚠️"
            print(f"{status_icon} {algorithm.upper()}:")
            print(f"   ⏱️  Time: {result.execution_time:.2f}s")
            print(f"   🎯 Objective: {result.objective_value:.2f}")
            print(f"   📊 Status: {result.convergence_status}")
            print()
        
        # Find best result
        valid_results = {k: v for k, v in self.results.items() 
                        if v.convergence_status == "success" and v.objective_value < float('inf')}
        
        if valid_results:
            best_algorithm = min(valid_results.keys(), 
                               key=lambda k: valid_results[k].objective_value)
            best_result = valid_results[best_algorithm]
            
            print("🏆 BEST RESULT:")
            print("-" * 40)
            print(f"Algorithm: {best_algorithm.upper()}")
            print(f"Execution Time: {best_result.execution_time:.2f}s")
            print(f"Objective Value: {best_result.objective_value:.2f}")
            
            print("\n💡 Recommended Quota Adjustments:")
            print("-" * 40)
            for category, multiplier in best_result.optimal_quotas.items():
                change_pct = (multiplier - 1.0) * 100
                predicted_price = best_result.predicted_prices.get(category, 0)
                current_quota = self.optimizer.current_quotas.get(category, 1000)
                new_quota = current_quota * multiplier
                
                change_icon = "📈" if change_pct > 0 else "📉" if change_pct < 0 else "➡️"
                print(f"{change_icon} {category}:")
                print(f"   Current: {current_quota:,.0f}")
                print(f"   Recommended: {new_quota:,.0f}")
                print(f"   Change: {change_pct:+.1f}%")
                print(f"   Predicted Price: ${predicted_price:,.0f}")
                print()
        else:
            print("❌ No valid optimization results found")
        
        # Performance summary
        print("📊 PERFORMANCE SUMMARY:")
        print("-" * 40)
        total_time = sum(result.execution_time for result in self.results.values())
        successful_count = sum(1 for result in self.results.values() 
                             if result.convergence_status == "success")
        
        print(f"Total Time: {total_time:.2f}s")
        print(f"Successful Algorithms: {successful_count}/{len(self.results)}")
        print(f"Average Time: {total_time/len(self.results):.2f}s")
        
        if successful_count > 0:
            fastest_time = min(result.execution_time for result in self.results.values() 
                             if result.convergence_status == "success")
            print(f"Fastest Algorithm: {fastest_time:.2f}s")
        
        print("\n" + "=" * 60)

def main():
    """Main function to run the fast optimization demo"""
    demo = FastOptimizationDemo()
    success = demo.run_demo()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print("💡 For production use, consider using only 'simple_gradient' for fastest results.")
    else:
        print("\n❌ Demo failed. Check logs for details.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 