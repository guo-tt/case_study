#!/usr/bin/env python3
"""
Optimized COE Quota Optimization Module

This module implements fast and reliable optimization algorithms for COE quota
adjustments with improved performance and convergence.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
import logging
import yaml
from datetime import datetime, timedelta
from pathlib import Path
import time
from dataclasses import dataclass

# Optimization libraries
from scipy.optimize import minimize, differential_evolution, dual_annealing
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class OptimizationResult:
    """Data class to store optimization results"""
    optimal_quotas: Dict[str, float]
    objective_value: float
    predicted_prices: Dict[str, float]
    volatility_metrics: Dict[str, float]
    constraints_satisfied: bool
    algorithm_used: str
    execution_time: float
    iterations: int
    convergence_status: str

class OptimizedCOEQuotaOptimizer:
    """
    Optimized multi-objective optimization system for COE quota adjustments
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the optimized quota optimizer
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.historical_data = None
        self.price_predictor = None
        self._prediction_cache = {}  # Cache for predictions to avoid repeated calls
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Optimization parameters
        self.objectives = self.config['optimization']['objectives']
        self.constraints = self.config['optimization']['constraints']
        self.algorithm_config = self.config['optimization'].get('algorithm_config', {})
        
        # Performance optimization settings
        self.max_prediction_time = 30  # seconds
        self.cache_predictions = True
        self.fallback_to_simple = True
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            raise
    
    def set_data_and_predictor(self, historical_data: pd.DataFrame, price_predictor):
        """Set historical data and price predictor for optimization"""
        self.logger.info("Setting historical data and price predictor for optimizer.")
        self.historical_data = historical_data
        self.price_predictor = price_predictor
        self.current_quotas = self._extract_current_quotas()
        
        # Dynamically determine which categories can be optimized
        self.optimizable_categories = [
            cat for cat in ['Cat A', 'Cat B', 'Cat C', 'Cat D', 'Cat E']
            if cat in self.price_predictor.models and self.price_predictor.models[cat]
        ]
        
        if not self.optimizable_categories:
            raise ValueError("No valid models available for optimization.")
            
        self.logger.info(f"Optimizer will run for the following categories: {self.optimizable_categories}")
        
        # Clear prediction cache
        self._prediction_cache.clear()
    
    def _extract_current_quotas(self) -> Dict[str, float]:
        """Extract the most recent quota for each category"""
        current_quotas = {}
        for category in ['Cat A', 'Cat B', 'Cat C', 'Cat D', 'Cat E']:
            category_data = self.historical_data[self.historical_data['Category'] == category]
            if not category_data.empty and 'Quota' in category_data.columns:
                latest_quota = category_data.sort_values('Date')['Quota'].dropna().iloc[-1]
                current_quotas[category] = latest_quota if pd.notna(latest_quota) else 1000
            else:
                current_quotas[category] = 1000
        return current_quotas
    
    def _get_historical_average_price(self, category: str) -> float:
        """Get historical average price for a category as fallback"""
        category_data = self.historical_data[self.historical_data['Category'] == category]
        if not category_data.empty and 'COE_Price' in category_data.columns:
            avg_price = category_data['COE_Price'].dropna().mean()
            return avg_price if pd.notna(avg_price) and avg_price > 0 else 50000
        return 50000
    
    def _get_target_price(self, category: str) -> float:
        """Get target price for a category (historical average)"""
        return self._get_historical_average_price(category)
    
    def _calculate_quota_price_impact(self, category: str, quota_multiplier: float) -> float:
        """Calculate price impact of quota changes using simplified model"""
        # Simplified price-quota relationship: price ∝ 1/quota^0.5
        # This is a reasonable approximation for COE markets
        return 1.0 / (quota_multiplier ** 0.5)
    
    def predict_prices_with_quotas_fast(self, quota_adjustments: Dict[str, float]) -> Dict[str, float]:
        """
        Fast price prediction with caching and timeout protection
        """
        # Create cache key
        cache_key = tuple(sorted(quota_adjustments.items()))
        
        # Check cache first
        if self.cache_predictions and cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]
        
        predicted_prices = {}
        start_time = time.time()
        
        for category in self.optimizable_categories:
            if time.time() - start_time > self.max_prediction_time:
                self.logger.warning(f"Prediction timeout for {category}, using fallback")
                predicted_prices[category] = self._get_historical_average_price(category)
                continue
                
            if category in self.current_quotas:
                try:
                    # Use simplified prediction for speed
                    baseline_price = self._get_historical_average_price(category)
                    quota_impact = self._calculate_quota_price_impact(category, quota_adjustments[category])
                    predicted_price = baseline_price * quota_impact
                    predicted_prices[category] = max(1000, predicted_price)
                    
                except Exception as e:
                    self.logger.warning(f"Prediction failed for {category}: {e}")
                    predicted_prices[category] = self._get_historical_average_price(category)
            else:
                predicted_prices[category] = 50000
        
        # Cache the result
        if self.cache_predictions:
            self._prediction_cache[cache_key] = predicted_prices
        
        return predicted_prices
    
    def objective_price_stability(self, predicted_prices: Dict[str, float]) -> float:
        """Objective function for price stability"""
        target_prices = {cat: self._get_target_price(cat) for cat in predicted_prices.keys()}
        
        total_deviation = 0
        for category, price in predicted_prices.items():
            if category in target_prices and target_prices[category] > 0:
                deviation = abs(price - target_prices[category])
                normalized_deviation = deviation / target_prices[category]
                total_deviation += normalized_deviation ** 2
        
        return total_deviation
    
    def objective_revenue_generation(self, predicted_prices: Dict[str, float], quota_adjustments: Dict[str, float]) -> float:
        """Objective function for revenue generation"""
        total_revenue = 0
        for category, price in predicted_prices.items():
            if category in self.current_quotas and category in quota_adjustments:
                new_quota = self.current_quotas[category] * quota_adjustments[category]
                revenue = price * new_quota
                total_revenue += revenue
        
        return -total_revenue  # Negative because we're minimizing
    
    def objective_market_efficiency(self, predicted_prices: Dict[str, float]) -> float:
        """Objective function for market efficiency"""
        prices = list(predicted_prices.values())
        if len(prices) < 2:
            return 0
        
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        
        if mean_price > 0:
            return std_price / mean_price
        else:
            return float('inf')
    
    def fast_multi_objective_function(self, quota_adjustments_array: np.ndarray) -> float:
        """
        Fast multi-objective function with simplified prediction
        """
        # Convert array to dictionary
        quota_adjustments = {
            cat: quota_adjustments_array[i] 
            for i, cat in enumerate(self.optimizable_categories) 
            if i < len(quota_adjustments_array)
        }
        
        # Fast prediction
        try:
            predicted_prices = self.predict_prices_with_quotas_fast(quota_adjustments)
            
            # Check for invalid prices
            if not predicted_prices or any(p <= 0 for p in predicted_prices.values()):
                return float('inf')
                
        except Exception as e:
            self.logger.debug(f"Fast prediction failed: {e}")
            return float('inf')
        
        # Calculate objectives
        obj1 = self.objective_price_stability(predicted_prices)
        obj2 = self.objective_revenue_generation(predicted_prices, quota_adjustments)
        obj3 = self.objective_market_efficiency(predicted_prices)
        
        # Weighted combination
        weights = self.config['optimization']['objective_weights']
        
        combined_objective = (weights.get('price_stability', 0.5) * obj1 + 
                            weights.get('revenue_generation', 0.3) * obj2 + 
                            weights.get('market_efficiency', 0.2) * obj3)
        
        # Final check
        if not np.isfinite(combined_objective):
            return float('inf')
            
        return combined_objective
    
    def optimize_fast_genetic_algorithm(self) -> OptimizationResult:
        """
        Fast genetic algorithm optimization with improved parameters
        """
        self.logger.info("Starting fast genetic algorithm optimization...")
        
        start_time = datetime.now()
        
        # Define bounds
        min_change = 1.0 + self.constraints['min_quota_change']
        max_change = 1.0 + self.constraints['max_quota_change']
        bounds = [(min_change, max_change)] * len(self.optimizable_categories)
        
        # Optimized parameters for speed
        ga_config = self.algorithm_config.get('genetic_algorithm', {})
        max_iter = min(ga_config.get('max_iterations', 50), 50)  # Cap at 50 for speed
        pop_size = min(ga_config.get('population_size', 10), 15)  # Cap at 15 for speed
        
        try:
            result = differential_evolution(
                self.fast_multi_objective_function,
                bounds,
                maxiter=max_iter,
                popsize=pop_size,
                seed=42,
                workers=1,  # Single worker for stability
                polish=False,  # Disable polishing for speed
                updating='deferred',  # Deferred updating for better performance
                strategy='best1bin'  # Fast strategy
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Convert result to dictionary
            optimal_quotas = {cat: result.x[i] for i, cat in enumerate(self.optimizable_categories)}
            
            # Calculate predicted prices and metrics
            predicted_prices = self.predict_prices_with_quotas_fast(optimal_quotas)
            volatility_metrics = {'overall_volatility': self._calculate_price_volatility(predicted_prices)}
            
            return OptimizationResult(
                optimal_quotas=optimal_quotas,
                objective_value=result.fun,
                predicted_prices=predicted_prices,
                volatility_metrics=volatility_metrics,
                constraints_satisfied=self.check_constraints(optimal_quotas),
                algorithm_used="fast_genetic_algorithm",
                execution_time=execution_time,
                iterations=result.nit,
                convergence_status="success" if result.success else "max_iterations"
            )
            
        except Exception as e:
            self.logger.warning(f"Fast genetic algorithm failed: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            return self._create_fallback_result("fast_genetic_algorithm", execution_time)
    
    def optimize_dual_annealing(self) -> OptimizationResult:
        """
        Dual annealing optimization - fast and reliable
        """
        self.logger.info("Starting dual annealing optimization...")
        
        start_time = datetime.now()
        
        # Define bounds
        min_change = 1.0 + self.constraints['min_quota_change']
        max_change = 1.0 + self.constraints['max_quota_change']
        bounds = [(min_change, max_change)] * len(self.optimizable_categories)
        
        try:
            result = dual_annealing(
                self.fast_multi_objective_function,
                bounds,
                maxiter=100,
                seed=42,
                no_local_search=True  # Disable local search for speed
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Convert result to dictionary
            optimal_quotas = {cat: result.x[i] for i, cat in enumerate(self.optimizable_categories)}
            
            # Calculate predicted prices and metrics
            predicted_prices = self.predict_prices_with_quotas_fast(optimal_quotas)
            volatility_metrics = {'overall_volatility': self._calculate_price_volatility(predicted_prices)}
            
            return OptimizationResult(
                optimal_quotas=optimal_quotas,
                objective_value=result.fun,
                predicted_prices=predicted_prices,
                volatility_metrics=volatility_metrics,
                constraints_satisfied=self.check_constraints(optimal_quotas),
                algorithm_used="dual_annealing",
                execution_time=execution_time,
                iterations=result.nit,
                convergence_status="success"
            )
            
        except Exception as e:
            self.logger.warning(f"Dual annealing failed: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            return self._create_fallback_result("dual_annealing", execution_time)
    
    def optimize_simple_gradient(self) -> OptimizationResult:
        """
        Simple gradient-based optimization using L-BFGS-B
        """
        self.logger.info("Starting simple gradient optimization...")
        
        start_time = datetime.now()
        
        # Define bounds
        min_change = 1.0 + self.constraints['min_quota_change']
        max_change = 1.0 + self.constraints['max_quota_change']
        bounds = [(min_change, max_change)] * len(self.optimizable_categories)
        
        # Initial guess (no change)
        x0 = np.ones(len(self.optimizable_categories))
        
        try:
            result = minimize(
                self.fast_multi_objective_function,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'maxfun': 100}
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Convert result to dictionary
            optimal_quotas = {cat: result.x[i] for i, cat in enumerate(self.optimizable_categories)}
            
            # Calculate predicted prices and metrics
            predicted_prices = self.predict_prices_with_quotas_fast(optimal_quotas)
            volatility_metrics = {'overall_volatility': self._calculate_price_volatility(predicted_prices)}
            
            return OptimizationResult(
                optimal_quotas=optimal_quotas,
                objective_value=result.fun,
                predicted_prices=predicted_prices,
                volatility_metrics=volatility_metrics,
                constraints_satisfied=self.check_constraints(optimal_quotas),
                algorithm_used="simple_gradient",
                execution_time=execution_time,
                iterations=result.nit,
                convergence_status="success" if result.success else "max_iterations"
            )
            
        except Exception as e:
            self.logger.warning(f"Simple gradient optimization failed: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            return self._create_fallback_result("simple_gradient", execution_time)
    
    def check_constraints(self, quota_adjustments: Dict[str, float]) -> bool:
        """Check if quota adjustments satisfy policy constraints"""
        min_change = self.constraints['min_quota_change']
        max_change = self.constraints['max_quota_change']
        
        for category, adjustment in quota_adjustments.items():
            change_pct = adjustment - 1.0
            
            if change_pct < min_change or change_pct > max_change:
                return False
        
        return True
    
    def _calculate_price_volatility(self, predicted_prices: Dict[str, float]) -> float:
        """Calculate price volatility metric"""
        prices = list(predicted_prices.values())
        if len(prices) < 2:
            return 0
        
        mean_price = np.mean(prices)
        if mean_price > 0:
            return np.std(prices) / mean_price
        else:
            return 0
    
    def _create_fallback_result(self, algorithm: str, execution_time: float) -> OptimizationResult:
        """Create a fallback result when optimization fails"""
        # Return current quotas as fallback
        optimal_quotas = {cat: 1.0 for cat in self.optimizable_categories}
        predicted_prices = self.predict_prices_with_quotas_fast(optimal_quotas)
        volatility_metrics = {'overall_volatility': self._calculate_price_volatility(predicted_prices)}
        
        return OptimizationResult(
            optimal_quotas=optimal_quotas,
            objective_value=float('inf'),
            predicted_prices=predicted_prices,
            volatility_metrics=volatility_metrics,
            constraints_satisfied=True,
            algorithm_used=algorithm,
            execution_time=execution_time,
            iterations=0,
            convergence_status="fallback"
        )
    
    def run_fast_optimizations(self) -> Dict[str, OptimizationResult]:
        """
        Run fast optimization algorithms with timeout protection
        """
        self.logger.info("Running fast optimization algorithms...")
        
        results = {}
        max_total_time = 60  # Maximum total time for all optimizations
        start_time = time.time()
        
        # Run algorithms in order of speed
        algorithms = [
            ('simple_gradient', self.optimize_simple_gradient),
            ('dual_annealing', self.optimize_dual_annealing),
            ('fast_genetic_algorithm', self.optimize_fast_genetic_algorithm)
        ]
        
        for name, method in algorithms:
            if time.time() - start_time > max_total_time:
                self.logger.warning(f"Total time limit reached, skipping {name}")
                break
                
            try:
                self.logger.info(f"Running {name}...")
                result = method()
                results[name] = result
                
                if result.convergence_status == "success":
                    self.logger.info(f"{name} completed successfully with objective value: {result.objective_value:.4f}")
                else:
                    self.logger.warning(f"{name} completed with status: {result.convergence_status}")
                    
            except Exception as e:
                self.logger.error(f"Failed to run {name}: {e}")
                results[name] = self._create_fallback_result(name, 0.0)
        
        return results
    
    def generate_policy_recommendations(self, results: Dict[str, OptimizationResult]) -> Dict:
        """Generate policy recommendations based on optimization results"""
        self.logger.info("Generating policy recommendations...")
        
        # Find best result
        valid_results = {k: v for k, v in results.items() 
                        if v.constraints_satisfied and v.objective_value < float('inf')}
        
        if not valid_results:
            self.logger.warning("No valid optimization results found")
            return {"status": "No valid solutions found"}
        
        best_algorithm = min(valid_results.keys(), 
                           key=lambda k: valid_results[k].objective_value)
        best_result = valid_results[best_algorithm]
        
        # Calculate impacts
        impacts = {}
        for category, new_multiplier in best_result.optimal_quotas.items():
            if category in self.current_quotas:
                current_quota = self.current_quotas[category]
                new_quota = current_quota * new_multiplier
                change_pct = (new_multiplier - 1.0) * 100
                
                impacts[category] = {
                    'current_quota': current_quota,
                    'recommended_quota': new_quota,
                    'change_percentage': change_pct,
                    'predicted_price': best_result.predicted_prices.get(category, 0)
                }
        
        # Generate summary
        total_revenue_change = sum(
            impact['predicted_price'] * impact['recommended_quota'] 
            for impact in impacts.values()
        )
        
        policy_summary = f"Best algorithm: {best_algorithm}. "
        policy_summary += f"Total revenue impact: ${total_revenue_change:,.0f}. "
        policy_summary += f"Execution time: {best_result.execution_time:.2f}s"
        
        return {
            'status': 'success',
            'best_algorithm': best_algorithm,
            'objective_value': best_result.objective_value,
            'execution_time': best_result.execution_time,
            'quota_adjustments': impacts,
            'policy_summary': policy_summary,
            'all_results': {name: {
                'objective_value': result.objective_value,
                'execution_time': result.execution_time,
                'convergence_status': result.convergence_status
            } for name, result in results.items()}
        }

if __name__ == "__main__":
    # Example usage
    from src.data_collection.coe_data_collector import COEDataCollector
    from src.models.coe_predictive_models import COEPredictiveModels
    
    # Collect data and train models
    collector = COEDataCollector()
    raw_data = collector.collect_and_process_data()
    
    models = COEPredictiveModels()
    feature_df = models.prepare_features(raw_data)
    models.train_all_models(feature_df)
    
    # Run fast optimization
    optimizer = OptimizedCOEQuotaOptimizer()
    optimizer.set_data_and_predictor(raw_data, models)
    
    results = optimizer.run_fast_optimizations()
    recommendations = optimizer.generate_policy_recommendations(results)
    
    print("Fast Optimization Results:")
    for algorithm, result in results.items():
        print(f"{algorithm}: {result.objective_value:.4f} ({result.execution_time:.2f}s)")
    
    print("\nPolicy Recommendations:")
    print(recommendations['policy_summary']) 