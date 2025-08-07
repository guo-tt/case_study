"""
COE Quota Optimization Module

This module implements optimization algorithms to determine optimal COE quota
adjustments for price stabilization and policy objectives.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
import logging
import yaml
from datetime import datetime, timedelta
from pathlib import Path

# Optimization libraries
import cvxpy as cp
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Custom imports
from dataclasses import dataclass

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

class COEQuotaOptimizer:
    """
    Multi-objective optimization system for COE quota adjustments
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the quota optimizer
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.historical_data = None
        self.price_predictor = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Optimization parameters
        self.objectives = self.config['optimization']['objectives']
        self.constraints = self.config['optimization']['constraints']
        self.algorithm_config = self.config['optimization'].get('algorithm_config', {})
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            raise
    
    def set_data_and_predictor(self, historical_data: pd.DataFrame, price_predictor):
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
    
    def _extract_current_quotas(self) -> Dict[str, float]:
        """Extract the most recent quota for each category from the wide-format data."""
        current_quotas = {}
        for category in ['Cat A', 'Cat B', 'Cat C', 'Cat D', 'Cat E']:
            category_data = self.historical_data[self.historical_data['Category'] == category]
            if not category_data.empty and 'Quota' in category_data.columns:
                # Get the last known quota for the category
                latest_quota = category_data.sort_values('Date')['Quota'].dropna().iloc[-1]
                current_quotas[category] = latest_quota if pd.notna(latest_quota) else 1000
            else:
                current_quotas[category] = 1000  # Default if no data or Quota column is missing
        return current_quotas
    
    def predict_prices_with_quotas(self, quota_adjustments: Dict[str, float]) -> Dict[str, float]:
        self.logger.debug(f"Entering predict_prices_with_quotas with adjustments: {quota_adjustments}")
        predicted_prices = {}
        
        # Circuit breaker for repeated failures
        if hasattr(self, '_prediction_failures') and self._prediction_failures > 10:
            self.logger.warning("Circuit breaker active: Using fallback prices due to repeated failures.")
            for category in quota_adjustments.keys():
                predicted_prices[category] = self._get_historical_average_price(category)
            return predicted_prices
        
        if not hasattr(self, '_prediction_failures'):
            self._prediction_failures = 0

        for category in quota_adjustments.keys():
            if category in self.current_quotas:
                try:
                    self.logger.debug(f"Predicting for {category}...")
                    baseline_pred = self.price_predictor.ensemble_predict(category, periods=1)
                    
                    if baseline_pred is not None and len(baseline_pred) > 0 and baseline_pred[0] > 0:
                        quota_impact = self._calculate_quota_price_impact(category, quota_adjustments[category])
                        predicted_price = baseline_pred[0] * quota_impact
                        predicted_prices[category] = max(1000, predicted_price)
                        self.logger.debug(f"Prediction for {category} successful: ${predicted_price:,.2f}")
                        self._prediction_failures = max(0, self._prediction_failures - 1) # Decrease counter on success
                    else:
                        self.logger.warning(f"Baseline prediction for {category} failed or returned invalid value. Using fallback.")
                        predicted_prices[category] = self._get_historical_average_price(category)
                        self._prediction_failures += 1
                except Exception as e:
                    self.logger.error(f"Exception during prediction for {category}: {e}", exc_info=True)
                    predicted_prices[category] = self._get_historical_average_price(category)
                    self._prediction_failures += 1
            else:
                self.logger.warning(f"Category {category} not in current quotas. Using fallback price.")
                predicted_prices[category] = 50000
        
        self.logger.debug(f"Exiting predict_prices_with_quotas. Predicted prices: {predicted_prices}")
        return predicted_prices
    
    def _calculate_quota_price_impact(self, category: str, quota_multiplier: float) -> float:
        """
        Calculate the impact of quota changes on prices
        
        Args:
            category: COE category
            quota_multiplier: Quota change multiplier (1.0 = no change, 1.1 = 10% increase)
            
        Returns:
            Price impact multiplier
        """
        # This is a simplified model - should be calibrated with historical data
        # Basic assumption: 10% quota increase leads to ~8% price decrease
        elasticity = -0.8  # Price elasticity with respect to quota
        
        # Calculate price impact
        quota_change_pct = quota_multiplier - 1.0
        price_impact = 1.0 + (elasticity * quota_change_pct)
        
        # Ensure price impact is positive
        return max(0.1, price_impact)
    
    def _get_recent_average_price(self, category: str, months: int = 12) -> float:
        """
        Get the average 'Quota Premium' price for a category over the last few months.
        """
        if self.historical_data is None:
            return 50000 # Default fallback

        category_data = self.historical_data[
            (self.historical_data['Category'] == category) &
            (self.historical_data['Metric_Type'] == 'Quota Premium')
        ].copy()

        if category_data.empty:
            return 50000 # Default fallback

        # Ensure 'Date' is datetime
        category_data['Date'] = pd.to_datetime(category_data['Date'])
        
        # Get data from the last 'months'
        last_date = category_data['Date'].max()
        cutoff_date = last_date - pd.DateOffset(months=months)
        recent_data = category_data[category_data['Date'] >= cutoff_date]

        if not recent_data['Value'].dropna().empty:
            return recent_data['Value'].mean()
        elif not category_data['Value'].dropna().empty:
            # Fallback to overall average if no recent data
            return category_data['Value'].mean()
        
        return 50000 # Ultimate fallback

    def _get_historical_average_price(self, category: str) -> float:
        """Get historical average price for a category from the long-format data."""
        # Use the more robust recent average calculation
        return self._get_recent_average_price(category, months=24)

    def _get_target_price(self, category: str) -> float:
        """Get target price for a category, using the recent average."""
        # Use the new helper function, focusing on the last 12 months for the target
        return self._get_recent_average_price(category, months=12)
    
    def objective_price_stability(self, predicted_prices: Dict[str, float]) -> float:
        """
        Objective function for price stability (takes predicted_prices as input).
        Minimizes price volatility across categories.
        """
        target_prices = {cat: self._get_target_price(cat) for cat in predicted_prices.keys()}
        
        total_deviation = 0
        for category, price in predicted_prices.items():
            if category in target_prices and target_prices[category] > 0:
                deviation = abs(price - target_prices[category])
                normalized_deviation = deviation / target_prices[category]
                total_deviation += normalized_deviation ** 2
        
        return total_deviation
    
    def objective_revenue_generation(self, predicted_prices: Dict[str, float], quota_adjustments: Dict[str, float]) -> float:
        """
        Objective function for revenue generation (takes predicted_prices as input).
        Maximizes total government revenue from COE premiums.
        """
        total_revenue = 0
        for category, price in predicted_prices.items():
            if category in self.current_quotas and category in quota_adjustments:
                new_quota = self.current_quotas[category] * quota_adjustments[category]
                revenue = price * new_quota
                total_revenue += revenue
        
        return -total_revenue  # Negative because we're minimizing
    
    def objective_market_efficiency(self, predicted_prices: Dict[str, float]) -> float:
        """
        Objective function for market efficiency (takes predicted_prices as input).
        Balances supply and demand across categories.
        """
        prices = list(predicted_prices.values())
        if len(prices) < 2:
            return 0
        
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        
        if mean_price > 0:
            return std_price / mean_price
        else:
            return float('inf')
    
    def multi_objective_function(self, quota_adjustments_array: np.ndarray) -> float:
        """
        Combined multi-objective function with a single prediction call.
        Returns a massive penalty if prediction fails, to avoid infinite loops.
        """
        self.logger.debug("Entering multi_objective_function...")
        # Convert array to dictionary for optimizable categories only
        quota_adjustments = {
            cat: quota_adjustments_array[i] 
            for i, cat in enumerate(self.optimizable_categories) 
            if i < len(quota_adjustments_array)
        }
        
        # --- Single Prediction Call ---
        try:
            predicted_prices = self.predict_prices_with_quotas(quota_adjustments)
            # Check for any invalid (zero or negative) prices
            if not predicted_prices or any(p <= 0 for p in predicted_prices.values()):
                self.logger.warning("Prediction returned invalid prices, returning penalty.")
                return float('inf')
        except Exception as e:
            self.logger.error(f"Prediction failed during optimization: {e}", exc_info=True)
            return float('inf')  # Return massive penalty
        
        # --- Calculate objectives using the single prediction result ---
        obj1 = self.objective_price_stability(predicted_prices)
        obj2 = self.objective_revenue_generation(predicted_prices, quota_adjustments)
        obj3 = self.objective_market_efficiency(predicted_prices)
        
        # Weighted combination
        weights = self.config['optimization']['objective_weights']
        
        combined_objective = (weights.get('price_stability', 0.5) * obj1 + 
                            weights.get('revenue_generation', 0.3) * obj2 + 
                            weights.get('market_efficiency', 0.2) * obj3)
        
        # Final sanity check
        if not np.isfinite(combined_objective):
            self.logger.warning(f"Combined objective is not finite ({combined_objective}), returning penalty.")
            return float('inf')
            
        return combined_objective
    
    def check_constraints(self, quota_adjustments: Dict[str, float]) -> bool:
        """
        Check if quota adjustments satisfy policy constraints
        
        Args:
            quota_adjustments: Quota adjustments by category
            
        Returns:
            True if all constraints are satisfied
        """
        min_change = self.constraints['min_quota_change']
        max_change = self.constraints['max_quota_change']
        
        for category, adjustment in quota_adjustments.items():
            change_pct = adjustment - 1.0
            
            if change_pct < min_change or change_pct > max_change:
                return False
        
        # Check volatility constraint
        predicted_prices = self.predict_prices_with_quotas(quota_adjustments)
        volatility = self._calculate_price_volatility(predicted_prices)
        
        if volatility > self.constraints['price_volatility_threshold']:
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
    
    def optimize_genetic_algorithm(self) -> OptimizationResult:
        """
        Optimize quotas using genetic algorithm
        
        Returns:
            Optimization result
        """
        self.logger.info("Starting genetic algorithm optimization...")
        
        start_time = datetime.now()
        
        # Define bounds for optimizable categories only
        min_change = 1.0 + self.constraints['min_quota_change']
        max_change = 1.0 + self.constraints['max_quota_change']
        bounds = [(min_change, max_change)] * len(self.optimizable_categories)
        
        # Get config for genetic algorithm
        ga_config = self.algorithm_config.get('genetic_algorithm', {})
        max_iter = ga_config.get('max_iterations', 50)
        pop_size = ga_config.get('population_size', 10)
        
        # Optimize with timeout protection
        try:
            result = differential_evolution(
                self.multi_objective_function,
                bounds,
                maxiter=max_iter,
                popsize=pop_size,
                seed=42,
                workers=-1,       # Use all available cores
                polish=False      # Disable polishing for speed
            )
        except Exception as e:
            self.logger.warning(f"Genetic algorithm failed: {e}")
            return self._create_fallback_result("genetic_algorithm", 0)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Convert result to dictionary
        optimal_quotas = {cat: result.x[i] for i, cat in enumerate(self.optimizable_categories)}
        
        # Calculate predicted prices and metrics
        predicted_prices = self.predict_prices_with_quotas(optimal_quotas)
        volatility_metrics = {'overall_volatility': self._calculate_price_volatility(predicted_prices)}
        
        return OptimizationResult(
            optimal_quotas=optimal_quotas,
            objective_value=result.fun,
            predicted_prices=predicted_prices,
            volatility_metrics=volatility_metrics,
            constraints_satisfied=self.check_constraints(optimal_quotas),
            algorithm_used="genetic_algorithm",
            execution_time=execution_time,
            iterations=result.nit
        )
    
    def optimize_linear_programming(self) -> OptimizationResult:
        """
        Optimize quotas using linear programming (simplified version)
        
        Returns:
            Optimization result
        """
        self.logger.info("Starting linear programming optimization...")
        
        start_time = datetime.now()
        
        try:
            # Define variables for quota adjustments
            categories = ['Cat A', 'Cat B', 'Cat C', 'Cat D', 'Cat E']
            n_categories = len(categories)
            
            # Variables: quota multipliers for each category
            quota_vars = cp.Variable(n_categories, pos=True)
            
            # Constraints
            constraints = []
            
            # Quota change bounds
            min_multiplier = 1.0 + self.constraints['min_quota_change']
            max_multiplier = 1.0 + self.constraints['max_quota_change']
            
            constraints.append(quota_vars >= min_multiplier)
            constraints.append(quota_vars <= max_multiplier)
            
            # Simplified objective: minimize deviation from current quotas
            # while staying within bounds (stability objective)
            objective = cp.Minimize(cp.sum_squares(quota_vars - 1.0))
            
            # Solve problem
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.ECOS)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if problem.status == cp.OPTIMAL:
                # Convert result to dictionary
                optimal_quotas = {cat: quota_vars.value[i] for i, cat in enumerate(categories)}
                
                # Calculate predicted prices and metrics
                predicted_prices = self.predict_prices_with_quotas(optimal_quotas)
                volatility_metrics = {'overall_volatility': self._calculate_price_volatility(predicted_prices)}
                
                return OptimizationResult(
                    optimal_quotas=optimal_quotas,
                    objective_value=problem.value,
                    predicted_prices=predicted_prices,
                    volatility_metrics=volatility_metrics,
                    constraints_satisfied=True,
                    algorithm_used="linear_programming",
                    execution_time=execution_time,
                    iterations=1
                )
            else:
                self.logger.error(f"Linear programming failed with status: {problem.status}")
                return self._create_fallback_result("linear_programming", execution_time)
                
        except Exception as e:
            self.logger.error(f"Linear programming optimization failed: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            return self._create_fallback_result("linear_programming", execution_time)
    
    def optimize_particle_swarm(self) -> OptimizationResult:
        """
        Optimize quotas using particle swarm optimization
        
        Returns:
            Optimization result
        """
        self.logger.info("Starting particle swarm optimization...")
        
        start_time = datetime.now()
        
        # PSO parameters
        n_particles = 20
        n_dimensions = len(self.optimizable_categories)
        n_iterations = 100
        
        # PSO parameters from config
        pso_config = self.algorithm_config.get('particle_swarm', {})
        n_particles = pso_config.get('particles', n_particles)
        n_iterations = pso_config.get('iterations', n_iterations)
        max_execution_time = pso_config.get('timeout_seconds', 60)
        
        # Initialize particles
        min_val = 1.0 + self.constraints['min_quota_change']
        max_val = 1.0 + self.constraints['max_quota_change']
        
        particles = np.random.uniform(min_val, max_val, (n_particles, n_dimensions))
        velocities = np.random.uniform(-0.1, 0.1, (n_particles, n_dimensions))
        
        # PSO parameters
        w = 0.7
        c1 = 1.5
        c2 = 1.5
        
        # Initialize best positions
        personal_best = particles.copy()
        personal_best_scores = np.array([self.multi_objective_function(p) for p in particles])
        
        global_best_idx = np.argmin(personal_best_scores)
        global_best = personal_best[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        
        # PSO iterations with timeout
        for iteration in range(n_iterations):
            elapsed_time = (datetime.now() - start_time).total_seconds()
            if elapsed_time > max_execution_time:
                self.logger.warning(f"PSO optimization timed out after {elapsed_time:.1f}s")
                break

            for i in range(n_particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = (w * velocities[i] + 
                               c1 * r1 * (personal_best[i] - particles[i]) + 
                               c2 * r2 * (global_best - particles[i]))
                
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], min_val, max_val)
                
                score = self.multi_objective_function(particles[i])
                
                if score < personal_best_scores[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_scores[i] = score
                    
                    if score < global_best_score:
                        global_best = particles[i].copy()
                        global_best_score = score
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Convert result to dictionary
        optimal_quotas = {cat: global_best[i] for i, cat in enumerate(self.optimizable_categories)}
        
        # Calculate predicted prices and metrics
        predicted_prices = self.predict_prices_with_quotas(optimal_quotas)
        volatility_metrics = {'overall_volatility': self._calculate_price_volatility(predicted_prices)}
        
        return OptimizationResult(
            optimal_quotas=optimal_quotas,
            objective_value=global_best_score,
            predicted_prices=predicted_prices,
            volatility_metrics=volatility_metrics,
            constraints_satisfied=self.check_constraints(optimal_quotas),
            algorithm_used="particle_swarm",
            execution_time=execution_time,
            iterations=n_iterations
        )
    
    def _create_fallback_result(self, algorithm: str, execution_time: float) -> OptimizationResult:
        """Create a fallback result when optimization fails"""
        categories = ['Cat A', 'Cat B', 'Cat C', 'Cat D', 'Cat E']
        
        # Return current quotas as fallback
        optimal_quotas = {cat: 1.0 for cat in categories}  # No change
        predicted_prices = self.predict_prices_with_quotas(optimal_quotas)
        volatility_metrics = {'overall_volatility': self._calculate_price_volatility(predicted_prices)}
        
        return OptimizationResult(
            optimal_quotas=optimal_quotas,
            objective_value=float('inf'),
            predicted_prices=predicted_prices,
            volatility_metrics=volatility_metrics,
            constraints_satisfied=True,
            algorithm_used=algorithm,
            execution_time=execution_time,
            iterations=0
        )
    
    def run_all_optimizations(self) -> Dict[str, OptimizationResult]:
        """
        Run all optimization algorithms and compare results
        
        Returns:
            Dictionary of results from all algorithms
        """
        self.logger.info("Running all optimization algorithms...")
        
        results = {}
        
        # Run each algorithm
        algorithms = [
            ('genetic_algorithm', self.optimize_genetic_algorithm),
            ('linear_programming', self.optimize_linear_programming),
            ('particle_swarm', self.optimize_particle_swarm)
        ]
        
        for name, method in algorithms:
            try:
                self.logger.info(f"Running {name}...")
                result = method()
                results[name] = result
                self.logger.info(f"{name} completed with objective value: {result.objective_value:.4f}")
            except Exception as e:
                self.logger.error(f"Failed to run {name}: {e}")
                results[name] = self._create_fallback_result(name, 0.0)
        
        return results
    
    def generate_policy_recommendations(self, results: Dict[str, OptimizationResult]) -> Dict:
        """
        Generate policy recommendations based on optimization results
        
        Args:
            results: Results from all optimization algorithms
            
        Returns:
            Policy recommendations
        """
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
        
        recommendations = {
            'status': 'success',
            'best_algorithm': best_algorithm,
            'objective_value': best_result.objective_value,
            'execution_time': best_result.execution_time,
            'quota_adjustments': impacts,
            'overall_volatility': best_result.volatility_metrics.get('overall_volatility', 0),
            'constraints_satisfied': best_result.constraints_satisfied,
            'policy_summary': self._generate_policy_summary(impacts),
            'comparison_results': {k: v.objective_value for k, v in results.items()}
        }
        
        return recommendations
    
    def _generate_policy_summary(self, impacts: Dict) -> str:
        """Generate a human-readable policy summary"""
        increases = []
        decreases = []
        no_changes = []
        
        for category, impact in impacts.items():
            change_pct = impact['change_percentage']
            if change_pct > 1:
                increases.append(f"{category} (+{change_pct:.1f}%)")
            elif change_pct < -1:
                decreases.append(f"{category} ({change_pct:.1f}%)")
            else:
                no_changes.append(category)
        
        summary_parts = []
        if increases:
            summary_parts.append(f"Increase quotas for: {', '.join(increases)}")
        if decreases:
            summary_parts.append(f"Decrease quotas for: {', '.join(decreases)}")
        if no_changes:
            summary_parts.append(f"Maintain current levels for: {', '.join(no_changes)}")
        
        return "; ".join(summary_parts) if summary_parts else "No significant changes recommended"

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
    
    # Run optimization
    optimizer = COEQuotaOptimizer()
    optimizer.set_data_and_predictor(raw_data, models)
    
    results = optimizer.run_all_optimizations()
    recommendations = optimizer.generate_policy_recommendations(results)
    
    print("Optimization Results:")
    for algorithm, result in results.items():
        print(f"{algorithm}: {result.objective_value:.4f}")
    
    print("\nPolicy Recommendations:")
    print(recommendations['policy_summary']) 