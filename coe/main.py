#!/usr/bin/env python3
"""
Singapore COE Price Prediction and Quota Optimization System
Main Runner Script

This script demonstrates the complete workflow:
1. Data collection from Singapore government APIs
2. Feature engineering and data preparation
3. Model training and evaluation
4. Quota optimization and policy recommendations
5. Results visualization and reporting
"""

import logging
import argparse
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Added for sensitivity analysis

# Import our modules
from src.data_collection.coe_data_collector import COEDataCollector
from src.models.coe_predictive_models import COEPredictiveModels
from src.optimization.quota_optimizer import COEQuotaOptimizer
from src.optimization.optimized_quota_optimizer import OptimizedCOEQuotaOptimizer

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/coe_system.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)

def collect_data():
    """Step 1: Collect COE data"""
    print("\n" + "="*60)
    print("STEP 1: DATA COLLECTION")
    print("="*60)
    
    collector = COEDataCollector()
    data = collector.collect_and_process_data()
    
    print(f"✅ Successfully collected {len(data)} records")
    print(f"📅 Date range: {data['Date'].min()} to {data['Date'].max()}")
    print(f"🏷️ Categories: {data['Category'].unique().tolist()}")
    return data

def train_models(data):
    """Step 2: Train predictive models"""
    print("\n" + "="*60)
    print("STEP 2: MODEL TRAINING")
    print("="*60)
    
    models = COEPredictiveModels()
    
    # Prepare features
    print("🔧 Preparing features...")
    feature_df = models.prepare_features(data)
    print(f"✅ Created {len(models.feature_columns)} features")
    
    # Train models
    print("🎯 Training predictive models...")
    trained_models = models.train_all_models(feature_df)
    
    print(f"✅ Successfully trained models for {len(trained_models)} categories")
    for category, category_models in trained_models.items():
        print(f"   {category}: {list(category_models.keys())}")
    
    return models, feature_df

def evaluate_models(models):
    """Step 3: Evaluate model performance"""
    print("\n" + "="*60)
    print("STEP 3: MODEL EVALUATION")
    print("="*60)
    
    evaluation_results = models.evaluate_models()
    
    # Create performance summary
    performance_summary = []
    for category, category_results in evaluation_results.items():
        for model_type, metrics in category_results.items():
            performance_summary.append({
                'Category': category,
                'Model': model_type.upper(),
                'MAE': metrics.get('mae', 0),
                'RMSE': metrics.get('rmse', 0),
                'R²': metrics.get('r2', 0)
            })
    
    if performance_summary:
        perf_df = pd.DataFrame(performance_summary)
        print("📊 Model Performance Summary:")
        print(perf_df.round(3).to_string(index=False))
        
        # Find best models
        print("\n🏆 Best Performing Models by Category:")
        best_models = perf_df.loc[perf_df.groupby('Category')['R²'].idxmax()]
        for _, row in best_models.iterrows():
            print(f"   {row['Category']}: {row['Model']} (R² = {row['R²']:.3f})")
    
    return evaluation_results

def generate_predictions(models, horizon=12):
    """Step 4: Generate price predictions"""
    print("\n" + "="*60)
    print("STEP 4: PRICE PREDICTIONS")
    print("="*60)
    
    all_predictions_df = pd.DataFrame()
    categories = ['Cat A', 'Cat B', 'Cat C', 'Cat D', 'Cat E']
    
    print(f"🔮 Generating {horizon}-month forecasts...")
    
    for category in categories:
        try:
            # Get predictions based on the configured strategy
            predictions = models.get_prediction(category, periods=horizon)
            
            if len(predictions) > 0:
                # Create a dataframe for the category's predictions
                pred_df = pd.DataFrame({
                    'Month_Ahead': range(1, horizon + 1),
                    'Category': category,
                    'Predicted_Price': predictions
                })
                all_predictions_df = pd.concat([all_predictions_df, pred_df])

                print(f"\n  Predictions for {category}:")
                for i, price in enumerate(predictions):
                    print(f"    Month {i+1}: ${price:,.0f}")
            else:
                 print(f"   ❌ {category}: No predictions returned.")

        except Exception as e:
            print(f"   ❌ {category}: Failed to generate predictions ({e})")
    
    return all_predictions_df

def simulate_policy_changes(data, models, increase_pct, decrease_pct):
    """Step 5: Simulate user-defined policy changes"""
    if increase_pct is None and decrease_pct is None:
        return

    print("\n" + "="*60)
    print("STEP 5: POLICY SIMULATION")
    print("="*60)

    optimizer = COEQuotaOptimizer()
    optimizer.set_data_and_predictor(data, models)

    # Baseline prediction (no change)
    print("  Calculating baseline prices (no quota change)...")
    baseline_adjustments = {cat: 1.0 for cat in optimizer.optimizable_categories}
    baseline_prices = optimizer.predict_prices_with_quotas(baseline_adjustments)
    print("  Baseline Predicted Prices:")
    for category, price in baseline_prices.items():
        print(f"    {category}: ${price:,.0f}")

    scenarios = {}
    if decrease_pct is not None:
        scenarios[f"Decrease Quota by {decrease_pct}%"] = 1.0 - (decrease_pct / 100.0)
    if increase_pct is not None:
        scenarios[f"Increase Quota by {increase_pct}%"] = 1.0 + (increase_pct / 100.0)
    
    for scenario_name, multiplier in scenarios.items():
        print(f"\n  Simulating: {scenario_name}...")
        
        adjustments = {cat: multiplier for cat in optimizer.optimizable_categories}
        
        predicted_prices = optimizer.predict_prices_with_quotas(adjustments)

        print("  Predicted Price Impact:")
        for category, price in predicted_prices.items():
            baseline_price = baseline_prices.get(category, 0)
            if baseline_price > 0:
                price_change_pct = ((price - baseline_price) / baseline_price) * 100
                print(f"    {category}: ${price:,.0f} (Change: {price_change_pct:+.1f}%)")
            else:
                print(f"    {category}: ${price:,.0f}")

def optimize_quotas(data, models):
    """Step 6: Optimize quota allocations"""
    print("\n" + "="*60)
    print("STEP 6: QUOTA OPTIMIZATION")
    print("="*60)
    
    optimizer = OptimizedCOEQuotaOptimizer()
    optimizer.set_data_and_predictor(data, models)
    
    print("⚡ Running optimization algorithms...")
    
    # Ultra-fast optimization
    result = optimizer.optimize_simple_gradient()  # 0.14s
    print(f"Time: {result.execution_time:.3f}s")
    print(f"Status: {result.convergence_status}")
    
    # Generate policy recommendations
    # Create a dictionary with the single result
    results_dict = {'simple_gradient': result}
    recommendations = optimizer.generate_policy_recommendations(results_dict)
    
    if recommendations['status'] == 'success':
        print("✅ Optimization completed successfully!")
        print(f"🎯 Best Algorithm: {recommendations['best_algorithm']}")
        print(f"📈 Objective Value: {recommendations['objective_value']:.4f}")
        print(f"⏱️ Execution Time: {recommendations['execution_time']:.2f}s")
        
        print(f"\n📋 Policy Recommendations:")
        print(f"   {recommendations['policy_summary']}")
        
        print(f"\n💡 Detailed Quota Adjustments:")
        for category, impact in recommendations['quota_adjustments'].items():
            change = impact['change_percentage']
            price = impact['predicted_price']
            print(f"   {category}: {change:+.1f}% → ${price:,.0f}")
    
    else:
        print("❌ Optimization failed - no valid solutions found")
        recommendations = None
    
    return result, recommendations

def generate_report(data, predictions, recommendations, output_file='coe_report.json'):
    """Step 7: Generate comprehensive report"""
    print("\n" + "="*60)
    print("STEP 7: REPORT GENERATION")
    print("="*60)
    
    # Create comprehensive report
    report = {
        'timestamp': datetime.now().isoformat(),
        'data_summary': {
            'total_records': len(data),
            'date_range': [data['Date'].min().isoformat(), data['Date'].max().isoformat()],
            'categories': data['Category'].unique().tolist()
        },
        'predictions': predictions.to_dict('records') if len(predictions) > 0 else [],
        'optimization_results': recommendations if recommendations else {},
        'system_status': 'operational'
    }
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"📄 Report saved to: {output_file}")
    
    # Print summary statistics
    print(f"\n📊 System Summary:")
    print(f"   Data Points: {len(data):,}")
    print(f"   Prediction Horizon: {len(predictions)} categories")
    print(f"   Optimization Status: {'✅ Success' if recommendations else '❌ Failed'}")
    
    return report

def create_visualizations(data, predictions):
    """Step 8: Create summary visualizations"""
    print("\n" + "="*60)
    print("STEP 8: VISUALIZATION")
    print("="*60)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Singapore COE Analysis Summary', fontsize=16, fontweight='bold')
    
    # 1. Historical price trends
    premium_data = data[data['Metric_Type'] == 'Quota Premium']
    recent_data = premium_data[premium_data['Date'] >= 
                              premium_data['Date'].max() - pd.DateOffset(years=3)]
    
    for category in ['Cat A', 'Cat B', 'Cat E']:
        cat_data = recent_data[recent_data['Category'] == category]
        if len(cat_data) > 0:
            axes[0, 0].plot(cat_data['Date'], cat_data['Value'], 
                           marker='o', markersize=3, label=category)
    
    axes[0, 0].set_title('COE Premium Trends (Last 3 Years)')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Premium (SGD)')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Price predictions
    if not predictions.empty:
        avg_predictions = predictions.groupby('Category')['Predicted_Price'].mean().reset_index()
        axes[0, 1].bar(avg_predictions['Category'], avg_predictions['Predicted_Price'])
        title_horizon = len(predictions['Month_Ahead'].unique())
        axes[0, 1].set_title(f'Price Predictions ({title_horizon}-Month Average)')
        axes[0, 1].set_ylabel('Predicted Premium (SGD)')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Volatility analysis
    if not predictions.empty:
        volatility = predictions.groupby('Category')['Predicted_Price'].std().reset_index()
        volatility = volatility.rename(columns={'Predicted_Price': 'Volatility'})
        axes[1, 0].bar(volatility['Category'], volatility['Volatility'])
        axes[1, 0].set_title('Price Volatility by Category')
        axes[1, 0].set_ylabel('Standard Deviation (SGD)')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Data coverage
    coverage = data.groupby(['Category', 'Metric_Type']).size().reset_index(name='Count')
    coverage_pivot = coverage.pivot(index='Category', columns='Metric_Type', values='Count').fillna(0)
    
    im = axes[1, 1].imshow(coverage_pivot.values, cmap='Blues', aspect='auto')
    axes[1, 1].set_xticks(range(len(coverage_pivot.columns)))
    axes[1, 1].set_xticklabels(coverage_pivot.columns, rotation=45)
    axes[1, 1].set_yticks(range(len(coverage_pivot.index)))
    axes[1, 1].set_yticklabels(coverage_pivot.index)
    axes[1, 1].set_title('Data Coverage Heatmap')
    
    # Add colorbar
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('coe_analysis_summary.png', dpi=300, bbox_inches='tight')
    print("📊 Visualization saved to: coe_analysis_summary.png")
    
    return fig

def create_quota_sensitivity_visualization(data, models):
    """Creates a visualization showing how price changes with quota adjustments."""
    print("\n" + "="*60)
    print("STEP 9: QUOTA SENSITIVITY ANALYSIS")
    print("="*60)
    
    optimizer = COEQuotaOptimizer()
    optimizer.set_data_and_predictor(data, models)
    
    # Define a range of quota changes to simulate
    quota_changes = np.linspace(-0.5, 0.5, 21) # From -50% to +50%
    results = []
    
    print("🔬 Running sensitivity analysis...")
    
    # Get baseline prices
    baseline_prices = optimizer.predict_prices_with_quotas({cat: 1.0 for cat in optimizer.optimizable_categories})

    for change in quota_changes:
        multiplier = 1.0 + change
        adjustments = {cat: multiplier for cat in optimizer.optimizable_categories}
        predicted_prices = optimizer.predict_prices_with_quotas(adjustments)
        
        for category, price in predicted_prices.items():
            results.append({
                'quota_change_pct': change * 100,
                'category': category,
                'predicted_price': price,
                'baseline_price': baseline_prices.get(category, price)
            })
            
    results_df = pd.DataFrame(results)
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=results_df, x='quota_change_pct', y='predicted_price', hue='category', marker='o')
    
    plt.title('COE Price Sensitivity to Quota Adjustments', fontsize=16)
    plt.xlabel('Quota Change (%)')
    plt.ylabel('Predicted COE Premium (SGD)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='r', linestyle='--', label='No Change')
    plt.legend(title='COE Category')
    
    # Save the plot
    output_path = 'coe_quota_sensitivity.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"📈 Sensitivity analysis chart saved to: {output_path}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Singapore COE Analysis System')
    parser.add_argument('--skip-data', action='store_true', help='Skip data collection')
    parser.add_argument('--skip-models', action='store_true', help='Skip model training')
    parser.add_argument('--skip-optimization', action='store_true', help='Skip optimization')
    parser.add_argument('--horizon', type=int, default=12, help='Prediction horizon in months')
    parser.add_argument('--increase-quota', type=float, help='Simulate a quota increase by a percentage.')
    parser.add_argument('--decrease-quota', type=float, help='Simulate a quota decrease by a percentage.')
    parser.add_argument('--run-sensitivity-analysis', action='store_true', help='Run quota sensitivity analysis and generate a chart.')
    parser.add_argument('--model-strategy', type=str, default=None, 
                        help='Choose prediction algorithm. Options: ensemble, arima, prophet, xgboost, lstm.')

    args = parser.parse_args()
    
    # Setup
    setup_logging()
    
    print("🚗 SINGAPORE COE PREDICTION & OPTIMIZATION SYSTEM")
    print("=" * 60)
    print("Analyzing Certificate of Entitlement data for policy optimization")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Data Collection
        if not args.skip_data:
            data = collect_data()
        else:
            print("⏭️ Skipping data collection")
            # Load cached data if available
            data = pd.read_csv('data/raw/coe_data_latest.csv', parse_dates=['Date'])
        
        # Step 2: Model Training
        if not args.skip_models:
            models, feature_df = train_models(data)
            evaluation_results = evaluate_models(models)
        else:
            print("⏭️ Skipping model training")
            models = None
            evaluation_results = None
        
        # Step 3: Predictions
        if models:
            predictions = generate_predictions(models, args.horizon)
        else:
            predictions = pd.DataFrame()
        
        # Override model strategy if provided via command line
        if args.model_strategy:
            if models:
                models.config['models']['prediction_strategy'] = args.model_strategy
                print(f"🔧 Overriding prediction strategy to: {args.model_strategy}")
            else:
                print("⚠️ --model-strategy flag provided, but model training was skipped. Flag will be ignored.")

        # Step 5: Policy Simulation
        if models and (args.increase_quota is not None or args.decrease_quota is not None):
            simulate_policy_changes(data, models, args.increase_quota, args.decrease_quota)

        # Step 6: Optimization
        if not args.skip_optimization and models:
            optimization_results, recommendations = optimize_quotas(data, models)
        else:
            print("⏭️ Skipping optimization")
            recommendations = None
        
        # Step 7: Reporting
        report = generate_report(data, predictions, recommendations)
        
        # Step 8: Visualization
        create_visualizations(data, predictions)
        
        # Step 9: Sensitivity Analysis
        if args.run_sensitivity_analysis and models:
            create_quota_sensitivity_visualization(data, models)
            
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("📊 Results saved to:")
        print("   • coe_report.json (comprehensive report)")
        print("   • coe_analysis_summary.png (visualizations)")
        print("   • logs/coe_system.log (execution logs)")
        
        if recommendations:
            print(f"\n🎯 Key Recommendation: {recommendations['policy_summary']}")
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        logging.error(f"System execution failed: {e}")
        print(f"❌ System execution failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 