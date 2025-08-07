#!/usr/bin/env python3
"""
Comprehensive COE Analysis and Visualization Script (LSTM Only)

This script generates complete data analysis, LSTM-only predictions for 1-12 months,
and creates multiple visualizations including quota sensitivity analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.data_collection.coe_data_collector import COEDataCollector
from src.models.coe_predictive_models import COEPredictiveModels
from src.optimization.quota_optimizer import COEQuotaOptimizer

# Set style for all plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class COEComprehensiveAnalysis:
    """Comprehensive analysis and visualization for COE data (LSTM only)"""
    
    def __init__(self):
        self.data = None
        self.models = None
        self.predictions = {}
        self.results_dir = "analysis_results"
        
        # Create results directory
        import os
        os.makedirs(self.results_dir, exist_ok=True)
        
    def collect_and_prepare_data(self):
        """Step 1: Collect and prepare data"""
        print("🔍 STEP 1: DATA COLLECTION AND PREPARATION")
        print("=" * 60)
        
        collector = COEDataCollector()
        self.data = collector.collect_and_process_data()
        
        print(f"✅ Successfully collected {len(self.data)} records")
        print(f"📅 Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        print(f"🏷️ Categories: {self.data['Category'].unique().tolist()}")
        
        return self.data
    
    def train_models(self):
        """Step 2: Train all prediction models"""
        print("\n🎯 STEP 2: MODEL TRAINING (LSTM FOCUS)")
        print("=" * 60)
        
        self.models = COEPredictiveModels()
        # Set prediction strategy to LSTM only
        self.models.config['models']['prediction_strategy'] = 'lstm'
        
        feature_df = self.models.prepare_features(self.data)
        trained_models = self.models.train_all_models(feature_df)
        
        print(f"✅ Successfully trained models for {len(trained_models)} categories")
        for category, category_models in trained_models.items():
            print(f"   {category}: {list(category_models.keys())}")
        
        return self.models
    
    def generate_lstm_predictions(self):
        """Step 3: Generate LSTM predictions only"""
        print("\n🔮 STEP 3: GENERATING LSTM PREDICTIONS")
        print("=" * 60)
        
        categories = ['Cat A', 'Cat B', 'Cat C', 'Cat D', 'Cat E']
        horizons = range(1, 13)  # 1-12 months
        
        all_predictions = []
        
        for category in categories:
            print(f"\n📊 Generating LSTM predictions for {category}...")
            
            try:
                # Use get_prediction which will use LSTM based on config
                predictions = self.models.get_prediction(category, periods=12)
                
                if len(predictions) > 0:
                    for i, pred in enumerate(predictions):
                        all_predictions.append({
                            'Category': category,
                            'Model': 'LSTM',
                            'Month_Ahead': i + 1,
                            'Predicted_Price': pred,
                            'Date_Predicted': datetime.now() + timedelta(days=30*(i+1))
                        })
                    print(f"   ✅ LSTM: {len(predictions)} predictions generated")
                else:
                    print(f"   ❌ LSTM: No predictions generated")
                    
            except Exception as e:
                print(f"   ⚠️ LSTM: Failed ({str(e)[:50]}...)")
        
        self.predictions_df = pd.DataFrame(all_predictions)
        print(f"\n✅ Total LSTM predictions generated: {len(all_predictions)}")
        
        return self.predictions_df
    
    def create_historical_analysis_plots(self):
        """Create historical data analysis plots"""
        print("\n📈 Creating historical analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('COE Historical Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Historical price trends (last 3 years)
        premium_data = self.data[self.data['Metric_Type'] == 'Quota Premium']
        recent_data = premium_data[premium_data['Date'] >= 
                                  premium_data['Date'].max() - pd.DateOffset(years=3)]
        
        for category in ['Cat A', 'Cat B', 'Cat E']:
            cat_data = recent_data[recent_data['Category'] == category]
            if len(cat_data) > 0:
                axes[0, 0].plot(cat_data['Date'], cat_data['Value'], 
                               marker='o', markersize=3, label=category, linewidth=2)
        
        axes[0, 0].set_title('COE Premium Trends (Last 3 Years)', fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Premium (SGD)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Average prices by category (last 12 months)
        last_12_months = premium_data[premium_data['Date'] >= 
                                     premium_data['Date'].max() - pd.DateOffset(months=12)]
        avg_prices = last_12_months.groupby('Category')['Value'].mean().sort_values(ascending=False)
        
        bars = axes[0, 1].bar(avg_prices.index, avg_prices.values, color=sns.color_palette("husl", len(avg_prices)))
        axes[0, 1].set_title('Average COE Prices (Last 12 Months)', fontweight='bold')
        axes[0, 1].set_ylabel('Average Premium (SGD)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_prices.values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                           f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Price volatility analysis
        volatility = last_12_months.groupby('Category')['Value'].std().sort_values(ascending=False)
        bars = axes[1, 0].bar(volatility.index, volatility.values, color=sns.color_palette("rocket", len(volatility)))
        axes[1, 0].set_title('Price Volatility by Category (Last 12 Months)', fontweight='bold')
        axes[1, 0].set_ylabel('Standard Deviation (SGD)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, volatility.values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                           f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Data coverage heatmap
        coverage = self.data.groupby(['Category', 'Metric_Type']).size().reset_index(name='Count')
        coverage_pivot = coverage.pivot(index='Category', columns='Metric_Type', values='Count').fillna(0)
        
        im = axes[1, 1].imshow(coverage_pivot.values, cmap='Blues', aspect='auto')
        axes[1, 1].set_xticks(range(len(coverage_pivot.columns)))
        axes[1, 1].set_xticklabels(coverage_pivot.columns, rotation=45, ha='right')
        axes[1, 1].set_yticks(range(len(coverage_pivot.index)))
        axes[1, 1].set_yticklabels(coverage_pivot.index)
        axes[1, 1].set_title('Data Coverage by Category and Metric', fontweight='bold')
        
        # Add text annotations
        for i in range(len(coverage_pivot.index)):
            for j in range(len(coverage_pivot.columns)):
                text = axes[1, 1].text(j, i, int(coverage_pivot.iloc[i, j]),
                                     ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=axes[1, 1], label='Number of Records')
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/01_historical_analysis.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Historical analysis saved to {self.results_dir}/01_historical_analysis.png")
        plt.close()
    
    def create_lstm_prediction_plots(self):
        """Create LSTM prediction plots"""
        print("\n🔮 Creating LSTM prediction plots...")
        
        if self.predictions_df.empty:
            print("   ⚠️ No predictions available for plotting")
            return
        
        # Create LSTM prediction visualization
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('COE Price Predictions: LSTM Model Analysis', fontsize=16, fontweight='bold')
        
        categories = ['Cat A', 'Cat B', 'Cat C', 'Cat D', 'Cat E']
        
        # Individual category plots
        for i, category in enumerate(categories):
            ax = axes[i//3, i%3]
            cat_data = self.predictions_df[self.predictions_df['Category'] == category].sort_values('Month_Ahead')
            
            if not cat_data.empty:
                # Line plot with markers
                ax.plot(cat_data['Month_Ahead'], cat_data['Predicted_Price'], 
                       marker='o', linewidth=3, markersize=6, color='darkblue', alpha=0.8)
                
                # Fill area under curve
                ax.fill_between(cat_data['Month_Ahead'], cat_data['Predicted_Price'], 
                               alpha=0.3, color='lightblue')
                
                ax.set_title(f'{category} - LSTM 12-Month Forecast', fontweight='bold')
                ax.set_xlabel('Months Ahead')
                ax.set_ylabel('Predicted Price (SGD)')
                ax.grid(True, alpha=0.3)
                ax.set_xticks(range(1, 13))
                
                # Add value annotations for key months
                for month in [1, 6, 12]:
                    if month <= len(cat_data):
                        price = cat_data[cat_data['Month_Ahead'] == month]['Predicted_Price'].iloc[0]
                        ax.annotate(f'${price:,.0f}', 
                                   xy=(month, price), xytext=(5, 10), 
                                   textcoords='offset points', fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Remove empty subplot
        if len(categories) < 6:
            fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/02_lstm_predictions.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ LSTM predictions saved to {self.results_dir}/02_lstm_predictions.png")
        plt.close()
    
    def create_quota_sensitivity_analysis(self):
        """Create quota sensitivity analysis showing price response to quota changes"""
        print("\n⚡ Creating quota sensitivity analysis...")
        
        # Initialize optimizer
        optimizer = COEQuotaOptimizer()
        optimizer.set_data_and_predictor(self.data, self.models)
        
        # Define quota change range: -50% to +50%
        quota_changes = np.linspace(-0.5, 0.5, 21)
        results = []
        
        print("   🔬 Running sensitivity analysis for quota changes...")
        
        # Get baseline prices (no change)
        baseline_prices = optimizer.predict_prices_with_quotas({cat: 1.0 for cat in optimizer.optimizable_categories})
        
        for change in quota_changes:
            multiplier = 1.0 + change
            adjustments = {cat: multiplier for cat in optimizer.optimizable_categories}
            predicted_prices = optimizer.predict_prices_with_quotas(adjustments)
            
            for category, price in predicted_prices.items():
                baseline_price = baseline_prices.get(category, price)
                price_change_pct = ((price - baseline_price) / baseline_price) * 100 if baseline_price > 0 else 0
                
                results.append({
                    'quota_change_pct': change * 100,
                    'category': category,
                    'predicted_price': price,
                    'baseline_price': baseline_price,
                    'price_change_pct': price_change_pct
                })
        
        results_df = pd.DataFrame(results)
        
        # Create the sensitivity plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('COE Quota Sensitivity Analysis', fontsize=16, fontweight='bold')
        
        # 1. Price vs Quota Change (absolute prices)
        for category in results_df['category'].unique():
            cat_data = results_df[results_df['category'] == category]
            axes[0, 0].plot(cat_data['quota_change_pct'], cat_data['predicted_price'], 
                           marker='o', label=category, linewidth=2, markersize=4)
        
        axes[0, 0].set_title('Price Response to Quota Changes', fontweight='bold')
        axes[0, 0].set_xlabel('Quota Change (%)')
        axes[0, 0].set_ylabel('Predicted COE Premium (SGD)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7, label='No Change')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Price Change Percentage vs Quota Change
        for category in results_df['category'].unique():
            cat_data = results_df[results_df['category'] == category]
            axes[0, 1].plot(cat_data['quota_change_pct'], cat_data['price_change_pct'], 
                           marker='s', label=category, linewidth=2, markersize=4)
        
        axes[0, 1].set_title('Price Change (%) vs Quota Change (%)', fontweight='bold')
        axes[0, 1].set_xlabel('Quota Change (%)')
        axes[0, 1].set_ylabel('Price Change (%)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Heatmap of price changes
        heatmap_data = results_df.pivot(index='category', columns='quota_change_pct', values='price_change_pct')
        im = axes[1, 0].imshow(heatmap_data.values, cmap='RdYlGn_r', aspect='auto', 
                              vmin=-20, vmax=20)
        axes[1, 0].set_title('Price Change Heatmap (%)', fontweight='bold')
        axes[1, 0].set_xlabel('Quota Change (%)')
        axes[1, 0].set_ylabel('COE Category')
        
        # Set ticks
        x_ticks = range(0, len(heatmap_data.columns), 4)  # Every 4th tick
        axes[1, 0].set_xticks(x_ticks)
        axes[1, 0].set_xticklabels([f'{heatmap_data.columns[i]:.0f}%' for i in x_ticks])
        axes[1, 0].set_yticks(range(len(heatmap_data.index)))
        axes[1, 0].set_yticklabels(heatmap_data.index)
        
        plt.colorbar(im, ax=axes[1, 0], label='Price Change (%)')
        
        # 4. Elasticity analysis (price sensitivity)
        elasticity_data = []
        for category in results_df['category'].unique():
            cat_data = results_df[results_df['category'] == category]
            # Calculate elasticity as percentage change in price / percentage change in quota
            for i in range(1, len(cat_data)):
                quota_change = cat_data.iloc[i]['quota_change_pct'] - cat_data.iloc[i-1]['quota_change_pct']
                price_change = cat_data.iloc[i]['price_change_pct'] - cat_data.iloc[i-1]['price_change_pct']
                if quota_change != 0:
                    elasticity = price_change / quota_change
                    elasticity_data.append({
                        'category': category,
                        'quota_change_pct': cat_data.iloc[i]['quota_change_pct'],
                        'elasticity': elasticity
                    })
        
        if elasticity_data:
            elast_df = pd.DataFrame(elasticity_data)
            avg_elasticity = elast_df.groupby('category')['elasticity'].mean()
            
            bars = axes[1, 1].bar(avg_elasticity.index, avg_elasticity.values, 
                                 color=sns.color_palette("viridis", len(avg_elasticity)))
            axes[1, 1].set_title('Average Price Elasticity by Category', fontweight='bold')
            axes[1, 1].set_ylabel('Price Elasticity')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.7)
            
            # Add value labels
            for bar, value in zip(bars, avg_elasticity.values):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, 
                               bar.get_height() + (0.05 if value >= 0 else -0.1),
                               f'{value:.2f}', ha='center', va='bottom' if value >= 0 else 'top', 
                               fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/03_quota_sensitivity.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Quota sensitivity analysis saved to {self.results_dir}/03_quota_sensitivity.png")
        plt.close()
        
        return results_df
    
    def create_combined_forecast_plot(self):
        """Create combined LSTM forecast with quota sensitivity overlay"""
        print("\n📊 Creating combined forecast and sensitivity plot...")
        
        if self.predictions_df.empty:
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        fig.suptitle('COE Analysis: LSTM Forecasts & Quota Sensitivity', fontsize=16, fontweight='bold')
        
        # 1. LSTM 12-month forecasts
        categories = ['Cat A', 'Cat B', 'Cat C', 'Cat D', 'Cat E']
        colors = sns.color_palette("husl", len(categories))
        
        for i, category in enumerate(categories):
            cat_data = self.predictions_df[self.predictions_df['Category'] == category].sort_values('Month_Ahead')
            if not cat_data.empty:
                axes[0].plot(cat_data['Month_Ahead'], cat_data['Predicted_Price'], 
                           marker='o', label=category, linewidth=3, markersize=6, color=colors[i])
        
        axes[0].set_title('LSTM 12-Month Price Forecasts', fontweight='bold')
        axes[0].set_xlabel('Months Ahead')
        axes[0].set_ylabel('Predicted Price (SGD)')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(range(1, 13))
        
        # 2. Quota sensitivity (simplified view)
        optimizer = COEQuotaOptimizer()
        optimizer.set_data_and_predictor(self.data, self.models)
        
        quota_changes = np.linspace(-0.3, 0.3, 13)  # -30% to +30%
        sensitivity_results = []
        
        baseline_prices = optimizer.predict_prices_with_quotas({cat: 1.0 for cat in optimizer.optimizable_categories})
        
        for change in quota_changes:
            multiplier = 1.0 + change
            adjustments = {cat: multiplier for cat in optimizer.optimizable_categories}
            predicted_prices = optimizer.predict_prices_with_quotas(adjustments)
            
            for category, price in predicted_prices.items():
                sensitivity_results.append({
                    'quota_change_pct': change * 100,
                    'category': category,
                    'predicted_price': price
                })
        
        sens_df = pd.DataFrame(sensitivity_results)
        
        for i, category in enumerate(categories):
            cat_data = sens_df[sens_df['category'] == category]
            if not cat_data.empty:
                axes[1].plot(cat_data['quota_change_pct'], cat_data['predicted_price'], 
                           marker='s', label=category, linewidth=3, markersize=6, color=colors[i])
        
        axes[1].set_title('Price Sensitivity to Quota Changes', fontweight='bold')
        axes[1].set_xlabel('Quota Change (%)')
        axes[1].set_ylabel('Predicted Price (SGD)')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(0, color='red', linestyle='--', alpha=0.7, label='No Change')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/04_combined_analysis.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Combined analysis saved to {self.results_dir}/04_combined_analysis.png")
        plt.close()
    
    def create_summary_report(self):
        """Create a summary report with key insights"""
        print("\n📋 Creating summary report...")
        
        report = {
            'analysis_date': datetime.now().isoformat(),
            'model_used': 'LSTM Only',
            'data_summary': {
                'total_records': len(self.data),
                'date_range': [self.data['Date'].min().isoformat(), self.data['Date'].max().isoformat()],
                'categories': self.data['Category'].unique().tolist()
            }
        }
        
        if not self.predictions_df.empty:
            # Key predictions summary
            lstm_1month = self.predictions_df[self.predictions_df['Month_Ahead'] == 1]
            lstm_3month = self.predictions_df[self.predictions_df['Month_Ahead'] == 3]
            
            if not lstm_1month.empty:
                report['lstm_predictions'] = {}
                for _, row in lstm_1month.iterrows():
                    category = row['Category']
                    month_3_data = lstm_3month[lstm_3month['Category'] == category]
                    
                    report['lstm_predictions'][category] = {
                        '1_month': float(row['Predicted_Price']),
                        '3_month': float(month_3_data['Predicted_Price'].iloc[0]) if len(month_3_data) > 0 else None,
                        '12_month': float(self.predictions_df[
                            (self.predictions_df['Category'] == category) &
                            (self.predictions_df['Month_Ahead'] == 12)
                        ]['Predicted_Price'].iloc[0]) if len(self.predictions_df[
                            (self.predictions_df['Category'] == category) &
                            (self.predictions_df['Month_Ahead'] == 12)
                        ]) > 0 else None
                    }
        
        # Save report
        import json
        with open(f'{self.results_dir}/lstm_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"   ✅ Summary report saved to {self.results_dir}/lstm_analysis_report.json")
        
        # Print key insights
        print("\n📊 KEY LSTM PREDICTIONS:")
        print("=" * 60)
        if 'lstm_predictions' in report:
            for category, predictions in report['lstm_predictions'].items():
                print(f"{category}:")
                print(f"   1-month:  ${predictions['1_month']:,.0f}")
                if predictions['3_month']:
                    print(f"   3-month:  ${predictions['3_month']:,.0f}")
                if predictions['12_month']:
                    print(f"   12-month: ${predictions['12_month']:,.0f}")
        
        return report
    
    def run_comprehensive_analysis(self):
        """Run the complete LSTM-focused analysis pipeline"""
        print("🚗 SINGAPORE COE COMPREHENSIVE ANALYSIS (LSTM + QUOTA SENSITIVITY)")
        print("=" * 80)
        print("Generating LSTM predictions and quota sensitivity analysis")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Step 1: Data collection
            self.collect_and_prepare_data()
            
            # Step 2: Model training (LSTM focus)
            self.train_models()
            
            # Step 3: Generate LSTM predictions
            self.generate_lstm_predictions()
            
            # Step 4: Create all visualizations
            self.create_historical_analysis_plots()
            self.create_lstm_prediction_plots()
            self.create_quota_sensitivity_analysis()
            self.create_combined_forecast_plot()
            
            # Step 5: Generate summary report
            self.create_summary_report()
            
            print("\n" + "=" * 80)
            print("✅ LSTM COMPREHENSIVE ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("📁 Results saved to:")
            print(f"   • {self.results_dir}/01_historical_analysis.png")
            print(f"   • {self.results_dir}/02_lstm_predictions.png")
            print(f"   • {self.results_dir}/03_quota_sensitivity.png")
            print(f"   • {self.results_dir}/04_combined_analysis.png")
            print(f"   • {self.results_dir}/lstm_analysis_report.json")
            print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            raise

if __name__ == "__main__":
    analyzer = COEComprehensiveAnalysis()
    analyzer.run_comprehensive_analysis() 