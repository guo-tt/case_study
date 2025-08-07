"""
COE Policy Dashboard

Interactive Streamlit dashboard for Singapore COE price prediction,
quota optimization, and policy scenario analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_collection.coe_data_collector import COEDataCollector
from src.models.coe_predictive_models import COEPredictiveModels
from src.optimization.quota_optimizer import COEQuotaOptimizer

# Page configuration
st.set_page_config(
    page_title="Singapore COE Policy Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-box {
        background-color: #e8f4fd;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

def load_data():
    """Load COE data"""
    with st.spinner("Loading COE data from Singapore government APIs..."):
        try:
            collector = COEDataCollector()
            data = collector.collect_and_process_data()
            st.session_state.raw_data = data
            st.session_state.data_loaded = True
            st.success(f"Successfully loaded {len(data)} records")
        except Exception as e:
            st.error(f"Failed to load data: {e}")

def train_models():
    """Train prediction models"""
    if not st.session_state.data_loaded:
        st.error("Please load data first")
        return
    
    with st.spinner("Training prediction models... This may take a few minutes."):
        try:
            models = COEPredictiveModels()
            feature_df = models.prepare_features(st.session_state.raw_data)
            trained_models = models.train_all_models(feature_df)
            
            st.session_state.models = models
            st.session_state.feature_df = feature_df
            st.session_state.models_trained = True
            
            # Evaluate models
            evaluation_results = models.evaluate_models()
            st.session_state.evaluation_results = evaluation_results
            
            st.success("Models trained successfully!")
            
        except Exception as e:
            st.error(f"Failed to train models: {e}")
            st.exception(e)

def display_data_overview():
    """Display data overview"""
    st.header("📊 Data Overview")
    
    if not st.session_state.data_loaded:
        st.warning("No data loaded. Please load data from the sidebar.")
        return
    
    data = st.session_state.raw_data
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(data))
    
    with col2:
        date_range = data['Date'].max() - data['Date'].min()
        st.metric("Data Span", f"{date_range.days} days")
    
    with col3:
        categories = data['Category'].nunique()
        st.metric("COE Categories", categories)
    
    with col4:
        latest_date = data['Date'].max().strftime("%Y-%m")
        st.metric("Latest Data", latest_date)
    
    # Data by category
    st.subheader("Data by Category")
    category_counts = data.groupby(['Category', 'Metric_Type']).size().reset_index(name='Count')
    
    fig = px.bar(category_counts, x='Category', y='Count', color='Metric_Type',
                 title="Data Points by Category and Metric Type")
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent price trends
    st.subheader("Recent COE Premium Trends")
    
    premium_data = data[data['Metric_Type'] == 'Quota Premium'].copy()
    if len(premium_data) > 0:
        # Filter last 24 months
        recent_data = premium_data[premium_data['Date'] >= 
                                 premium_data['Date'].max() - pd.DateOffset(months=24)]
        
        fig = px.line(recent_data, x='Date', y='Value', color='Category',
                      title="COE Premium Trends (Last 24 Months)",
                      labels={'Value': 'COE Premium (SGD)', 'Date': 'Date'})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

def display_predictions():
    """Display model predictions"""
    st.header("🔮 Price Predictions")
    
    if not st.session_state.models_trained:
        st.warning("Models not trained. Please train models from the sidebar.")
        return
    
    models = st.session_state.models
    
    # Prediction controls
    col1, col2 = st.columns([1, 3])
    
    with col1:
        forecast_months = st.slider("Forecast Horizon (months)", 1, 24, 12)
        selected_categories = st.multiselect(
            "Select Categories",
            ['Cat A', 'Cat B', 'Cat C', 'Cat D', 'Cat E'],
            default=['Cat A', 'Cat B', 'Cat E']
        )
    
    with col2:
        if st.button("Generate Predictions"):
            predictions_data = []
            
            for category in selected_categories:
                try:
                    # Get ensemble predictions
                    ensemble_pred = models.ensemble_predict(category, periods=forecast_months)
                    
                    if len(ensemble_pred) > 0:
                        # Create future dates
                        last_date = st.session_state.raw_data['Date'].max()
                        future_dates = pd.date_range(
                            start=last_date + pd.DateOffset(months=1),
                            periods=forecast_months,
                            freq='MS'
                        )
                        
                        for i, (date, price) in enumerate(zip(future_dates, ensemble_pred)):
                            predictions_data.append({
                                'Date': date,
                                'Category': category,
                                'Predicted_Price': price,
                                'Month_Ahead': i + 1
                            })
                
                except Exception as e:
                    st.error(f"Failed to generate predictions for {category}: {e}")
            
            if predictions_data:
                pred_df = pd.DataFrame(predictions_data)
                
                # Plot predictions
                fig = px.line(pred_df, x='Date', y='Predicted_Price', color='Category',
                             title=f"COE Price Forecasts - Next {forecast_months} Months",
                             labels={'Predicted_Price': 'Predicted COE Premium (SGD)'})
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show predictions table
                st.subheader("Prediction Summary")
                
                # Average predictions by category
                avg_predictions = pred_df.groupby('Category')['Predicted_Price'].agg(['mean', 'min', 'max']).round(0)
                avg_predictions.columns = ['Average', 'Minimum', 'Maximum']
                st.dataframe(avg_predictions, use_container_width=True)

def display_optimization():
    """Display quota optimization results"""
    st.header("⚡ Quota Optimization")
    
    if not st.session_state.models_trained:
        st.warning("Models not trained. Please train models first.")
        return
    
    # Optimization controls
    st.subheader("Optimization Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        objective_weights = st.expander("Objective Weights")
        with objective_weights:
            stability_weight = st.slider("Price Stability", 0.0, 1.0, 0.5)
            revenue_weight = st.slider("Revenue Generation", 0.0, 1.0, 0.3)
            efficiency_weight = st.slider("Market Efficiency", 0.0, 1.0, 0.2)
    
    with col2:
        constraints = st.expander("Constraints")
        with constraints:
            max_increase = st.slider("Max Quota Increase (%)", 0, 50, 30)
            max_decrease = st.slider("Max Quota Decrease (%)", 0, 50, 20)
            volatility_threshold = st.slider("Max Price Volatility", 0.05, 0.3, 0.15)
    
    if st.button("Run Optimization", type="primary"):
        with st.spinner("Running optimization algorithms..."):
            try:
                # Initialize optimizer
                optimizer = COEQuotaOptimizer()
                optimizer.set_data_and_predictor(st.session_state.raw_data, st.session_state.models)
                
                # Update constraints
                optimizer.constraints['max_quota_change'] = max_increase / 100
                optimizer.constraints['min_quota_change'] = -max_decrease / 100
                optimizer.constraints['price_volatility_threshold'] = volatility_threshold
                
                # Run optimization
                results = optimizer.run_all_optimizations()
                recommendations = optimizer.generate_policy_recommendations(results)
                
                st.session_state.optimization_results = results
                st.session_state.recommendations = recommendations
                
                # Display results
                display_optimization_results(results, recommendations)
                
            except Exception as e:
                st.error(f"Optimization failed: {e}")
                st.exception(e)
    
    # Display cached results if available
    if 'optimization_results' in st.session_state:
        display_optimization_results(
            st.session_state.optimization_results, 
            st.session_state.recommendations
        )

def display_optimization_results(results, recommendations):
    """Display optimization results"""
    st.subheader("🎯 Optimization Results")
    
    if recommendations['status'] != 'success':
        st.error("No valid optimization solutions found")
        return
    
    # Best algorithm and objective value
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Best Algorithm", recommendations['best_algorithm'].replace('_', ' ').title())
    
    with col2:
        st.metric("Objective Value", f"{recommendations['objective_value']:.4f}")
    
    with col3:
        st.metric("Execution Time", f"{recommendations['execution_time']:.2f}s")
    
    # Policy summary
    st.markdown(f"""
    <div class="recommendation-box">
        <h4>📋 Policy Recommendations</h4>
        <p>{recommendations['policy_summary']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quota adjustments table
    st.subheader("Recommended Quota Adjustments")
    
    adjustments_data = []
    for category, impact in recommendations['quota_adjustments'].items():
        adjustments_data.append({
            'Category': category,
            'Current Quota': int(impact['current_quota']),
            'Recommended Quota': int(impact['recommended_quota']),
            'Change (%)': f"{impact['change_percentage']:+.1f}%",
            'Predicted Price (SGD)': f"${impact['predicted_price']:,.0f}"
        })
    
    adj_df = pd.DataFrame(adjustments_data)
    st.dataframe(adj_df, use_container_width=True)
    
    # Visualization of changes
    fig = go.Figure()
    
    categories = list(recommendations['quota_adjustments'].keys())
    current_quotas = [recommendations['quota_adjustments'][cat]['current_quota'] for cat in categories]
    recommended_quotas = [recommendations['quota_adjustments'][cat]['recommended_quota'] for cat in categories]
    
    fig.add_trace(go.Bar(
        name='Current Quota',
        x=categories,
        y=current_quotas,
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Recommended Quota',
        x=categories,
        y=recommended_quotas,
        marker_color='darkblue'
    ))
    
    fig.update_layout(
        title="Current vs Recommended Quotas",
        xaxis_title="COE Category",
        yaxis_title="Quota",
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Algorithm comparison
    st.subheader("Algorithm Performance Comparison")
    
    comparison_data = []
    for algorithm, result in results.items():
        comparison_data.append({
            'Algorithm': algorithm.replace('_', ' ').title(),
            'Objective Value': result.objective_value,
            'Execution Time (s)': result.execution_time,
            'Iterations': result.iterations,
            'Constraints Satisfied': '✅' if result.constraints_satisfied else '❌'
        })
    
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True)

def display_model_performance():
    """Display model performance metrics"""
    st.header("🎯 Model Performance")
    
    if not st.session_state.models_trained:
        st.warning("Models not trained. Please train models first.")
        return
    
    if 'evaluation_results' not in st.session_state:
        st.error("No evaluation results available.")
        return
    
    evaluation_results = st.session_state.evaluation_results
    
    # Create performance summary
    performance_data = []
    
    for category, category_results in evaluation_results.items():
        for model_type, metrics in category_results.items():
            performance_data.append({
                'Category': category,
                'Model': model_type.upper(),
                'MAE': metrics.get('mae', 0),
                'RMSE': metrics.get('rmse', 0),
                'R²': metrics.get('r2', 0)
            })
    
    if performance_data:
        perf_df = pd.DataFrame(performance_data)
        
        # Performance metrics by model
        st.subheader("Model Performance by Category")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Mean Absolute Error', 'Root Mean Square Error', 
                          'R² Score', 'Model Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # MAE
        mae_fig = px.bar(perf_df, x='Category', y='MAE', color='Model', 
                        title="Mean Absolute Error by Category and Model")
        for trace in mae_fig.data:
            fig.add_trace(trace, row=1, col=1)
        
        # RMSE
        rmse_fig = px.bar(perf_df, x='Category', y='RMSE', color='Model',
                         title="RMSE by Category and Model")
        for trace in rmse_fig.data:
            fig.add_trace(trace, row=1, col=2)
        
        # R²
        r2_fig = px.bar(perf_df, x='Category', y='R²', color='Model',
                       title="R² Score by Category and Model")
        for trace in r2_fig.data:
            fig.add_trace(trace, row=2, col=1)
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Best performing models
        st.subheader("Best Performing Models by Category")
        
        best_models = []
        for category in perf_df['Category'].unique():
            cat_data = perf_df[perf_df['Category'] == category]
            if len(cat_data) > 0:
                # Best model based on R² score
                best_model = cat_data.loc[cat_data['R²'].idxmax()]
                best_models.append({
                    'Category': category,
                    'Best Model': best_model['Model'],
                    'R² Score': f"{best_model['R²']:.3f}",
                    'MAE': f"{best_model['MAE']:.0f}",
                    'RMSE': f"{best_model['RMSE']:.0f}"
                })
        
        if best_models:
            best_df = pd.DataFrame(best_models)
            st.dataframe(best_df, use_container_width=True)

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<div class="main-header">🚗 Singapore COE Policy Dashboard</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This dashboard provides comprehensive analysis and optimization tools for Singapore's
    Certificate of Entitlement (COE) system, helping policy makers make data-driven decisions.
    """)
    
    # Sidebar
    st.sidebar.title("🔧 Controls")
    
    # Data loading
    st.sidebar.subheader("Data Management")
    if st.sidebar.button("Load COE Data"):
        load_data()
    
    if st.sidebar.button("Train Models"):
        train_models()
    
    # Navigation
    st.sidebar.subheader("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Data Overview", "Price Predictions", "Quota Optimization", "Model Performance"]
    )
    
    # Status indicators
    st.sidebar.subheader("System Status")
    data_status = "✅ Loaded" if st.session_state.data_loaded else "❌ Not Loaded"
    st.sidebar.text(f"Data: {data_status}")
    
    models_status = "✅ Trained" if st.session_state.models_trained else "❌ Not Trained"
    st.sidebar.text(f"Models: {models_status}")
    
    # Main content
    if page == "Data Overview":
        display_data_overview()
    elif page == "Price Predictions":
        display_predictions()
    elif page == "Quota Optimization":
        display_optimization()
    elif page == "Model Performance":
        display_model_performance()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        Singapore COE Policy Dashboard | Land Transport Authority | Data from data.gov.sg
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 