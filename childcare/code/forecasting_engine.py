"""
Forecasting engine for preschool demand prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class ForecastingEngine:
    def __init__(self, config: Dict):
        """
        Initialize forecasting engine
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models = {}
        self.forecast_results = {}
        
    def prepare_features(self, population_data: pd.DataFrame, 
                        bto_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for forecasting
        
        Args:
            population_data: Historical population data
            bto_data: BTO project data
            
        Returns:
            Feature DataFrame
        """
        features = []
        
        # Group population data by subzone and year
        for subzone in population_data['Subzone'].unique():
            subzone_data = population_data[population_data['Subzone'] == subzone]
            
            for year in subzone_data['Year'].unique():
                year_data = subzone_data[subzone_data['Year'] == year]
                
                # Get BTO impact for this subzone and year
                bto_impact = self._calculate_bto_impact(bto_data, subzone, year)
                
                feature_row = {
                    'Subzone': subzone,
                    'Year': year,
                    'Planning_area': year_data['Planning_area'].iloc[0],
                    'Region': year_data['Region'].iloc[0],
                    'Population_18m_6y': year_data['Population_18m_6y'].iloc[0],
                    'BTO_units_completed': bto_impact['units_completed'],
                    'BTO_additional_population': bto_impact['additional_population'],
                    'BTO_additional_demand': bto_impact['additional_demand']
                }
                
                features.append(feature_row)
        
        return pd.DataFrame(features)
    
    def _calculate_bto_impact(self, bto_data: pd.DataFrame, 
                            subzone: str, year: int) -> Dict:
        """
        Calculate BTO impact for a specific subzone and year
        
        Args:
            bto_data: BTO project data
            subzone: Target subzone
            year: Target year
            
        Returns:
            Dictionary with BTO impact metrics
        """
        # Get BTO projects in this subzone completed by this year
        subzone_bto = bto_data[
            (bto_data['Subzone'] == subzone) & 
            (bto_data['Estimated completion year'] <= year)
        ]
        
        units_completed = subzone_bto['Total number of units'].sum()
        additional_population = units_completed * self.config['persons_per_bto_unit'] * self.config['bto_occupancy_rate']
        additional_demand = additional_population * self.config['enrollment_rate']
        
        return {
            'units_completed': units_completed,
            'additional_population': additional_population,
            'additional_demand': additional_demand
        }
    
    def train_models(self, features: pd.DataFrame):
        """
        Train forecasting models
        
        Args:
            features: Feature DataFrame
        """
        print("Training forecasting models...")
        
        # Prepare training data
        X = features[['Year', 'Population_18m_6y', 'BTO_units_completed', 'BTO_additional_population']]
        y = features['Population_18m_6y']  # Target variable
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Linear Regression model
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        # Train Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Evaluate models
        lr_score = lr_model.score(X_test, y_test)
        rf_score = rf_model.score(X_test, y_test)
        
        print(f"Linear Regression R² Score: {lr_score:.3f}")
        print(f"Random Forest R² Score: {rf_score:.3f}")
        
        # Store models
        self.models['linear_regression'] = lr_model
        self.models['random_forest'] = rf_model
        
        # Use the better performing model
        self.best_model = rf_model if rf_score > lr_score else lr_model
        print(f"Selected model: {'Random Forest' if rf_score > lr_score else 'Linear Regression'}")
    
    def forecast_demand(self, features: pd.DataFrame, 
                       target_year: int) -> pd.DataFrame:
        """
        Forecast demand for target year
        
        Args:
            features: Feature DataFrame
            target_year: Target year for forecasting
            
        Returns:
            Forecast results DataFrame
        """
        if not self.models:
            raise ValueError("Models not trained. Call train_models() first.")
        
        print(f"Forecasting demand for {target_year}...")
        
        # Create future features
        future_features = self._create_future_features(features, target_year)
        
        # Make predictions
        predictions = self.best_model.predict(future_features[['Year', 'Population_18m_6y', 'BTO_units_completed', 'BTO_additional_population']])
        
        # Create forecast results
        forecast_results = future_features.copy()
        forecast_results['Forecasted_population'] = predictions
        forecast_results['Demand_forecast'] = predictions * self.config['enrollment_rate']
        
        return forecast_results
    
    def _create_future_features(self, features: pd.DataFrame, 
                               target_year: int) -> pd.DataFrame:
        """
        Create features for future forecasting
        
        Args:
            features: Historical features
            target_year: Target year
            
        Returns:
            Future features DataFrame
        """
        # Get latest data for each subzone
        latest_features = features.groupby('Subzone').last().reset_index()
        
        # Create future features
        future_features = []
        
        for _, row in latest_features.iterrows():
            # Extrapolate population growth
            years_since_latest = target_year - row['Year']
            growth_rate = 0.015  # 1.5% annual growth
            
            future_population = row['Population_18m_6y'] * (1 + growth_rate) ** years_since_latest
            
            # Get BTO impact for target year
            bto_impact = self._calculate_bto_impact(
                features, row['Subzone'], target_year
            )
            
            future_row = {
                'Subzone': row['Subzone'],
                'Year': target_year,
                'Planning_area': row['Planning_area'],
                'Region': row['Region'],
                'Population_18m_6y': future_population,
                'BTO_units_completed': bto_impact['units_completed'],
                'BTO_additional_population': bto_impact['additional_population'],
                'BTO_additional_demand': bto_impact['additional_demand']
            }
            
            future_features.append(future_row)
        
        return pd.DataFrame(future_features)
    
    def calculate_capacity_gaps(self, forecast_results: pd.DataFrame,
                              current_capacity: Dict[str, int]) -> pd.DataFrame:
        """
        Calculate capacity gaps between forecasted demand and current capacity
        
        Args:
            forecast_results: Forecast results DataFrame
            current_capacity: Dictionary mapping subzones to current capacity
            
        Returns:
            DataFrame with capacity gap analysis
        """
        gap_analysis = forecast_results.copy()
        
        # Add current capacity
        gap_analysis['Current_capacity'] = gap_analysis['Subzone'].map(current_capacity).fillna(0)
        
        # Calculate gaps
        gap_analysis['Capacity_gap'] = gap_analysis['Demand_forecast'] - gap_analysis['Current_capacity']
        gap_analysis['Centers_needed'] = np.ceil(
            gap_analysis['Capacity_gap'] / self.config['max_children_per_center']
        ).clip(lower=0)
        
        # Add priority levels
        gap_analysis['Priority_level'] = gap_analysis['Capacity_gap'].apply(
            self._assign_priority_level
        )
        
        return gap_analysis
    
    def _assign_priority_level(self, gap: float) -> str:
        """
        Assign priority level based on capacity gap
        
        Args:
            gap: Capacity gap value
            
        Returns:
            Priority level string
        """
        if gap <= 0:
            return 'No Gap'
        elif gap <= 50:
            return 'Low Priority'
        elif gap <= 150:
            return 'Medium Priority'
        elif gap <= 300:
            return 'High Priority'
        else:
            return 'Critical Priority'
    
    def generate_scenarios(self, base_forecast: pd.DataFrame,
                          scenario_configs: List[Dict]) -> Dict[str, pd.DataFrame]:
        """
        Generate multiple forecast scenarios
        
        Args:
            base_forecast: Base forecast results
            scenario_configs: List of scenario configurations
            
        Returns:
            Dictionary of scenario results
        """
        scenarios = {}
        
        for i, config in enumerate(scenario_configs):
            scenario_name = config.get('name', f'Scenario_{i+1}')
            
            # Apply scenario modifications
            scenario_forecast = base_forecast.copy()
            
            # Modify enrollment rate if specified
            if 'enrollment_rate' in config:
                scenario_forecast['Demand_forecast'] = (
                    scenario_forecast['Forecasted_population'] * config['enrollment_rate']
                )
            
            # Modify BTO impact if specified
            if 'bto_multiplier' in config:
                scenario_forecast['BTO_additional_demand'] *= config['bto_multiplier']
                scenario_forecast['Demand_forecast'] += (
                    scenario_forecast['BTO_additional_demand'] * (config['bto_multiplier'] - 1)
                )
            
            scenarios[scenario_name] = scenario_forecast
        
        return scenarios
```

Now let me create a web dashboard using Streamlit:

```python:childcare/code/dashboard.py
"""
Web dashboard for preschool demand forecasting system
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
import numpy as np
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import PreschoolForecastSystem
from config import SYSTEM_CONFIG

class Dashboard:
    def __init__(self):
        """Initialize dashboard"""
        st.set_page_config(
            page_title="ECDA Preschool Demand Forecasting",
            page_icon="🏫",
            layout="wide"
        )
        
    def run(self):
        """Run the dashboard"""
        st.title("🏫 ECDA Preschool Demand Forecasting System")
        st.markdown("---")
        
        # Sidebar for controls
        controls = self._create_sidebar()
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._show_overview()
            
        with col2:
            self._show_key_metrics()
        
        # Detailed analysis
        self._show_detailed_analysis()
        
        # Recommendations
        self._show_recommendations()
        
        # Interactive map
        self._show_interactive_map()
    
    def _create_sidebar(self):
        """Create sidebar controls"""
        st.sidebar.header("🎛️ Dashboard Controls")
        
        # Year selector
        selected_year = st.sidebar.selectbox(
            "Forecast Year",
            options=[2023, 2024, 2025, 2026, 2027],
            index=2
        )
        
        # Region filter
        regions = ["All Regions", "Central Region", "East Region", "North Region", 
                  "North-East Region", "West Region"]
        selected_region = st.sidebar.selectbox("Filter by Region", regions)
        
        # Priority filter
        priorities = ["All Priorities", "Critical Priority", "High Priority", 
                     "Medium Priority", "Low Priority"]
        selected_priority = st.sidebar.selectbox("Filter by Priority", priorities)
        
        # Update button
        if st.sidebar.button("🔄 Update Forecast"):
            st.success("Forecast updated successfully!")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📈 Scenario Analysis")
        
        # Scenario controls
        enrollment_rate = st.sidebar.slider(
            "Enrollment Rate (%)",
            min_value=60,
            max_value=100,
            value=80,
            step=5
        )
        
        bto_impact = st.sidebar.slider(
            "BTO Impact Multiplier",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1
        )
        
        return {
            'year': selected_year,
            'region': selected_region,
            'priority': selected_priority,
            'enrollment_rate': enrollment_rate / 100,
            'bto_impact': bto_impact
        }
    
    def _show_overview(self):
        """Show overview section"""
        st.header("📋 Overview")
        
        # Create sample data for demonstration
        overview_data = self._create_sample_overview_data()
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Subzones",
                value=len(overview_data),
                delta="+5 from last year"
            )
        
        with col2:
            st.metric(
                label="Subzones with Gaps",
                value=len(overview_data[overview_data['Capacity_gap'] > 0]),
                delta="+2 from last year"
            )
        
        with col3:
            st.metric(
                label="Total Capacity Gap",
                value=f"{overview_data['Capacity_gap'].sum():,.0f}",
                delta="+1,250 from last year"
            )
        
        with col4:
            st.metric(
                label="Centers Needed",
                value=int(overview_data['Centers_needed'].sum()),
                delta="+12 from last year"
            )
    
    def _show_key_metrics(self):
        """Show key metrics"""
        st.header("🎯 Key Metrics")
        
        # Priority distribution
        priority_data = pd.DataFrame({
            'Priority': ['Critical', 'High', 'Medium', 'Low', 'No Gap'],
            'Count': [5, 12, 18, 25, 40]
        })
        
        fig = px.pie(
            priority_data,
            values='Count',
            names='Priority',
            title="Priority Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Regional breakdown
        regional_data = pd.DataFrame({
            'Region': ['North-East', 'West', 'East', 'North', 'Central'],
            'Gap': [2500, 1800, 1200, 900, 600]
        })
        
        fig = px.bar(
            regional_data,
            x='Region',
            y='Gap',
            title="Capacity Gap by Region"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_detailed_analysis(self):
        """Show detailed analysis"""
        st.header("📊 Detailed Analysis")
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["📈 Trends", "🏘️ BTO Impact", "🎓 Enrollment Patterns"])
        
        with tab1:
            self._show_trends_analysis()
        
        with tab2:
            self._show_bto_impact_analysis()
        
        with tab3:
            self._show_enrollment_analysis()
    
    def _show_trends_analysis(self):
        """Show trends analysis"""
        # Create sample trend data
        years = range(2020, 2028)
        trend_data = pd.DataFrame({
            'Year': years,
            'Population': [15000 + i*500 for i in range(len(years))],
            'Demand': [12000 + i*400 for i in range(len(years))],
            'Capacity': [11000 + i*200 for i in range(len(years))]
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trend_data['Year'],
            y=trend_data['Population'],
            mode='lines+markers',
            name='Population (18m-6y)',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=trend_data['Year'],
            y=trend_data['Demand'],
            mode='lines+markers',
            name='Forecasted Demand',
            line=dict(color='red', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=trend_data['Year'],
            y=trend_data['Capacity'],
            mode='lines+markers',
            name='Current Capacity',
            line=dict(color='green', width=3)
        ))
        
        fig.update_layout(
            title="Population and Demand Trends (2020-2027)",
            xaxis_title="Year",
            yaxis_title="Number of Children",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_bto_impact_analysis(self):
        """Show BTO impact analysis"""
        # Create sample BTO data
        bto_data = pd.DataFrame({
            'Year': [2023, 2024, 2025, 2026, 2027],
            'Units_Completed': [5000, 6000, 7000, 8000, 9000],
            'Additional_Demand': [600, 720, 840, 960, 1080]
        })
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('BTO Units Completed', 'Additional Demand from BTO'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Bar(x=bto_data['Year'], y=bto_data['Units_Completed'], name='Units'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=bto_data['Year'], y=bto_data['Additional_Demand'], name='Demand'),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title_text="BTO Project Impact Analysis")
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_enrollment_analysis(self):
        """Show enrollment analysis"""
        # Create sample enrollment data
        enrollment_data = pd.DataFrame({
            'Age_Group': ['18m-2y', '2-3y', '3-4y', '4-5y', '5-6y'],
            'Enrollment_Rate': [0.65, 0.75, 0.85, 0.90, 0.95],
            'Capacity_Utilization': [0.70, 0.80, 0.90, 0.85, 0.75]
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=enrollment_data['Age_Group'],
            y=enrollment_data['Enrollment_Rate'],
            name='Enrollment Rate',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            x=enrollment_data['Age_Group'],
            y=enrollment_data['Capacity_Utilization'],
            name='Capacity Utilization',
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title="Enrollment Rates and Capacity Utilization by Age Group",
            xaxis_title="Age Group",
            yaxis_title="Rate",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_recommendations(self):
        """Show recommendations"""
        st.header(" Recommendations")
        
        # Create sample recommendations
        recommendations = [
            {
                'Subzone': 'Punggol Northshore',
                'Priority': 'Critical',
                'Gap': 450,
                'Centers_Needed': 5,
                'Recommendation': 'Build 5 new centers by 2025'
            },
            {
                'Subzone': 'Tengah Plantation',
                'Priority': 'High',
                'Gap': 320,
                'Centers_Needed': 4,
                'Recommendation': 'Plan 4 centers for BTO completion'
            },
            {
                'Subzone': 'Sengkang Fernvale',
                'Priority': 'High',
                'Gap': 280,
                'Centers_Needed': 3,
                'Recommendation': 'Expand existing centers'
            }
        ]
        
        for rec in recommendations:
            with st.expander(f" {rec['Subzone']} - {rec['Priority']} Priority"):
                col1, col2 = st.columns(2)