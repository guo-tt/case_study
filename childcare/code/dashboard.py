"""
Gradio dashboard for preschool demand forecasting system
"""

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
import numpy as np
import sys
import os
import json

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import PreschoolForecastSystem
from config import SYSTEM_CONFIG

class GradioDashboard:
    def __init__(self):
        """Initialize Gradio dashboard"""
        self.forecast_system = None
        self.current_data = None
        self.forecast_cache = None
        
        # Load forecast data from JSON file
        self._load_forecast_data()
        
    def _load_forecast_data(self):
        """Load forecast data from JSON file"""
        try:
            json_file = 'forecast_data.json'
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    self.forecast_cache = json.load(f)
                print(f"✅ Loaded forecast data from {json_file}")
                print(f"📊 Available scenarios: {self.forecast_cache['metadata']['total_scenarios']}")
            else:
                print(f"⚠️ Forecast data file {json_file} not found. Will use real-time forecasting.")
                self.forecast_cache = None
        except Exception as e:
            print(f"❌ Error loading forecast data: {e}")
            self.forecast_cache = None    
    
    def create_dashboard(self):
        """Create the Gradio dashboard interface"""
        
        with gr.Blocks(title="ECDA Preschool Demand Forecasting", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🏫 ECDA Preschool Demand Forecasting System")
            gr.Markdown("---")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Main content area
                    overview_tab = gr.Tab("📋 Overview")
                    analysis_tab = gr.Tab("📊 Analysis")
                    recommendations_tab = gr.Tab(" Recommendations")
                    map_tab = gr.Tab("🗺️ Geographic View")
                    
                    with overview_tab:
                        self._create_overview_section()
                    
                    with analysis_tab:
                        self._create_analysis_section()
                    
                    with recommendations_tab:
                        self._create_recommendations_section()
                    
                    with map_tab:
                        self._create_map_section()
                
                with gr.Column(scale=1):
                    # Sidebar controls
                    self._create_sidebar_controls()
            
                        # Event handlers
            self._setup_event_handlers(demo)
        
        return demo
    
    def _create_sidebar_controls(self):
        """Create sidebar controls"""
        gr.Markdown("### 🎛️ Dashboard Controls")
        
        # Year selector
        self.year_selector = gr.Dropdown(
            choices=[2023, 2024, 2025, 2026, 2027],
            value=2025,
            label="Forecast Year"
        )
        
        # Region filter
        self.region_filter = gr.Dropdown(
            choices=["All Regions", "Central Region", "East Region", "North Region", 
                    "North-East Region", "West Region"],
            value="All Regions",
            label="Filter by Region"
        )
        
        # Priority filter
        self.priority_filter = gr.Dropdown(
            choices=["All Priorities", "Critical Priority", "High Priority", 
                    "Medium Priority", "Low Priority"],
            value="All Priorities",
            label="Filter by Priority"
        )
        
        # Scenario controls
        gr.Markdown("### 📈 Scenario Analysis")
        
        self.enrollment_rate = gr.Slider(
            minimum=60,
            maximum=100,
            value=80,
            step=5,
            label="Enrollment Rate (%)"
        )
        
        self.bto_impact = gr.Slider(
            minimum=0.5,
            maximum=2.0,
            value=1.0,
            step=0.1,
            label="BTO Impact Multiplier"
        )
        
        # Forecast method switch
        self.use_lstm = gr.Checkbox(label="Use LSTM (Deep Learning)", value=False)
        
        # Update button
        self.update_btn = gr.Button("🔄 Update Forecast", variant="primary")
        
        # Status message
        self.status_msg = gr.Textbox(label="Status", interactive=False)
    
    def _create_overview_section(self):
        """Create overview section"""
        gr.Markdown("## 📋 Overview")
        
        # Summary metrics
        with gr.Row():
            self.total_subzones = gr.Number(label="Total Subzones", interactive=False)
            self.subzones_with_gaps = gr.Number(label="Subzones with Gaps", interactive=False)
            self.total_capacity_gap = gr.Number(label="Total Capacity Gap", interactive=False)
            self.centers_needed = gr.Number(label="Centers Needed", interactive=False)
        
        # Charts
        with gr.Row():
            with gr.Column():
                self.priority_chart = gr.Plot(label="Priority Distribution")
            with gr.Column():
                self.regional_chart = gr.Plot(label="Capacity Gap by Region")
    
    def _create_analysis_section(self):
        """Create analysis section"""
        gr.Markdown("## 📊 Detailed Analysis")
        
        # Analysis tabs
        analysis_tabs = gr.Tabs(["📈 Trends", "🏘️ BTO Impact", "🎓 Enrollment Patterns"])
        
        with analysis_tabs:
            with gr.TabItem("📈 Trends"):
                self.trends_chart = gr.Plot(label="Population and Demand Trends")
            
            with gr.TabItem("🏘️ BTO Impact"):
                self.bto_chart = gr.Plot(label="BTO Project Impact Analysis")
            
            with gr.TabItem("🎓 Enrollment Patterns"):
                self.enrollment_chart = gr.Plot(label="Enrollment Analysis")
    
    def _create_recommendations_section(self):
        """Create recommendations section"""
        gr.Markdown("##  Recommendations")
        
        # Top recommendations table
        self.recommendations_table = gr.Dataframe(
            headers=["Subzone", "Priority", "Gap", "Centers Needed", "Recommendation"],
            label="Top Priority Areas"
        )
        
        # Detailed recommendations
        self.detailed_recommendations = gr.Markdown("Loading recommendations...")
    
    def _create_map_section(self):
        """Create map section"""
        gr.Markdown("## 🗺️ Geographic Distribution")
        
        # Map placeholder
        self.map_display = gr.HTML(label="Interactive Map")
        
        # Map controls
        with gr.Row():
            self.map_metric = gr.Dropdown(
                choices=["Capacity Gap", "Demand Forecast", "Current Capacity"],
                value="Capacity Gap",
                label="Map Metric"
            )
            self.map_region = gr.Dropdown(
                choices=["All Regions", "Central Region", "East Region", "North Region", 
                        "North-East Region", "West Region"],
                value="All Regions",
                label="Map Region"
            )
    

    
    def _setup_event_handlers(self, demo):
        """Setup event handlers for interactive elements"""
        
        # Update forecast button
        self.update_btn.click(
            fn=self._update_forecast,
            inputs=[self.year_selector, self.region_filter, self.priority_filter,
                   self.enrollment_rate, self.bto_impact, self.use_lstm],
            outputs=[self.status_msg, self.total_subzones, self.subzones_with_gaps,
                    self.total_capacity_gap, self.centers_needed, self.priority_chart,
                    self.regional_chart, self.trends_chart, self.bto_chart,
                    self.enrollment_chart, self.recommendations_table,
                    self.detailed_recommendations, self.map_display]
        )
        
        # Auto-load initial data when dashboard starts
        demo.load(
            fn=self._update_forecast,
            inputs=[self.year_selector, self.region_filter, self.priority_filter,
                   self.enrollment_rate, self.bto_impact, self.use_lstm],
            outputs=[self.status_msg, self.total_subzones, self.subzones_with_gaps,
                    self.total_capacity_gap, self.centers_needed, self.priority_chart,
                    self.regional_chart, self.trends_chart, self.bto_chart,
                    self.enrollment_chart, self.recommendations_table,
                    self.detailed_recommendations, self.map_display]
        )
        
        # Filter changes
        self.region_filter.change(
            fn=self._filter_data,
            inputs=[self.region_filter, self.priority_filter],
            outputs=[self.recommendations_table, self.detailed_recommendations]
        )
        
        self.priority_filter.change(
            fn=self._filter_data,
            inputs=[self.region_filter, self.priority_filter],
            outputs=[self.recommendations_table, self.detailed_recommendations]
        )
        
        # Map controls
        self.map_metric.change(
            fn=self._update_map,
            inputs=[self.map_metric, self.map_region],
            outputs=[self.map_display]
        )
        
        self.map_region.change(
            fn=self._update_map,
            inputs=[self.map_metric, self.map_region],
            outputs=[self.map_display]
        )
    
    def _update_forecast(self, year, region, priority, enrollment_rate, bto_impact, use_lstm):
        """Update forecast based on user inputs"""
        try:
            # Get data from cache or run real-time forecast
            results = self._get_forecast_data(year, enrollment_rate, bto_impact, use_lstm)
            
            if results is None:
                # Fallback to real-time forecasting if cache is not available
                if self.forecast_system is None:
                    self.forecast_system = PreschoolForecastSystem()
                    self.forecast_system.load_data()
                results = self.forecast_system.forecast_demand(target_year=year, use_lstm=use_lstm)
            self.current_data = results
            
            # Calculate summary statistics
            total_subzones = len(results)
            subzones_with_gaps = len(results[results['Capacity_gap'] > 0])
            total_capacity_gap = results['Capacity_gap'].sum()
            centers_needed = results['Centers_needed'].sum()
            
            # Create charts
            priority_chart = self._create_priority_chart(results)
            regional_chart = self._create_regional_chart(results)
            trends_chart = self._create_trends_chart()
            bto_chart = self._create_bto_chart()
            enrollment_chart = self._create_enrollment_chart()
            
            # Create recommendations
            recommendations_table = self._create_recommendations_table(results)
            detailed_recommendations = self._create_detailed_recommendations(results)
            
            # Create map
            map_html = self._create_map(results, "Capacity Gap", "All Regions")
            
            status_msg = f"✅ Forecast updated successfully for {year}"
            
            return (status_msg, total_subzones, subzones_with_gaps, total_capacity_gap,
                   centers_needed, priority_chart, regional_chart, trends_chart,
                   bto_chart, enrollment_chart, recommendations_table,
                   detailed_recommendations, map_html)
        
        except Exception as e:
            # Create empty charts and data for error case
            empty_chart = go.Figure().add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            
            return (f"❌ Error updating forecast: {str(e)}", 0, 0, 0, 0,
                   empty_chart, empty_chart, empty_chart, empty_chart, empty_chart, 
                   pd.DataFrame(), "No recommendations available", "<p>No data available for map</p>")
    
    def _filter_data(self, region, priority):
        """Filter data based on region and priority"""
        if self.current_data is None:
            return None, "No data available"
        
        filtered_data = self.current_data.copy()
        
        # Apply region filter
        if region != "All Regions":
            filtered_data = filtered_data[filtered_data['Region'] == region]
        
        # Apply priority filter
        if priority != "All Priorities":
            # Add priority levels to data
            filtered_data['Priority_level'] = filtered_data['Capacity_gap'].apply(
                self._assign_priority_level
            )
            filtered_data = filtered_data[filtered_data['Priority_level'] == priority]
        
        # Create recommendations table
        recommendations_table = self._create_recommendations_table(filtered_data)
        detailed_recommendations = self._create_detailed_recommendations(filtered_data)
        
        return recommendations_table, detailed_recommendations
    
    def _update_map(self, metric, region):
        """Update map display"""
        if self.current_data is None:
            return "<p>No data available for map</p>"
        
        return self._create_map(self.current_data, metric, region)
    
    def _create_priority_chart(self, data):
        """Create priority distribution chart"""
        # Add priority levels
        data_copy = data.copy()
        data_copy['Priority_level'] = data_copy['Capacity_gap'].apply(self._assign_priority_level)
        
        priority_counts = data_copy['Priority_level'].value_counts()
        
        fig = px.pie(
            values=priority_counts.values,
            names=priority_counts.index,
            title="Priority Distribution"
        )
        
        return fig
    
    def _create_regional_chart(self, data):
        """Create regional capacity gap chart"""
        regional_gaps = data.groupby('Region')['Capacity_gap'].sum()
        
        fig = px.bar(
            x=regional_gaps.index,
            y=regional_gaps.values,
            title="Capacity Gap by Region",
            labels={'x': 'Region', 'y': 'Capacity Gap'}
        )
        
        return fig
    
    def _create_trends_chart(self):
        """Create trends analysis chart"""
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
        
        return fig
    
    def _create_bto_chart(self):
        """Create BTO impact analysis chart"""
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
        
        return fig
    
    def _create_enrollment_chart(self):
        """Create enrollment analysis chart"""
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
        
        return fig
    
    def _create_recommendations_table(self, data):
        """Create recommendations table"""
        if data is None or len(data) == 0:
            return pd.DataFrame()
        
        # Sort by capacity gap and get top 10
        top_recommendations = data.nlargest(10, 'Capacity_gap')
        
        # Create recommendations
        recommendations = []
        for _, row in top_recommendations.iterrows():
            recommendation = self._generate_recommendation(row)
            recommendations.append([
                row['Subzone'],
                self._assign_priority_level(row['Capacity_gap']),
                int(row['Capacity_gap']),
                int(row['Centers_needed']),
                recommendation
            ])
        
        return pd.DataFrame(
            recommendations,
            columns=["Subzone", "Priority", "Gap", "Centers Needed", "Recommendation"]
        )
    
    def _create_detailed_recommendations(self, data):
        """Create detailed recommendations markdown"""
        if data is None or len(data) == 0:
            return "No recommendations available"
        
        # Get top 5 recommendations
        top_5 = data.nlargest(5, 'Capacity_gap')
        
        markdown = "## Top 5 Priority Areas\n\n"
        
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            priority = self._assign_priority_level(row['Capacity_gap'])
            recommendation = self._generate_recommendation(row)
            
            markdown += f"### {i}. {row['Subzone']} ({priority})\n"
            markdown += f"- **Capacity Gap:** {int(row['Capacity_gap']):,} children\n"
            markdown += f"- **Centers Needed:** {int(row['Centers_needed'])}\n"
            markdown += f"- **Recommendation:** {recommendation}\n\n"
        
        return markdown
    
    def _create_map(self, data, metric, region):
        """Create interactive map"""
        if data is None:
            return "<p>No data available for map</p>"
        
        # Filter by region if specified
        if region != "All Regions":
            data = data[data['Region'] == region]
        
        # Map metric names to column names
        metric_mapping = {
            "Capacity Gap": "Capacity_gap",
            "Demand Forecast": "Demand_forecast", 
            "Current Capacity": "Current_capacity"
        }
        
        column_name = metric_mapping.get(metric, metric.replace(' ', '_').lower())
        
        # Create a simple HTML map representation
        # In a real implementation, this would use folium or similar
        map_html = f"""
        <div style="width: 100%; height: 400px; background: #f0f0f0; border: 1px solid #ccc; display: flex; align-items: center; justify-content: center;">
            <div style="text-align: center;">
                <h3>Interactive Map</h3>
                <p>Showing {metric} for {region}</p>
                <p>Data points: {len(data)} subzones</p>
                <p>Total {metric.lower()}: {data[column_name].sum():,.0f}</p>
            </div>
        </div>
        """
        
        return map_html
    
    def _assign_priority_level(self, gap):
        """Assign priority level based on capacity gap"""
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
    
    def _generate_recommendation(self, row):
        """Generate recommendation based on data row"""
        gap = row['Capacity_gap']
        centers_needed = row['Centers_needed']
        
        if gap <= 0:
            return "No action needed - sufficient capacity"
        elif centers_needed == 1:
            return f"Build 1 new center to accommodate {int(gap)} children"
        else:
            return f"Build {int(centers_needed)} new centers to accommodate {int(gap)} children"
    
    def _get_forecast_data(self, year, enrollment_rate, bto_multiplier, use_lstm=False):
        """Get forecast data for specific parameters"""
        if self.forecast_cache is None:
            # Fallback to real-time forecasting
            if self.forecast_system is None:
                self.forecast_system = PreschoolForecastSystem()
                self.forecast_system.load_data()
            return self.forecast_system.forecast_demand(target_year=year, use_lstm=use_lstm)
        
        # Convert keys to string to match JSON
        year_key = str(year)
        enrollment_key = str(int(enrollment_rate))
        bto_key = str(float(bto_multiplier))
        try:
            year_data = self.forecast_cache['forecasts'].get(year_key, {})
            enrollment_data = year_data.get(enrollment_key, {})
            scenario_data = enrollment_data.get(bto_key, {})
            
            if scenario_data and 'data' in scenario_data:
                return pd.DataFrame(scenario_data['data'])
            else:
                print(f"⚠️ No cached data for Year={year}, Enrollment={enrollment_rate}%, BTO={bto_multiplier}")
                return None
        except Exception as e:
            print(f"❌ Error retrieving cached data: {e}")
            return None

def launch_dashboard():
    """Launch the Gradio dashboard"""
    dashboard = GradioDashboard()
    demo = dashboard.create_dashboard()
    
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    launch_dashboard() 