#!/usr/bin/env python3
"""
Preschool Demand Forecasting System for ECDA
============================================

This system forecasts subzone-level demand for preschool services (18 months to 6 years)
over the next 5 years to help ECDA prioritize building/relocating preschools.

Key Features:
- Population-based demand forecasting
- BTO project impact analysis
- Current preschool capacity mapping
- Gap analysis and recommendations
- Interactive dashboard for decision making
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from datetime import datetime, timedelta
import warnings
import logging
warnings.filterwarnings('ignore')

class PreschoolForecastSystem:
    def __init__(self):
        """Initialize the forecasting system with data sources"""
        self.population_data = None
        self.bto_data = None
        self.preschool_data = None
        self.forecast_results = None
        # Set up logging
        self.logger = logging.getLogger('PreschoolForecastSystem')
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def load_data(self):
        """Load all required datasets"""
        self.logger.info("Loading datasets...")
        
        # Load BTO mapping data
        self.bto_data = pd.read_csv('../data/btomapping.csv')
        self.logger.info(f"Loaded BTO data: {len(self.bto_data)} projects")
        
        # Load preschool centers data
        self.preschool_data = pd.read_csv('../data/ListingofCentres.csv')
        self.logger.info(f"Loaded preschool data: {len(self.preschool_data)} centers")
        
        # Load population data from Excel file
        self._load_population_data()
        
        self.logger.info("All datasets loaded successfully!")
        
    def _load_population_data(self):
        """Load population data from Excel file"""
        self.logger.info("Loading population data from Excel file...")
        
        # The Excel file has multiple sheets with different year ranges
        sheet_names = ['2000', '2001-2010', '2011-2019', '2020']
        all_data = []
        
        for sheet_name in sheet_names:
            self.logger.info(f"Reading sheet: {sheet_name}")
            
            try:
                # Read the sheet without header first to get the correct headers
                df_raw = pd.read_excel('../data/respopagesex2000to2020e.xlsx', sheet_name=sheet_name, header=None)
                
                # Get headers from row 2 (index 2)
                headers = df_raw.iloc[2].tolist()
                
                # Read the sheet again starting from row 3 (index 3) with the correct headers
                sheet_df = pd.read_excel('../data/respopagesex2000to2020e.xlsx', sheet_name=sheet_name, header=None, skiprows=3)
                sheet_df.columns = headers
                
                self.logger.info(f"Sheet {sheet_name} shape: {sheet_df.shape}")
                self.logger.info(f"Sheet {sheet_name} columns: {list(sheet_df.columns)}")
                
                # Clean up column names - remove any leading/trailing whitespace
                sheet_df.columns = [str(col).strip() if pd.notna(col) else str(col) for col in sheet_df.columns]
                
                # Convert Age column to numeric, handling non-numeric values
                sheet_df['Age'] = pd.to_numeric(sheet_df['Age'], errors='coerce')
                
                # Filter for children aged 0-6 years and Total sex
                children_data = sheet_df[
                    (sheet_df['Age'].isin([0, 1, 2, 3, 4, 5, 6])) & 
                    (sheet_df['Sex'] == 'Total') &
                    (sheet_df['Planning Area'] != 'Total') &
                    (sheet_df['Subzone'] != 'Total')
                ].copy()
                
                if len(children_data) > 0:
                    # Melt the data to get years as rows
                    id_vars = ['Planning Area', 'Subzone', 'Age', 'Sex']
                    value_vars = [col for col in children_data.columns if str(col).isdigit() and 2000 <= int(col) <= 2020]
                    
                    if value_vars:
                        melted_data = children_data.melt(
                            id_vars=id_vars,
                            value_vars=value_vars,
                            var_name='Year',
                            value_name='Population'
                        )
                        
                        # Clean up year column
                        melted_data['Year'] = melted_data['Year'].astype(int)
                        
                        # Convert population to numeric, handling any non-numeric values
                        melted_data['Population'] = pd.to_numeric(melted_data['Population'], errors='coerce').fillna(0)
                        
                        all_data.append(melted_data)
                        self.logger.info(f"Added {len(melted_data)} records from sheet {sheet_name}")
                    else:
                        self.logger.warning(f"No year columns found in sheet {sheet_name}")
                else:
                    self.logger.warning(f"No children data found in sheet {sheet_name}")
                    
            except Exception as e:
                self.logger.error(f"Error reading sheet {sheet_name}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data could be extracted from any sheet")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"Combined data shape: {combined_data.shape}")
        
        # Group by subzone and year to get total population for ages 0-6
        population_by_subzone = combined_data.groupby(['Planning Area', 'Subzone', 'Year'])['Population'].sum().reset_index()
        
        # Map planning areas to regions (based on Singapore's planning regions)
        region_mapping = {
            'Ang Mo Kio': 'North-East Region',
            'Bedok': 'East Region',
            'Bishan': 'Central Region',
            'Boon Lay/Pioneer': 'West Region',
            'Bukit Batok': 'West Region',
            'Bukit Merah': 'Central Region',
            'Bukit Panjang': 'West Region',
            'Bukit Timah': 'Central Region',
            'Central Water Catchment': 'Central Region',
            'Changi': 'East Region',
            'Changi Bay': 'East Region',
            'Choa Chu Kang': 'West Region',
            'Clementi': 'West Region',
            'Downtown Core': 'Central Region',
            'Geylang': 'East Region',
            'Hougang': 'North-East Region',
            'Jurong East': 'West Region',
            'Jurong West': 'West Region',
            'Kallang': 'Central Region',
            'Lim Chu Kang': 'West Region',
            'Mandai': 'North Region',
            'Marina East': 'Central Region',
            'Marina South': 'Central Region',
            'Marine Parade': 'East Region',
            'Museum': 'Central Region',
            'Newton': 'Central Region',
            'North-Eastern Islands': 'East Region',
            'Novena': 'Central Region',
            'Orchard': 'Central Region',
            'Outram': 'Central Region',
            'Pasir Ris': 'East Region',
            'Paya Lebar': 'East Region',
            'Pioneer': 'West Region',
            'Punggol': 'North-East Region',
            'Queenstown': 'Central Region',
            'River Valley': 'Central Region',
            'Rochor': 'Central Region',
            'Seletar': 'North-East Region',
            'Sembawang': 'North Region',
            'Sengkang': 'North-East Region',
            'Serangoon': 'North-East Region',
            'Simpang': 'North Region',
            'Singapore River': 'Central Region',
            'Southern Islands': 'Central Region',
            'Straits View': 'East Region',
            'Sungei Kadut': 'North Region',
            'Tampines': 'East Region',
            'Tanglin': 'Central Region',
            'Tengah': 'West Region',
            'Toa Payoh': 'Central Region',
            'Tuas': 'West Region',
            'Western Islands': 'West Region',
            'Western Water Catchment': 'West Region',
            'Woodlands': 'North Region',
            'Yishun': 'North Region'
        }
        
        population_by_subzone['Region'] = population_by_subzone['Planning Area'].map(region_mapping)
        
        # Rename columns to match expected format
        population_by_subzone = population_by_subzone.rename(columns={
            'Planning Area': 'Planning_area',
            'Population': 'Population_18m_6y'
        })
        
        self.population_data = population_by_subzone
        self.logger.info(f"Loaded population data: {len(self.population_data)} records")
        self.logger.info(f"Years available: {sorted(self.population_data['Year'].unique())}")
        self.logger.info(f"Subzones available: {len(self.population_data['Subzone'].unique())}")
        
    def analyze_current_capacity(self):
        """Analyze current preschool capacity by subzone"""
        self.logger.info("Analyzing current preschool capacity...")
        
        # Extract postal code and map to subzones
        # For demo, we'll create a mapping based on available data
        capacity_by_subzone = {}
        
        # Count centers by subzone (simplified mapping)
        for _, center in self.preschool_data.iterrows():
            # Extract subzone from address (simplified approach)
            address = str(center['centre_address'])
            
            # Find matching subzone from BTO data
            for _, bto in self.bto_data.iterrows():
                if bto['Subzone'].lower() in address.lower():
                    subzone = bto['Subzone']
                    if subzone not in capacity_by_subzone:
                        capacity_by_subzone[subzone] = 0
                    capacity_by_subzone[subzone] += 100  # Assume 100 children per center
                    break
        
        return capacity_by_subzone
    
    def forecast_population_trend(self, subzone_data, target_year):
        """Forecast population for a subzone based on historical trends"""
        # Get historical data for this subzone
        # subzone_data = self.population_data[self.population_data['Subzone'] == subzone].copy()
        
        if len(subzone_data) < 3:  # Need at least 3 data points for trend
            return None
        
        # Sort by year
        subzone_data = subzone_data.sort_values('Year')
        
        # Calculate trend using linear regression
        years = subzone_data['Year'].values
        population = subzone_data['Population_18m_6y'].values
        
        # Simple linear regression: y = mx + b
        n = len(years)
        sum_x = np.sum(years)
        sum_y = np.sum(population)
        sum_xy = np.sum(years * population)
        sum_x2 = np.sum(years * years)
        
        # Calculate slope and intercept
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Forecast population for target year
        forecasted_population = slope * target_year + intercept
        
        # Ensure non-negative population
        return max(0, int(forecasted_population))

    def forecast_population_lstm(self, subzone_data, target_year):
        import numpy as np
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout
        from keras.callbacks import EarlyStopping
        from sklearn.preprocessing import MinMaxScaler

        try:
            years = subzone_data['Year'].values
            population = subzone_data['Population_18m_6y'].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            population_scaled = scaler.fit_transform(population)

            # Prepare sequences for LSTM
            X, y = [], []
            for i in range(len(population_scaled) - 10):
                X.append(population_scaled[i:i+10])
                y.append(population_scaled[i+10])
            X, y = np.array(X), np.array(y)

            if len(X) == 0:
                self.logger.warning(f"Not enough data for LSTM in subzone. Falling back to linear regression.")
                return self.forecast_population_trend(subzone_data, target_year)

            X = X.reshape((X.shape[0], X.shape[1], 1))

            # Build smaller, faster LSTM model
            model = Sequential()
            model.add(LSTM(32, activation='relu', return_sequences=True, input_shape=(10, 1)))
            model.add(Dropout(0.1))
            model.add(LSTM(16, activation='relu'))
            model.add(Dropout(0.1))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')

            # Train model with aggressive early stopping and fewer epochs
            early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            model.fit(X, y, epochs=30, verbose=0, validation_split=0.2, callbacks=[early_stop])

            # Make predictions
            last_seq = population_scaled[-10:]
            for year in range(years[-1]+1, target_year+1):
                pred = model.predict(last_seq.reshape(1, 10, 1), verbose=0)
                last_seq = np.append(last_seq[1:], pred, axis=0)

            forecasted = scaler.inverse_transform(last_seq[-1].reshape(-1, 1))[0, 0]
            self.logger.info(f"LSTM forecast for subzone in {target_year}: {forecasted:.1f}")
            return max(0, int(forecasted))

        except ImportError as e:
            self.logger.error(f"Keras import failed: {e}. Please install tensorflow and keras.")
            raise e
        except Exception as e:
            self.logger.error(f"LSTM forecasting failed: {e}. Falling back to linear regression.")
            return self.forecast_population_trend(subzone_data, target_year)

    def forecast_population(self, subzone, target_year, use_lstm=False):
        subzone_data = self.population_data[self.population_data['Subzone'] == subzone].sort_values('Year')
        n_points = len(subzone_data)
        try:
            if use_lstm and n_points > 10:
                self.logger.info(f"Using LSTM for {subzone} (data points: {n_points})")
                return self.forecast_population_lstm(subzone_data, target_year)
            else:
                self.logger.info(f"Using linear regression for {subzone} (data points: {n_points})")
                return self.forecast_population_trend(subzone_data, target_year)
        except Exception as e:
            self.logger.error(f"Forecasting failed for {subzone}: {e}. Returning None.")
            return None
    
    def forecast_demand(self, target_year=2025, use_lstm=False):
        """Forecast preschool demand for target year"""
        self.logger.info(f"Forecasting demand for {target_year}...")
        
        # Check available years in population data
        available_years = sorted(self.population_data['Year'].unique())
        self.logger.info(f"Available years in population data: {available_years}")
        
        # Get current capacity
        current_capacity = self.analyze_current_capacity()
        
        # Get all unique subzones
        all_subzones = self.population_data[['Subzone', 'Planning_area', 'Region']].drop_duplicates()
        self.logger.info(f"Total unique subzones: {len(all_subzones)}")
        
        # Calculate demand (assuming 80% enrollment rate)
        enrollment_rate = 0.8
        demand_forecast = []
        
        for _, subzone_info in all_subzones.iterrows():
            subzone = subzone_info['Subzone']
            planning_area = subzone_info['Planning_area']
            region = subzone_info['Region']
            
            # Check if we have data for target year
            target_year_data = self.population_data[
                (self.population_data['Subzone'] == subzone) & 
                (self.population_data['Year'] == target_year)
            ]
            
            if len(target_year_data) > 0:
                # Use actual data for target year
                population = target_year_data.iloc[0]['Population_18m_6y']
                self.logger.info(f"Using actual data for {subzone} in {target_year}: {population}")
            else:
                # Forecast population based on historical trends
                population = self.forecast_population(subzone, target_year, use_lstm=use_lstm)
                if population is None:
                    # If forecasting fails, use the latest available year's data
                    latest_data = self.population_data[self.population_data['Subzone'] == subzone]
                    if len(latest_data) > 0:
                        latest_year = latest_data['Year'].max()
                        population = latest_data[latest_data['Year'] == latest_year].iloc[0]['Population_18m_6y']
                        self.logger.info(f"Using latest available data for {subzone} ({latest_year}): {population}")
                    else:
                        self.logger.warning(f"No data available for {subzone}, skipping...")
                        continue
                else:
                    self.logger.info(f"Forecasted population for {subzone} in {target_year}: {population}")
            
            demand = int(population * enrollment_rate)
            current_cap = current_capacity.get(subzone, 0)
            
            demand_forecast.append({
                'Subzone': subzone,
                'Planning_area': planning_area,
                'Region': region,
                'Population_18m_6y': population,
                'Demand_forecast': demand,
                'Current_capacity': current_cap,
                'Capacity_gap': demand - current_cap,
                'Centers_needed': max(0, (demand - current_cap) // 100 + 1)
            })
        
        self.forecast_results = pd.DataFrame(demand_forecast)
        self.logger.info(f"Created forecast results with {len(self.forecast_results)} records")
        return self.forecast_results
    
    def analyze_bto_impact(self, target_year=2025):
        """Analyze impact of BTO projects on demand"""
        self.logger.info("Analyzing BTO project impact...")
        
        # Get BTO projects completing by target year
        bto_impact = self.bto_data[self.bto_data['Estimated completion year'] <= target_year].copy()
        
        # Estimate additional demand from BTO projects
        # Assume 2.5 persons per unit, 15% are children 18m-6y
        bto_impact['Additional_population'] = bto_impact['Total number of units'] * 2.5 * 0.15
        bto_impact['Additional_demand'] = bto_impact['Additional_population'] * 0.8  # 80% enrollment
        
        return bto_impact
    
    def analyze_forecasting_accuracy(self):
        """Analyze the accuracy of population forecasting"""
        self.logger.info("Analyzing forecasting accuracy...")
        
        # Get the latest available year for validation
        latest_year = max(self.population_data['Year'].unique())
        validation_year = latest_year - 5  # Use 5 years ago for validation
        
        if validation_year < min(self.population_data['Year'].unique()):
            self.logger.warning("Not enough historical data for validation")
            return None
        
        # Get actual data for validation year
        actual_data = self.population_data[self.population_data['Year'] == validation_year]
        
        # Forecast for validation year using data up to 5 years before
        forecast_data = []
        for _, row in actual_data.iterrows():
            subzone = row['Subzone']
            actual_population = row['Population_18m_6y']
            
            # Get historical data up to 5 years before validation year
            historical_data = self.population_data[
                (self.population_data['Subzone'] == subzone) & 
                (self.population_data['Year'] < validation_year)
            ]
            
            if len(historical_data) >= 3:
                # Forecast using historical data
                forecasted_population = self.forecast_population_trend(historical_data, validation_year)
                if forecasted_population is not None:
                    forecast_data.append({
                        'Subzone': subzone,
                        'Actual': actual_population,
                        'Forecasted': forecasted_population,
                        'Error': abs(actual_population - forecasted_population),
                        'Error_Percentage': abs(actual_population - forecasted_population) / actual_population * 100 if actual_population > 0 else 0
                    })
        
        if forecast_data:
            accuracy_df = pd.DataFrame(forecast_data)
            mean_error = accuracy_df['Error'].mean()
            mean_error_percentage = accuracy_df['Error_Percentage'].mean()
            
            self.logger.info(f"Forecasting Accuracy Analysis:")
            self.logger.info(f"Validation Year: {validation_year}")
            self.logger.info(f"Number of subzones tested: {len(accuracy_df)}")
            self.logger.info(f"Mean Absolute Error: {mean_error:.1f} children")
            self.logger.info(f"Mean Error Percentage: {mean_error_percentage:.1f}%")
            
            return accuracy_df
        else:
            self.logger.warning("No data available for accuracy analysis")
            return None
    
    def generate_recommendations(self):
        """Generate recommendations for preschool development"""
        self.logger.info("Generating recommendations...")
        
        if self.forecast_results is None:
            self.forecast_demand()
        
        # Debug: Check what columns are available
        self.logger.info(f"Forecast results columns: {list(self.forecast_results.columns)}")
        self.logger.info(f"Forecast results shape: {self.forecast_results.shape}")
        if len(self.forecast_results) > 0:
            self.logger.info(f"First row: {self.forecast_results.iloc[0].to_dict()}")
        
        # Check if forecast results are empty
        if self.forecast_results.empty:
            self.logger.warning("Warning: No forecast results available. Returning empty DataFrame.")
            return pd.DataFrame()
        
        # Sort by capacity gap (highest need first)
        recommendations = self.forecast_results.sort_values('Capacity_gap', ascending=False)
        
        # Filter for areas with significant gaps
        high_priority = recommendations[recommendations['Capacity_gap'] > 50]
        
        return high_priority
    
    def create_dashboard(self):
        """Create an interactive dashboard for decision making"""
        self.logger.info("Creating dashboard...")
        
        # Check if forecast results are available
        if self.forecast_results is None or self.forecast_results.empty:
            self.logger.warning("Warning: No forecast results available for dashboard.")
            return {
                'Total_Subzones': 0,
                'Subzones_with_Gaps': 0,
                'Total_Capacity_Gap': 0,
                'Total_Centers_Needed': 0
            }
        
        # Create summary statistics
        summary_stats = {
            'Total_Subzones': len(self.forecast_results),
            'Subzones_with_Gaps': len(self.forecast_results[self.forecast_results['Capacity_gap'] > 0]),
            'Total_Capacity_Gap': self.forecast_results['Capacity_gap'].sum(),
            'Total_Centers_Needed': self.forecast_results['Centers_needed'].sum()
        }
        
        # Create visualizations
        self._create_visualizations()
        
        return summary_stats
    
    def _create_visualizations(self):
        """Create visualizations for the dashboard"""
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Capacity gap by region
        region_gaps = self.forecast_results.groupby('Region')['Capacity_gap'].sum()
        axes[0, 0].bar(region_gaps.index, region_gaps.values)
        axes[0, 0].set_title('Capacity Gap by Region')
        axes[0, 0].set_ylabel('Capacity Gap')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Top 10 subzones with highest gaps
        top_gaps = self.forecast_results.nlargest(10, 'Capacity_gap')
        axes[0, 1].barh(top_gaps['Subzone'], top_gaps['Capacity_gap'])
        axes[0, 1].set_title('Top 10 Subzones with Highest Capacity Gaps')
        axes[0, 1].set_xlabel('Capacity Gap')
        
        # 3. Demand vs Capacity scatter plot
        axes[1, 0].scatter(self.forecast_results['Current_capacity'], 
                          self.forecast_results['Demand_forecast'])
        axes[1, 0].plot([0, self.forecast_results['Current_capacity'].max()], 
                       [0, self.forecast_results['Current_capacity'].max()], 'r--')
        axes[1, 0].set_xlabel('Current Capacity')
        axes[1, 0].set_ylabel('Demand Forecast')
        axes[1, 0].set_title('Demand vs Current Capacity')
        
        # 4. Distribution of capacity gaps
        axes[1, 1].hist(self.forecast_results['Capacity_gap'], bins=20, edgecolor='black')
        axes[1, 1].set_xlabel('Capacity Gap')
        axes[1, 1].set_ylabel('Number of Subzones')
        axes[1, 1].set_title('Distribution of Capacity Gaps')
        
        plt.tight_layout()
        plt.savefig('preschool_forecast_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_population_trends(self, sample_subzones=5):
        """Plot population trends for sample subzones"""
        self.logger.info("Creating population trend visualizations...")
        
        # Get sample subzones
        all_subzones = self.population_data['Subzone'].unique()
        sample_subzones = min(sample_subzones, len(all_subzones))
        selected_subzones = np.random.choice(all_subzones, sample_subzones, replace=False)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, subzone in enumerate(selected_subzones):
            if i >= len(axes):
                break
                
            # Get historical data
            subzone_data = self.population_data[self.population_data['Subzone'] == subzone].sort_values('Year')
            
            if len(subzone_data) > 0:
                # Plot historical data
                axes[i].plot(subzone_data['Year'], subzone_data['Population_18m_6y'], 'o-', label='Historical')
                
                # Forecast future values
                years = subzone_data['Year'].values
                population = subzone_data['Population_18m_6y'].values
                
                if len(years) >= 3:
                    # Calculate trend
                    n = len(years)
                    sum_x = np.sum(years)
                    sum_y = np.sum(population)
                    sum_xy = np.sum(years * population)
                    sum_x2 = np.sum(years * years)
                    
                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                    intercept = (sum_y - slope * sum_x) / n
                    
                    # Plot forecast line
                    forecast_years = np.arange(min(years), 2025 + 1)
                    forecast_population = slope * forecast_years + intercept
                    axes[i].plot(forecast_years, forecast_population, '--', label='Forecast')
                
                axes[i].set_title(f'{subzone}')
                axes[i].set_xlabel('Year')
                axes[i].set_ylabel('Population (18m-6y)')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(selected_subzones), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('population_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def export_results(self, filename='preschool_forecast_results.csv'):
        """Export forecast results to CSV"""
        if self.forecast_results is not None:
            self.forecast_results.to_csv(filename, index=False)
            self.logger.info(f"Results exported to {filename}")
        else:
            self.logger.warning("No forecast results to export. Run forecast_demand() first.")

def main():
    """Main function to run the forecasting system"""
    print("=" * 60)
    print("PRESCHOOL DEMAND FORECASTING SYSTEM")
    print("Early Childhood Development Agency (ECDA)")
    print("=" * 60)
    
    # Initialize the system
    forecast_system = PreschoolForecastSystem()
    
    # Load data
    forecast_system.load_data()
    
    # Analyze forecasting accuracy first
    accuracy_results = forecast_system.analyze_forecasting_accuracy()
    
    # Run forecast for 2025
    results = forecast_system.forecast_demand(target_year=2025)
    
    # Analyze BTO impact
    bto_impact = forecast_system.analyze_bto_impact(target_year=2025)
    
    # Generate recommendations
    recommendations = forecast_system.generate_recommendations()
    
    # Create dashboard
    summary_stats = forecast_system.create_dashboard()
    
    # Create population trend visualizations
    forecast_system.plot_population_trends(sample_subzones=6)
    
    # Display results
    print("\n" + "=" * 60)
    print("FORECAST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Subzones Analyzed: {summary_stats['Total_Subzones']}")
    print(f"Subzones with Capacity Gaps: {summary_stats['Subzones_with_Gaps']}")
    print(f"Total Capacity Gap: {summary_stats['Total_Capacity_Gap']:,} children")
    print(f"Total Centers Needed: {summary_stats['Total_Centers_Needed']}")
    
    print("\n" + "=" * 60)
    print("TOP 10 PRIORITY AREAS")
    print("=" * 60)
    print(recommendations[['Subzone', 'Planning_area', 'Region', 'Capacity_gap', 'Centers_needed']].head(10))
    
    # Export results
    forecast_system.export_results()
    
    print("\n" + "=" * 60)
    print("SYSTEM READY FOR REGULAR UPDATES")
    print("=" * 60)
    print("The system can be updated with:")
    print("- New population statistics")
    print("- Updated preschool openings/closings")
    print("- New BTO project announcements")
    print("- Changes in enrollment rates")
    
if __name__ == "__main__":
    main() 