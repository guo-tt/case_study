#!/usr/bin/env python3
"""
Generate comprehensive forecast data for dashboard
"""

import json
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import PreschoolForecastSystem

def generate_forecast_data():
    """Generate forecast data for multiple scenarios"""
    print("=" * 60)
    print("🏫 Generating Comprehensive Forecast Data")
    print("=" * 60)
    
    # Initialize forecast system
    try:
        forecast_system = PreschoolForecastSystem()
        forecast_system.load_data()
        print("✅ Forecast system initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing forecast system: {e}")
        return None
    
    # Define parameter ranges
    years = [2021, 2022, 2023, 2024, 2025]
    enrollment_rates = list(range(60, 101, 5))  # 60% to 100%, step 5%
    bto_multipliers = [round(x * 0.1, 1) for x in range(5, 21)]  # 0.5 to 2.0, step 0.1
    
    print(f"📊 Generating forecasts for:")
    print(f"   - Years: {years}")
    print(f"   - Enrollment rates: {enrollment_rates}%")
    print(f"   - BTO multipliers: {bto_multipliers}")
    
    # Store all forecast data
    all_forecasts = {}
    
    # Generate base forecasts for each year
    base_forecasts = {}
    for year in years:
        try:
            print(f"🔄 Generating base forecast for {year}...")
            results = forecast_system.forecast_demand(target_year=year)
            base_forecasts[year] = results
            print(f"✅ Base forecast for {year} completed")
        except Exception as e:
            print(f"❌ Error generating forecast for {year}: {e}")
            continue
    
    # Generate scenario variations
    total_scenarios = len(years) * len(enrollment_rates) * len(bto_multipliers)
    current_scenario = 0
    
    for year in years:
        if year not in base_forecasts:
            continue
            
        all_forecasts[year] = {}
        
        for enrollment_rate in enrollment_rates:
            all_forecasts[year][enrollment_rate] = {}
            
            for bto_multiplier in bto_multipliers:
                current_scenario += 1
                print(f"🔄 Scenario {current_scenario}/{total_scenarios}: Year={year}, Enrollment={enrollment_rate}%, BTO={bto_multiplier}")
                
                try:
                    # Apply enrollment rate adjustment
                    adjusted_results = base_forecasts[year].copy()
                    
                    # Adjust demand based on enrollment rate
                    enrollment_factor = enrollment_rate / 100.0
                    adjusted_results['Demand_forecast'] = adjusted_results['Population_18m_6y'] * enrollment_factor
                    
                    # Recalculate capacity gap
                    adjusted_results['Capacity_gap'] = np.maximum(0, adjusted_results['Demand_forecast'] - adjusted_results['Current_capacity'])
                    
                    # Apply BTO impact multiplier
                    bto_impact = forecast_system.analyze_bto_impact(target_year=year)
                    if bto_impact is not None and len(bto_impact) > 0:
                        # Apply BTO multiplier to capacity gap
                        adjusted_results['Capacity_gap'] = adjusted_results['Capacity_gap'] * bto_multiplier
                    
                    # Recalculate centers needed (assuming 150 children per center)
                    adjusted_results['Centers_needed'] = np.ceil(adjusted_results['Capacity_gap'] / 150).astype(int)
                    
                    # Calculate summary statistics
                    summary = {
                        'total_subzones': len(adjusted_results),
                        'subzones_with_gaps': len(adjusted_results[adjusted_results['Capacity_gap'] > 0]),
                        'total_capacity_gap': adjusted_results['Capacity_gap'].sum(),
                        'total_centers_needed': adjusted_results['Centers_needed'].sum(),
                        'average_capacity_gap': adjusted_results['Capacity_gap'].mean(),
                        'max_capacity_gap': adjusted_results['Capacity_gap'].max(),
                        'min_capacity_gap': adjusted_results['Capacity_gap'].min()
                    }
                    
                    # Store results
                    all_forecasts[year][enrollment_rate][bto_multiplier] = {
                        'summary': summary,
                        'data': adjusted_results.to_dict('records')
                    }
                    
                except Exception as e:
                    print(f"❌ Error in scenario: {e}")
                    continue
    
    # Add metadata
    forecast_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'years': years,
            'enrollment_rates': enrollment_rates,
            'bto_multipliers': bto_multipliers,
            'total_scenarios': total_scenarios,
            'description': 'Comprehensive forecast data for ECDA preschool demand forecasting dashboard'
        },
        'forecasts': all_forecasts
    }
    
    # Save to JSON file
    output_file = 'forecast_data.json'
    try:
        with open(output_file, 'w') as f:
            json.dump(forecast_data, f, indent=2, default=str)
        print(f"✅ Forecast data saved to {output_file}")
        print(f"📁 File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"❌ Error saving forecast data: {e}")
        return None
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 FORECAST DATA SUMMARY")
    print("=" * 60)
    print(f"Years: {len(years)}")
    print(f"Enrollment rates: {len(enrollment_rates)}")
    print(f"BTO multipliers: {len(bto_multipliers)}")
    print(f"Total scenarios: {total_scenarios}")
    print(f"Output file: {output_file}")
    
    return forecast_data

def create_sample_forecast_data():
    """Create sample forecast data for demonstration"""
    print("🔄 Creating sample forecast data...")
    
    # Define parameter ranges
    years = [2021, 2022, 2023, 2024, 2025]
    enrollment_rates = list(range(60, 101, 5))
    bto_multipliers = [round(x * 0.1, 1) for x in range(5, 21)]
    
    # Create sample data
    regions = ["Central Region", "East Region", "North Region", "North-East Region", "West Region"]
    subzones = [f"Subzone_{i:03d}" for i in range(1, 21)]  # 20 subzones for sample
    
    all_forecasts = {}
    
    for year in years:
        all_forecasts[year] = {}
        
        for enrollment_rate in enrollment_rates:
            all_forecasts[year][enrollment_rate] = {}
            
            for bto_multiplier in bto_multipliers:
                # Generate sample data for this scenario
                data = []
                for i, subzone in enumerate(subzones):
                    region = regions[i % len(regions)]
                    base_population = 1000 + (year - 2020) * 50  # Growing population
                    population = base_population + np.random.normal(0, 100)
                    current_capacity = 800 + np.random.normal(0, 150)
                    
                    # Apply enrollment rate
                    demand = population * (enrollment_rate / 100.0)
                    
                    # Apply BTO multiplier
                    capacity_gap = max(0, demand - current_capacity) * bto_multiplier
                    centers_needed = max(0, int(np.ceil(capacity_gap / 150)))
                    
                    data.append({
                        'Subzone': subzone,
                        'Region': region,
                        'Population': int(population),
                        'Current_capacity': int(current_capacity),
                        'Demand_forecast': int(demand),
                        'Capacity_gap': int(capacity_gap),
                        'Centers_needed': centers_needed
                    })
                
                # Calculate summary
                df = pd.DataFrame(data)
                summary = {
                    'total_subzones': len(df),
                    'subzones_with_gaps': len(df[df['Capacity_gap'] > 0]),
                    'total_capacity_gap': df['Capacity_gap'].sum(),
                    'total_centers_needed': df['Centers_needed'].sum(),
                    'average_capacity_gap': df['Capacity_gap'].mean(),
                    'max_capacity_gap': df['Capacity_gap'].max(),
                    'min_capacity_gap': df['Capacity_gap'].min()
                }
                
                all_forecasts[year][enrollment_rate][bto_multiplier] = {
                    'summary': summary,
                    'data': data
                }
    
    # Create metadata
    forecast_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'years': years,
            'enrollment_rates': enrollment_rates,
            'bto_multipliers': bto_multipliers,
            'total_scenarios': len(years) * len(enrollment_rates) * len(bto_multipliers),
            'description': 'Sample forecast data for ECDA preschool demand forecasting dashboard',
            'note': 'This is sample data for demonstration purposes'
        },
        'forecasts': all_forecasts
    }
    
    # Save to JSON file
    output_file = 'forecast_data.json'
    with open(output_file, 'w') as f:
        json.dump(forecast_data, f, indent=2, default=str)
    
    print(f"✅ Sample forecast data saved to {output_file}")
    return forecast_data

if __name__ == "__main__":
    # Try to generate real forecast data first
    try:
        forecast_data = generate_forecast_data()
        if forecast_data is None:
            print("🔄 Falling back to sample data...")
            forecast_data = create_sample_forecast_data()
    except Exception as e:
        print(f"❌ Error generating forecast data: {e}")
        print("🔄 Creating sample data instead...")
        forecast_data = create_sample_forecast_data()
    
    print("\n🎉 Forecast data generation completed!")
    print("📊 The dashboard can now use this data for interactive analysis.") 