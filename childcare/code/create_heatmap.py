#!/usr/bin/env python3
"""
Create heatmap visualization for capacity gap by subzone in 2025
"""

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def create_capacity_gap_heatmap():
    """Create heatmap showing capacity gap by subzone for 2025"""
    
    print("🏫 Creating Capacity Gap Heatmap for 2025")
    print("=" * 60)
    
    # Load forecast data
    try:
        with open('forecast_data.json', 'r') as f:
            forecast_data = json.load(f)
        print("✅ Loaded forecast data successfully")
    except Exception as e:
        print(f"❌ Error loading forecast data: {e}")
        return None
    
    # Get 2025 data with default parameters (80% enrollment, 1.0 BTO multiplier)
    year_2025 = forecast_data['forecasts'].get('2025', {})
    enrollment_80 = year_2025.get('80', {})
    scenario_1_0 = enrollment_80.get('1.0', {})
    
    if not scenario_1_0:
        print("❌ No data found for 2025 with 80% enrollment and 1.0 BTO multiplier")
        return None
    
    # Convert to DataFrame
    data = pd.DataFrame(scenario_1_0['data'])
    print(f"✅ Loaded {len(data)} subzone records")
    
    # Create heatmap data
    heatmap_data = data.pivot_table(
        values='Capacity_gap', 
        index='Planning_area', 
        columns='Subzone', 
        aggfunc='sum'
    ).fillna(0)
    
    print(f"✅ Created heatmap data with {heatmap_data.shape[0]} planning areas and {heatmap_data.shape[1]} subzones")
    
    # Create Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdYlBu_r',  # Red for high gaps, Blue for low gaps
        colorbar=dict(title="Capacity Gap"),
        hovertemplate='<b>%{y}</b><br>Subzone: %{x}<br>Gap: %{z:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Capacity Gap by Subzone - 2025 Forecast",
        xaxis_title="Subzone",
        yaxis_title="Planning Area",
        width=1200,
        height=800,
        font=dict(size=12)
    )
    
    # Save as HTML (interactive)
    fig.write_html("capacity_gap_heatmap_2025.html")
    print("✅ Saved interactive heatmap as capacity_gap_heatmap_2025.html")
    
    # Create matplotlib version for static image
    plt.figure(figsize=(16, 10))
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt='.0f',
        cmap='RdYlBu_r',
        cbar_kws={'label': 'Capacity Gap'},
        linewidths=0.5
    )
    plt.title('Capacity Gap by Subzone - 2025 Forecast', fontsize=16, pad=20)
    plt.xlabel('Subzone', fontsize=12)
    plt.ylabel('Planning Area', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save matplotlib version
    plt.savefig("capacity_gap_heatmap_2025.png", dpi=300, bbox_inches='tight')
    print("✅ Saved static heatmap as capacity_gap_heatmap_2025.png")
    plt.close()
    
    # Create summary statistics
    print("\n📊 Summary Statistics:")
    print(f"Total Capacity Gap: {data['Capacity_gap'].sum():,.0f}")
    print(f"Average Gap per Subzone: {data['Capacity_gap'].mean():,.0f}")
    print(f"Maximum Gap: {data['Capacity_gap'].max():,.0f}")
    print(f"Minimum Gap: {data['Capacity_gap'].min():,.0f}")
    
    # Top 10 subzones with highest gaps
    top_gaps = data.nlargest(10, 'Capacity_gap')[['Planning_area', 'Subzone', 'Capacity_gap']]
    print("\n🔝 Top 10 Subzones with Highest Capacity Gaps:")
    print(top_gaps.to_string(index=False))
    
    return fig

def create_enrollment_impact_heatmap():
    """Create heatmap showing how enrollment rate affects capacity gap"""
    
    print("\n📈 Creating Enrollment Impact Heatmap")
    print("=" * 50)
    
    # Load forecast data
    try:
        with open('forecast_data.json', 'r') as f:
            forecast_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading forecast data: {e}")
        return None
    
    # Get 2025 data for different enrollment rates
    year_2025 = forecast_data['forecasts'].get('2025', {})
    bto_1_0 = '1.0'  # Default BTO multiplier
    
    enrollment_rates = []
    planning_areas = []
    capacity_gaps = []
    
    for enrollment_rate in ['60', '70', '80', '90', '100']:
        if enrollment_rate in year_2025 and bto_1_0 in year_2025[enrollment_rate]:
            data = pd.DataFrame(year_2025[enrollment_rate][bto_1_0]['data'])
            
            # Aggregate by planning area
            area_gaps = data.groupby('Planning_area')['Capacity_gap'].sum().reset_index()
            
            for _, row in area_gaps.iterrows():
                enrollment_rates.append(int(enrollment_rate))
                planning_areas.append(row['Planning_area'])
                capacity_gaps.append(row['Capacity_gap'])
    
    # Create DataFrame for heatmap
    impact_df = pd.DataFrame({
        'Enrollment_Rate': enrollment_rates,
        'Planning_area': planning_areas,
        'Capacity_Gap': capacity_gaps
    })
    
    # Pivot for heatmap
    impact_heatmap = impact_df.pivot_table(
        values='Capacity_Gap',
        index='Planning_area',
        columns='Enrollment_Rate',
        aggfunc='sum'
    ).fillna(0)
    
    # Create matplotlib heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        impact_heatmap,
        annot=True,
        fmt='.0f',
        cmap='RdYlBu_r',
        cbar_kws={'label': 'Capacity Gap'},
        linewidths=0.5
    )
    plt.title('Capacity Gap by Planning Area and Enrollment Rate - 2025', fontsize=16, pad=20)
    plt.xlabel('Enrollment Rate (%)', fontsize=12)
    plt.ylabel('Planning Area', fontsize=12)
    plt.tight_layout()
    
    # Save
    plt.savefig("enrollment_impact_heatmap_2025.png", dpi=300, bbox_inches='tight')
    print("✅ Saved enrollment impact heatmap as enrollment_impact_heatmap_2025.png")
    plt.close()
    
    return impact_heatmap

if __name__ == "__main__":
    # Create main capacity gap heatmap
    fig1 = create_capacity_gap_heatmap()
    
    # Create enrollment impact heatmap
    fig2 = create_enrollment_impact_heatmap()
    
    print("\n🎉 Heatmap generation completed!")
    print("📁 Generated files:")
    print("   - capacity_gap_heatmap_2025.html (interactive)")
    print("   - capacity_gap_heatmap_2025.png (static)")
    print("   - enrollment_impact_heatmap_2025.png (enrollment impact)") 