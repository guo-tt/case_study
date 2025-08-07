# ECDA Preschool Demand Forecasting System

This directory contains the complete preschool demand forecasting system for the Early Childhood Development Agency (ECDA).

## 🏗️ System Architecture

The system consists of several key components:

- **`main.py`**: Core forecasting system with data loading, analysis, and visualization
- **`config.py`**: Configuration settings and parameters
- **`data_processor.py`**: Data processing and cleaning utilities
- **`forecasting_engine.py`**: Machine learning models for demand prediction
- **`dashboard.py`**: Gradio web interface for interactive analysis
- **`run_dashboard.py`**: Launcher script for the dashboard

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Required packages (see `requirements.txt`)

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
python run_dashboard.py
```

3. Open your browser and go to: http://localhost:7860

## �� Features

### Core Forecasting
- Population-based demand prediction
- BTO project impact analysis
- Current capacity mapping
- Gap analysis and recommendations

### Interactive Dashboard
- Real-time forecast updates
- Multiple scenario analysis
- Geographic visualization
- Priority-based recommendations
- Export capabilities

### Data Sources
- Population statistics (SingStat)
- BTO project mapping
- Preschool center listings
- Geographic boundaries

## �� Key Capabilities

1. **Demand Forecasting**: Predict preschool demand for next 5 years
2. **Capacity Analysis**: Identify gaps between demand and supply
3. **BTO Impact**: Analyze impact of new housing projects
4. **Recommendations**: Prioritize areas for new preschool development
5. **Scenario Planning**: Test different enrollment rates and growth scenarios

## 📈 Usage

### Basic Forecast
```python
from main import PreschoolForecastSystem

# Initialize system
system = PreschoolForecastSystem()
system.load_data()

# Run forecast
results = system.forecast_demand(target_year=2025)

# Generate recommendations
recommendations = system.generate_recommendations()
```

### Interactive Analysis
```python
from dashboard import launch_dashboard

# Launch web interface
launch_dashboard()
```

## �� Configuration

Key parameters can be adjusted in `config.py`:

- `max_children_per_center`: 100 (planning norm)
- `enrollment_rate`: 0.8 (80% enrollment assumption)
- `bto_occupancy_rate`: 0.15 (15% of BTO residents are children 18m-6y)
- `capacity_threshold`: 50 (minimum gap for recommendations)

## 📁 Data Structure

### Input Data
- `../data/btomapping.csv`: BTO project information
- `../data/ListingofCentres.csv`: Current preschool centers
- `../data/respopagesex2000to2020e.xlsx`: Population statistics

### Output Files
- `preschool_forecast_results.csv`: Forecast results
- `preschool_forecast_dashboard.png`: Dashboard visualizations
- `preschool_recommendations.csv`: Priority recommendations

## 🔄 Regular Updates

The system is designed for regular updates:

1. **Population Data**: Update with new SingStat releases
2. **BTO Projects**: Add new announced projects
3. **Preschool Centers**: Update openings/closings
4. **Parameters**: Adjust enrollment rates and assumptions

## 🛠️ Development

### Adding New Features
1. Extend the `PreschoolForecastSystem` class in `main.py`
2. Add new visualization methods
3. Update the dashboard interface in `dashboard.py`
4. Test with sample data

### Custom Models
1. Implement new forecasting models in `forecasting_engine.py`
2. Add model selection logic
3. Update configuration parameters

## 📞 Support

For technical support or questions about the forecasting methodology, please refer to the main project documentation.

## 📄 License

This system is developed for ECDA internal use. Please ensure compliance with data privacy and security requirements. 