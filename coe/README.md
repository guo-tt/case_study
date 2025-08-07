# Singapore COE Price Prediction and Quota Optimization System

## Overview

This project develops a comprehensive machine learning system to predict Certificate of Entitlement (COE) prices in Singapore and recommend optimal quota adjustment strategies to stabilize market volatility.

## Problem Statement

Singapore's COE system faces challenges with:
- High price volatility across vehicle categories
- Sustained high COE premiums affecting car affordability
- Need for data-driven policy decisions on quota adjustments
- Complex interactions between supply (quotas) and demand factors

## Solution Approach

### 1. Data Collection
- Automated data collection from Singapore's data.gov.sg APIs
- Historical COE quota and premium data (2002-2025)
- Economic indicators and external factors

### 2. Predictive Modeling
- Time series forecasting models for COE prices
- Machine learning models incorporating multiple factors
- Category-specific models for different vehicle types
- Volatility prediction models

### 3. Optimization Strategy
- Mathematical optimization for quota adjustments
- Multi-objective optimization balancing price stability and revenue
- Scenario analysis for policy impact assessment

### 4. Policy Dashboard
- Interactive dashboard for policy makers
- Real-time price predictions and scenario modeling
- Policy impact visualization

## Project Structure

```
├── data/                   # Data storage
├── src/                    # Source code
│   ├── data_collection/    # Data collection modules
│   ├── models/            # ML models and algorithms
│   ├── optimization/      # Quota optimization algorithms
│   └── visualization/     # Dashboard and plots
├── notebooks/             # Jupyter notebooks for analysis
├── config/               # Configuration files
├── tests/                # Unit tests
└── app/                  # Streamlit dashboard
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Collection**: Run data collection scripts to fetch latest COE data
2. **Model Training**: Train predictive models on historical data
3. **Prediction**: Generate COE price forecasts
4. **Optimization**: Run quota optimization algorithms
5. **Dashboard**: Launch interactive policy dashboard

## Key Features

- **Multi-model Ensemble**: Combines multiple ML approaches for robust predictions
- **Real-time Updates**: Automated data collection and model retraining
- **Policy Simulation**: Test different quota strategies before implementation
- **Comprehensive Analysis**: Economic impact assessment and volatility analysis

## Data Sources

- [Singapore Motor Vehicle Quota Data](https://data.gov.sg/datasets/d_22094bf608253d36c0c63b52d852dd6e/view)
- Singapore Department of Statistics
- Land Transport Authority (LTA) data

## License

This project is developed for Singapore's Land Transport Authority for policy research and development. 