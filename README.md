# Data Science Case Studies Repository

This repository contains three comprehensive data science projects focused on Singapore's key policy areas:

1. **🏫 Preschool Demand Forecasting** - ECDA preschool capacity planning
2. **🚗 COE Price Prediction & Optimization** - Certificate of Entitlement market analysis
3. **🏠 HDB Resale Portal Impact Analysis** - Property agent market impact study

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Project Overview](#-project-overview)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Details](#-project-details)
- [Data Sources](#-data-sources)
- [Docker Support](#-docker-support)
- [Troubleshooting](#-troubleshooting)

## 🚀 Quick Start

### Prerequisites
- Docker installed on your system

### Installation & Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd case_study
```

2. **Download required data files:**
```bash
# Large data files (CSV, XLSX) are gitignored to keep repository size manageable
# You'll need to download the following files:

# Preschool Forecasting Data:
# - childcare/data/btomapping.csv
# - childcare/data/ListingofCentres.csv  
# - childcare/data/respopagesex2000to2020e.xlsx

# HDB Resale Data:
# - hdb_resale/data/ResaleFlatPricesBasedonRegistrationDateFromJan2015toDec2016.csv
# - hdb_resale/data/ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv
# - hdb_resale/data/CEASalespersonsPropertyTransactionRecordsresidential.csv

# COE Data:
# - coe/data/raw/ (contains historical COE data files)
```

3. **Build and run with Docker:**
```bash
# Build the Docker image
make build

# Run the container
make test
```

3. **Access the projects:**

#### Preschool Forecasting Dashboard
```bash
# The dashboard will be available at:
# http://localhost:7860
```

#### COE Prediction System
```bash
# Access via Jupyter Lab in the container
# http://localhost:8888
# Navigate to coe/ directory and run:
# python main.py
# python demo_simple.py
# python fast_optimization_demo.py
```

#### HDB Resale Analysis
```bash
# Access via Jupyter Lab in the container
# http://localhost:8888
# Navigate to hdb_resale/ directory and open:
# hdb_notebook.ipynb
```

## 📊 Project Overview

### 1. Preschool Demand Forecasting (ECDA)
**Location:** `childcare/`

A comprehensive system for forecasting preschool demand across Singapore's subzones, helping ECDA prioritize new preschool development.

**Key Features:**
- Population-based demand prediction (18 months - 6 years)
- BTO project impact analysis
- Current capacity mapping and gap analysis
- Interactive Gradio dashboard
- Priority-based recommendations

**Quick Run:**
```bash
cd childcare/code
python run_dashboard.py
```

### 2. COE Price Prediction & Optimization
**Location:** `coe/`

Advanced machine learning system for predicting Certificate of Entitlement prices and optimizing quota strategies to stabilize market volatility.

**Key Features:**
- Multi-model ensemble predictions
- Real-time data collection from Singapore APIs
- Quota optimization algorithms
- Policy impact simulation
- Fast optimization algorithms (0.14s execution)

**Quick Run:**
```bash
cd coe
python demo_simple.py
# For fast optimization:
python fast_optimization_demo.py
```

### 3. HDB Resale Portal Impact Analysis
**Location:** `hdb_resale/`

Analysis of the impact of HDB's resale portal on property agent involvement in transactions.

**Key Features:**
- Transaction pattern analysis (2017-2024)
- Agent involvement trends
- Geographic analysis by town
- Interactive visualizations

**Quick Run:**
```bash
cd hdb_resale
jupyter notebook hdb_notebook.ipynb
```

## 🛠️ Installation

### Docker Installation (Recommended)
```bash
# Build and run with Docker
make build
make test

# Or run directly
docker run -it -v $(pwd):/notebooks -p 8888:8888 tiangg/case-study:0.0.0
```

## 📖 Usage

### Preschool Forecasting System

**Access via Docker:**
```bash
# Start the container
make test

# Access the dashboard at:
# http://localhost:7860
```

**Code Usage (in container):**
```python
# Navigate to childcare/code/ in Jupyter Lab
from main import PreschoolForecastSystem

# Initialize and run forecast
system = PreschoolForecastSystem()
system.load_data()
results = system.forecast_demand(target_year=2025)
recommendations = system.generate_recommendations()
```

### COE Prediction System

**Access via Docker:**
```bash
# Start the container
make test

# Access Jupyter Lab at:
# http://localhost:8888
# Navigate to coe/ directory
```

**Simple Demo:**
```bash
# In the container, navigate to coe/ directory
python demo_simple.py
```

**Fast Optimization:**
```python
# In the container, navigate to coe/ directory
from src.optimization.optimized_quota_optimizer import OptimizedCOEQuotaOptimizer

optimizer = OptimizedCOEQuotaOptimizer()
result = optimizer.optimize_simple_gradient()  # 0.14s execution
```

**Full Pipeline:**
```bash
# In the container, navigate to coe/ directory
python main.py  # Complete workflow
```

### HDB Resale Analysis

**Access via Docker:**
```bash
# Start the container
make test

# Access Jupyter Lab at:
# http://localhost:8888
# Navigate to hdb_resale/ directory
```

**Jupyter Notebook:**
```bash
# In the container, navigate to hdb_resale/ directory
# Open hdb_notebook.ipynb in Jupyter Lab
```

**Individual Scripts:**
```bash
# In the container, navigate to hdb_resale/code/ directory
python hdb_agent_involved_vs_direct.py
python hdb_agent_involved_by_town_heatmap.py
```

## 📁 Project Details

### Preschool Forecasting (`childcare/`)

**Architecture:**
- `main.py` - Core forecasting system
- `dashboard.py` - Gradio web interface
- `forecasting_engine.py` - ML models
- `data_processor.py` - Data processing utilities
- `config.py` - Configuration settings

**Data Sources:**
- Population statistics (SingStat)
- BTO project mapping
- Preschool center listings
- Geographic boundaries

**Output:**
- Demand forecasts by subzone
- Capacity gap analysis
- Priority recommendations
- Interactive visualizations

### COE Prediction (`coe/`)

**Architecture:**
- `src/data_collection/` - Automated data collection
- `src/models/` - ML prediction models
- `src/optimization/` - Quota optimization algorithms
- `app/` - Streamlit dashboard

**Key Algorithms:**
- **Simple Gradient (L-BFGS-B)**: 0.14s ⚡ (Recommended)
- **Dual Annealing**: 11.84s
- **Fast Genetic Algorithm**: 9.14s

**Data Sources:**
- Singapore data.gov.sg APIs
- Historical COE data (2002-2025)
- Economic indicators

### HDB Resale Analysis (`hdb_resale/`)

**Analysis Components:**
- Transaction volume trends (2017-2024)
- Agent involvement patterns
- Geographic analysis by town
- Before/after portal comparison

**Data Sources:**
- HDB resale transaction data
- CEA agent transaction records
- Property portal data

## 📊 Data Sources

### Preschool Forecasting
- `data/btomapping.csv` - BTO project information
- `data/ListingofCentres.csv` - Current preschool centers  
- `data/respopagesex2000to2020e.xlsx` - Population statistics

### COE Prediction
- Singapore data.gov.sg APIs
- Historical COE quota and premium data
- Economic indicators

### HDB Resale
- `data/ResaleFlatPricesBasedonRegistrationDateFromJan2015toDec2016.csv`
- `data/ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv`
- `data/CEASalespersonsPropertyTransactionRecordsresidential.csv`

## 📁 Data File Management

### Git Ignored Files
Large data files are excluded from the repository to keep it lightweight:

```bash
# Files ignored by .gitignore:
*.csv
*.xlsx
*.zip
```

### Required Data Files
You must download these files manually before running the projects:

#### Preschool Forecasting
- **BTO Mapping**: `childcare/data/btomapping.csv`
- **Preschool Centers**: `childcare/data/ListingofCentres.csv`
- **Population Data**: `childcare/data/respopagesex2000to2020e.xlsx`

#### HDB Resale Analysis
- **HDB Resale Data (2015-2016)**: `hdb_resale/data/ResaleFlatPricesBasedonRegistrationDateFromJan2015toDec2016.csv`
- **HDB Resale Data (2017+)**: `hdb_resale/data/ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv`
- **CEA Agent Records**: `hdb_resale/data/CEASalespersonsPropertyTransactionRecordsresidential.csv`

#### COE Prediction
- **Historical COE Data**: `coe/data/raw/` directory containing multiple CSV files
- **Note**: COE system can also fetch data automatically from Singapore APIs

### Data Sources
- **Singapore Government APIs**: data.gov.sg
- **Department of Statistics**: SingStat
- **Land Transport Authority**: LTA data
- **HDB**: Resale transaction data
- **CEA**: Property agent transaction records

## 🐳 Docker Deployment

### Build and Run
```bash
# Build image
make build

# Run container
make test

# Or run directly
docker run -it -v $(pwd):/notebooks -p 8888:8888 tiangg/case-study:0.0.0
```

### Docker Commands
```bash
# Clean build artifacts
make clean

# Push to registry
make push

# Remove image
make remove
```

### Container Access
- **Jupyter Lab**: http://localhost:8888
- **Preschool Dashboard**: http://localhost:7860
- **File System**: All project files are mounted at `/notebooks`

## 🔧 Configuration

### Preschool Forecasting
Edit `childcare/code/config.py`:
```python
max_children_per_center = 100  # Planning norm
enrollment_rate = 0.8          # 80% enrollment assumption
bto_occupancy_rate = 0.15      # 15% of BTO residents are children
capacity_threshold = 50        # Minimum gap for recommendations
```

### COE Prediction
Edit `coe/config/config.yaml`:
```yaml
data_collection:
  api_endpoints:
    - "https://data.gov.sg/api/action/datastore_search"
  
optimization:
  max_iterations: 1000
  tolerance: 1e-6
```

## 🚨 Troubleshooting

### Common Issues

**1. Docker Build Issues**
```bash
# Ensure Docker is running
docker --version

# Clean and rebuild
make clean
make build
```

**2. Container Access Issues**
```bash
# Check if container is running
docker ps

# Check container logs
docker logs case_study

# Restart container
make test
```

**3. Data Loading Issues**
```bash
# Verify data files exist in mounted volume
# In the container, check:
ls -la /notebooks/childcare/data/
ls -la /notebooks/coe/data/
ls -la /notebooks/hdb_resale/data/

# If files are missing, download them from the sources listed above
# Make sure to place them in the correct directories before running the container
```

**4. Port Conflicts**
```bash
# Check if ports are in use
lsof -i :8888
lsof -i :7860

# Stop conflicting services or modify Makefile ports
```

**5. Memory Issues (Large Datasets)**
```bash
# In the container, for COE system with large datasets
cd /notebooks/coe
python demo_minimal.py  # Uses smaller dataset
```

### Performance Optimization

**COE System:**
- Use `optimize_simple_gradient()` for fastest results (0.14s)
- Enable prediction caching for repeated runs
- Use `demo_fast.py` for quick testing
- All commands run inside the Docker container

**Preschool System:**
- Reduce sample size for quick testing
- Use LSTM models only when needed
- Enable parallel processing for large datasets
- Dashboard runs on port 7860 in container

**General Tips:**
- All code execution happens inside the Docker container
- Use Jupyter Lab (port 8888) for interactive development
- Mounted volumes ensure data persistence

## 📈 Output Examples

### Preschool Forecasting
- Interactive dashboard with demand heatmaps
- Priority recommendations by subzone
- BTO impact analysis visualizations
- Capacity gap reports

### COE Prediction
- Price forecasts with confidence intervals
- Optimal quota recommendations
- Policy impact simulations
- Performance comparison charts

### HDB Resale
- Agent involvement trend charts
- Geographic heatmaps by town
- Transaction volume analysis
- Portal impact assessment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is developed for research and policy analysis purposes. Please ensure compliance with data privacy and security requirements when using this code.

## 📞 Support

For technical support or questions about the methodology:
- Check individual project READMEs in each directory
- Review the troubleshooting guides
- Examine the demo scripts for usage examples

### Data Access
If you need help accessing the required data files:
- Contact the project maintainer for data file access
- Check Singapore government data portals for public datasets
- Ensure you have proper permissions for restricted datasets

---

**Last Updated:** December 2024  
**Docker Image:** tiangg/case-study:0.0.0  
**Base Image:** jupyter/datascience-notebook:latest  
**Dependencies:** See `requirements.txt` 