# COE System Troubleshooting Guide

## Common Errors and Solutions

### 1. Import Errors

**Error**: `ModuleNotFoundError: No module named 'yaml'`
**Solution**: Install PyYAML
```bash
pip install PyYAML
```

**Error**: `ModuleNotFoundError: No module named 'skopt'`
**Solution**: Install scikit-optimize
```bash
pip install scikit-optimize
```

**Error**: `ModuleNotFoundError: No module named 'prophet'`
**Solution**: Install Prophet
```bash
pip install prophet
```

### 2. Configuration Errors

**Error**: `FileNotFoundError: config/config.yaml`
**Solution**: Ensure you're running from the `coe` directory
```bash
cd coe
python demo_simple.py
```

**Error**: `KeyError: 'api'` or `KeyError: 'paths'`
**Solution**: The configuration file is missing required sections. Check `config/config.yaml`

### 3. Data Collection Errors

**Error**: `requests.exceptions.ConnectionError`
**Solution**: Check internet connection and API availability
- The system fetches data from Singapore's data.gov.sg API
- Ensure you have internet access
- The API might be temporarily unavailable

**Error**: `ValueError: Invalid data structure received from API`
**Solution**: The API response format may have changed
- Check if the dataset ID is still valid
- The API endpoint might have been updated

### 4. Permission Errors

**Error**: `PermissionError: [Errno 13] Permission denied`
**Solution**: Check file permissions
```bash
chmod +x *.py
chmod -R 755 src/
```

### 5. Memory Errors

**Error**: `MemoryError` or `Killed`
**Solution**: The system is using too much memory
- Try running `demo_simple.py` instead of `main.py`
- Close other applications to free memory
- Consider running on a machine with more RAM

## Step-by-Step Debugging

### Step 1: Test Basic Setup
```bash
cd coe
python test_imports.py
```

### Step 2: Test Minimal Functionality
```bash
python demo_minimal.py
```

### Step 3: Test Simple Demo
```bash
python demo_simple.py
```

### Step 4: Test Full System
```bash
python main.py
```

## Installation Issues

### Missing Dependencies
If you get import errors, install all dependencies:
```bash
pip install -r requirements.txt
```

### Version Conflicts
If you have version conflicts:
```bash
pip install --upgrade -r requirements.txt
```

### Virtual Environment
It's recommended to use a virtual environment:
```bash
python -m venv coe_env
source coe_env/bin/activate  # On Windows: coe_env\Scripts\activate
pip install -r requirements.txt
```

## API Issues

### Singapore Data.gov.sg API
The system uses Singapore's official data API:
- **URL**: https://data.gov.sg/api/action/datastore_search
- **Dataset**: Motor Vehicle Quota, Quota Premium And Prevailing Quota Premium
- **ID**: d_22094bf608253d36c0c63b52d852dd6e

### API Rate Limits
The API has rate limits. If you get rate limit errors:
- Wait a few minutes before retrying
- The system includes retry logic

### API Changes
If the API structure changes:
- Check the official documentation
- Update the data collector code
- Contact the Singapore government data team

## File Structure Issues

### Missing Directories
The system creates these directories automatically:
- `data/raw/`
- `data/processed/`
- `data/models/`
- `data/results/`
- `logs/`

If creation fails, create them manually:
```bash
mkdir -p data/{raw,processed,models,results} logs
```

### Missing Files
Ensure these files exist:
- `config/config.yaml`
- `src/data_collection/coe_data_collector.py`
- `src/models/coe_predictive_models.py`
- `src/optimization/quota_optimizer.py`

## Performance Issues

### Slow Execution
- Use `demo_simple.py` for quick testing
- The full system can take several minutes
- ML model training is computationally intensive

### Memory Usage
- Close other applications
- Use a machine with at least 4GB RAM
- Consider running on cloud infrastructure

## Getting Help

If you continue to have issues:

1. **Check the logs**: Look in the `logs/` directory for error details
2. **Run test scripts**: Use `test_imports.py` and `demo_minimal.py`
3. **Check system requirements**: Ensure Python 3.8+ and sufficient memory
4. **Verify internet connection**: The system needs internet for data collection

## System Requirements

- **Python**: 3.8 or higher
- **Memory**: At least 4GB RAM (8GB recommended)
- **Storage**: At least 1GB free space
- **Internet**: Required for data collection
- **OS**: Linux, macOS, or Windows

## Quick Fix Commands

```bash
# Install all dependencies
pip install -r requirements.txt

# Test basic functionality
python test_imports.py

# Run minimal demo
python demo_minimal.py

# Run simple demo
python demo_simple.py

# Check logs
tail -f logs/coe_system.log
``` 