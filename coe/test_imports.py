#!/usr/bin/env python3
"""
Test script to check if all imports work correctly
"""

import sys
import traceback

def test_imports():
    """Test all the imports used in the COE system"""
    
    print("🔍 Testing imports for COE system...")
    print("=" * 50)
    
    # Basic imports
    try:
        import pandas as pd
        print("✅ pandas")
    except ImportError as e:
        print(f"❌ pandas: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy")
    except ImportError as e:
        print(f"❌ numpy: {e}")
        return False
    
    try:
        import requests
        print("✅ requests")
    except ImportError as e:
        print(f"❌ requests: {e}")
        return False
    
    try:
        import yaml
        print("✅ PyYAML")
    except ImportError as e:
        print(f"❌ PyYAML: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib")
    except ImportError as e:
        print(f"❌ matplotlib: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✅ seaborn")
    except ImportError as e:
        print(f"❌ seaborn: {e}")
        return False
    
    # ML imports
    try:
        from sklearn.ensemble import RandomForestRegressor
        print("✅ scikit-learn")
    except ImportError as e:
        print(f"❌ scikit-learn: {e}")
        return False
    
    try:
        import xgboost as xgb
        print("✅ xgboost")
    except ImportError as e:
        print(f"❌ xgboost: {e}")
        return False
    
    try:
        from prophet import Prophet
        print("✅ prophet")
    except ImportError as e:
        print(f"❌ prophet: {e}")
        return False
    
    try:
        from skopt import gp_minimize
        print("✅ scikit-optimize")
    except ImportError as e:
        print(f"❌ scikit-optimize: {e}")
        return False
    
    # Deep learning imports
    try:
        import tensorflow as tf
        print("✅ tensorflow")
    except ImportError as e:
        print(f"❌ tensorflow: {e}")
        return False
    
    # Time series imports
    try:
        from statsmodels.tsa.arima.model import ARIMA
        print("✅ statsmodels")
    except ImportError as e:
        print(f"❌ statsmodels: {e}")
        return False
    
    print("\n🎉 All imports successful!")
    return True

def test_config():
    """Test if configuration file can be loaded"""
    print("\n📋 Testing configuration...")
    print("-" * 30)
    
    try:
        import yaml
        with open("config/config.yaml", 'r') as file:
            config = yaml.safe_load(file)
        print("✅ Configuration file loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_data_collector():
    """Test if data collector can be imported"""
    print("\n📊 Testing data collector...")
    print("-" * 30)
    
    try:
        from src.data_collection.coe_data_collector import COEDataCollector
        print("✅ Data collector imported successfully")
        return True
    except Exception as e:
        print(f"❌ Data collector import error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚗 COE System Import Test")
    print("=" * 50)
    
    success = True
    success &= test_imports()
    success &= test_config()
    success &= test_data_collector()
    
    if success:
        print("\n🎉 All tests passed! The system should work correctly.")
        print("You can now run: python demo_simple.py")
    else:
        print("\n❌ Some tests failed. Please install missing dependencies.")
        print("Run: pip install -r requirements.txt")
    
    sys.exit(0 if success else 1) 