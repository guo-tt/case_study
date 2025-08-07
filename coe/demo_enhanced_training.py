#!/usr/bin/env python3
"""
Demo: Using Enhanced COE Training Datasets

This script demonstrates how to load and use the enhanced training datasets
for different types of machine learning models.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_collection.training_data_preparator import TrainingDataPreparator

def demo_enhanced_training():
    """Demonstrate the enhanced training datasets"""
    
    print("🚀 Enhanced COE Training Datasets Demo")
    print("=" * 60)
    
    # Initialize the training data preparator
    preparator = TrainingDataPreparator()
    
    # Prepare all training datasets
    print("📊 Preparing training datasets...")
    datasets = preparator.prepare_all_training_datasets()
    
    if not datasets:
        print("❌ No datasets were prepared")
        return
    
    print(f"✅ Successfully prepared {len(datasets)} training datasets:")
    
    # Demonstrate each dataset type
    for dataset_name, dataset in datasets.items():
        print(f"\n📋 Dataset: {dataset_name}")
        print("-" * 40)
        
        if isinstance(dataset, tuple):
            # Time series dataset
            X, y = dataset
            print(f"  Type: Time Series (LSTM/RNN)")
            print(f"  X shape: {X.shape}")
            print(f"  y shape: {y.shape}")
            print(f"  Sequence length: {X.shape[1]}")
            print(f"  Features per sequence: {X.shape[2]}")
            
            # Show sample data
            print(f"  Sample X[0]: {X[0][0][:5]}...")  # First 5 features of first timestep
            print(f"  Sample y[0]: {y[0]}")
            
        elif isinstance(dataset, dict):
            # Regression dataset
            X_train = dataset['X_train']
            y_train = dataset['y_train']
            X_val = dataset['X_val']
            y_val = dataset['y_val']
            X_test = dataset['X_test']
            y_test = dataset['y_test']
            
            print(f"  Type: Regression")
            print(f"  Training samples: {X_train.shape[0]}")
            print(f"  Validation samples: {X_val.shape[0]}")
            print(f"  Test samples: {X_test.shape[0]}")
            print(f"  Features: {X_train.shape[1]}")
            
            # Show feature names
            if 'feature_names' in dataset:
                print(f"  Feature names: {dataset['feature_names'][:5]}...")
            
            # Show sample data
            print(f"  Sample X_train[0]: {X_train.iloc[0][:5].values}")
            print(f"  Sample y_train[0]: {y_train.iloc[0]}")
            
            # Show target statistics
            print(f"  Target mean: {y_train.mean():.2f}")
            print(f"  Target std: {y_train.std():.2f}")
            print(f"  Target min: {y_train.min():.2f}")
            print(f"  Target max: {y_train.max():.2f}")
    
    print("\n" + "=" * 60)
    print("💡 Usage Examples:")
    print("=" * 60)
    
    # Example 1: Using regression dataset
    if 'regression' in datasets:
        print("\n🔹 Example 1: Traditional ML Regression")
        print("   datasets['regression']['X_train'] - Training features")
        print("   datasets['regression']['y_train'] - Training targets")
        print("   datasets['regression']['X_val'] - Validation features")
        print("   datasets['regression']['y_val'] - Validation targets")
        print("   datasets['regression']['X_test'] - Test features")
        print("   datasets['regression']['y_test'] - Test targets")
        print("   datasets['regression']['scaler'] - Fitted scaler for new data")
    
    # Example 2: Using time series dataset
    if 'time_series' in datasets:
        print("\n🔹 Example 2: LSTM/RNN Time Series")
        print("   X, y = datasets['time_series']")
        print("   X.shape = (samples, sequence_length, features)")
        print("   y.shape = (samples,)")
        print("   Each sample contains 12 months of historical data")
    
    # Example 3: Using category-specific datasets
    category_datasets = [name for name in datasets.keys() if name.startswith('regression_Cat_')]
    if category_datasets:
        print(f"\n🔹 Example 3: Category-Specific Models")
        for cat_dataset in category_datasets:
            category = cat_dataset.replace('regression_', '')
            print(f"   {category}: datasets['{cat_dataset}']")
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("=" * 60)
    print("1. Use 'regression' dataset for XGBoost, Random Forest, etc.")
    print("2. Use 'time_series' dataset for LSTM, GRU, etc.")
    print("3. Use category-specific datasets for specialized models")
    print("4. Apply the scaler to new data before prediction")
    print("5. Train multiple models and ensemble them")
    
    return datasets

def demo_model_training_example(datasets):
    """Demonstrate a simple model training example"""
    
    print("\n🔬 Simple Model Training Example")
    print("=" * 60)
    
    if 'regression' not in datasets:
        print("❌ Regression dataset not available")
        return
    
    # Get the regression dataset
    reg_data = datasets['regression']
    X_train = reg_data['X_train']
    y_train = reg_data['y_train']
    X_val = reg_data['X_val']
    y_val = reg_data['y_val']
    
    print(f"📊 Training a simple model on {X_train.shape[0]} samples...")
    
    try:
        # Try to import sklearn for a simple example
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Train a simple linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        print(f"✅ Model trained successfully!")
        print(f"   Training MSE: {train_mse:.2f}")
        print(f"   Validation MSE: {val_mse:.2f}")
        print(f"   Training R²: {train_r2:.3f}")
        print(f"   Validation R²: {val_r2:.3f}")
        
        # Show feature importance (coefficients)
        feature_names = reg_data.get('feature_names', [f'Feature_{i}' for i in range(X_train.shape[1])])
        coefficients = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        print(f"\n🔍 Top 10 Most Important Features:")
        print(coefficients.head(10).to_string(index=False))
        
    except ImportError:
        print("⚠️  sklearn not available, skipping model training example")
        print("   Install with: pip install scikit-learn")

if __name__ == "__main__":
    # Run the demo
    datasets = demo_enhanced_training()
    
    if datasets:
        # Run a simple training example
        demo_model_training_example(datasets)
    
    print("\n🎉 Demo completed!") 