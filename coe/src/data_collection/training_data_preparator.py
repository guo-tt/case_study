#!/usr/bin/env python3
"""
Training Data Preparator for Singapore COE Analysis

This module prepares training datasets from the enhanced COE data for machine learning models.
It handles feature engineering, data preprocessing, and creates various training datasets
for different types of models (time series, regression, classification).
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import yaml
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from .enhanced_data_collector import EnhancedCOEDataCollector

class TrainingDataPreparator:
    """
    Prepares training datasets from enhanced COE data
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the training data preparator
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        self._create_directories()
        
        # Initialize data collector
        self.data_collector = EnhancedCOEDataCollector(config_path)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            raise
    
    def _create_directories(self):
        """Create necessary directories for data storage"""
        for path_key, path_value in self.config['paths'].items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
    
    def load_latest_data(self) -> pd.DataFrame:
        """
        Load the latest enhanced COE data
        
        Returns:
            DataFrame with the latest data
        """
        latest_file = os.path.join(self.config['paths']['raw_data'], "enhanced_coe_data_latest.csv")
        
        if os.path.exists(latest_file):
            self.logger.info(f"Loading latest data from {latest_file}")
            df = pd.read_csv(latest_file)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        else:
            self.logger.info("No latest data found, collecting fresh data...")
            return self.data_collector.collect_and_merge_data()
    
    def create_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time series features for the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with time series features
        """
        self.logger.info("Creating time series features...")
        
        # Sort by date and category
        df = df.sort_values(['Category', 'Date'])
        
        # Create time-based features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['DayOfYear'] = df['Date'].dt.dayofyear
        
        # Create lag features for each category
        for category in df['Category'].unique():
            cat_mask = df['Category'] == category
            
            # Lag features for COE_Price
            if 'COE_Price' in df.columns:
                df.loc[cat_mask, 'COE_Price_Lag1'] = df.loc[cat_mask, 'COE_Price'].shift(1)
                df.loc[cat_mask, 'COE_Price_Lag2'] = df.loc[cat_mask, 'COE_Price'].shift(2)
                df.loc[cat_mask, 'COE_Price_Lag3'] = df.loc[cat_mask, 'COE_Price'].shift(3)
                df.loc[cat_mask, 'COE_Price_Lag6'] = df.loc[cat_mask, 'COE_Price'].shift(6)
                df.loc[cat_mask, 'COE_Price_Lag12'] = df.loc[cat_mask, 'COE_Price'].shift(12)
            
            # Rolling statistics for COE_Price
            if 'COE_Price' in df.columns:
                df.loc[cat_mask, 'COE_Price_Rolling_Mean_3'] = df.loc[cat_mask, 'COE_Price'].rolling(window=3, min_periods=1).mean()
                df.loc[cat_mask, 'COE_Price_Rolling_Mean_6'] = df.loc[cat_mask, 'COE_Price'].rolling(window=6, min_periods=1).mean()
                df.loc[cat_mask, 'COE_Price_Rolling_Mean_12'] = df.loc[cat_mask, 'COE_Price'].rolling(window=12, min_periods=1).mean()
                df.loc[cat_mask, 'COE_Price_Rolling_Std_6'] = df.loc[cat_mask, 'COE_Price'].rolling(window=6, min_periods=1).std()
            
            # Lag features for Quota
            if 'Quota' in df.columns:
                df.loc[cat_mask, 'Quota_Lag1'] = df.loc[cat_mask, 'Quota'].shift(1)
                df.loc[cat_mask, 'Quota_Lag2'] = df.loc[cat_mask, 'Quota'].shift(2)
                df.loc[cat_mask, 'Quota_Lag3'] = df.loc[cat_mask, 'Quota'].shift(3)
            
            # Lag features for Bids_Received
            if 'Bids_Received' in df.columns:
                df.loc[cat_mask, 'Bids_Received_Lag1'] = df.loc[cat_mask, 'Bids_Received'].shift(1)
                df.loc[cat_mask, 'Bids_Received_Lag2'] = df.loc[cat_mask, 'Bids_Received'].shift(2)
                df.loc[cat_mask, 'Bids_Received_Lag3'] = df.loc[cat_mask, 'Bids_Received'].shift(3)
            
            # Lag features for Successful_Bids
            if 'Successful_Bids' in df.columns:
                df.loc[cat_mask, 'Successful_Bids_Lag1'] = df.loc[cat_mask, 'Successful_Bids'].shift(1)
                df.loc[cat_mask, 'Successful_Bids_Lag2'] = df.loc[cat_mask, 'Successful_Bids'].shift(2)
                df.loc[cat_mask, 'Successful_Bids_Lag3'] = df.loc[cat_mask, 'Successful_Bids'].shift(3)
        
        # Create derived features
        if 'Bids_Received' in df.columns and 'Quota' in df.columns:
            df['Bid_Quota_Ratio'] = df['Bids_Received'] / df['Quota'].replace(0, np.nan)
        
        if 'Successful_Bids' in df.columns and 'Bids_Received' in df.columns:
            df['Success_Rate'] = df['Successful_Bids'] / df['Bids_Received'].replace(0, np.nan)
        
        if 'COE_Price' in df.columns and 'COE_Price_Lag1' in df.columns:
            df['COE_Price_Change'] = df['COE_Price'] - df['COE_Price_Lag1']
            df['COE_Price_Change_Pct'] = (df['COE_Price'] - df['COE_Price_Lag1']) / df['COE_Price_Lag1'].replace(0, np.nan)
        
        return df
    
    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create categorical features and encode them
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with categorical features
        """
        self.logger.info("Creating categorical features...")
        
        # Create category dummy variables
        if 'Category' in df.columns:
            category_dummies = pd.get_dummies(df['Category'], prefix='Category')
            df = pd.concat([df, category_dummies], axis=1)
        
        # Create bidding round dummy variables
        if 'Bidding_Round' in df.columns:
            bidding_dummies = pd.get_dummies(df['Bidding_Round'], prefix='Bidding')
            df = pd.concat([df, bidding_dummies], axis=1)
        
        # Create seasonal features
        df['Season'] = df['Month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        season_dummies = pd.get_dummies(df['Season'], prefix='Season')
        df = pd.concat([df, season_dummies], axis=1)
        
        return df
    
    def prepare_time_series_dataset(self, df: pd.DataFrame, target_column: str = 'COE_Price', 
                                  sequence_length: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time series dataset for LSTM/RNN models
        
        Args:
            df: Input DataFrame
            target_column: Target variable column
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X, y) arrays for time series training
        """
        self.logger.info(f"Preparing time series dataset with sequence length {sequence_length}...")
        
        # Filter out rows with missing target values
        df_clean = df.dropna(subset=[target_column])
        
        # Create sequences for each category
        X_sequences = []
        y_sequences = []
        
        for category in df_clean['Category'].unique():
            cat_data = df_clean[df_clean['Category'] == category].sort_values('Date')
            
            if len(cat_data) < sequence_length + 1:
                continue
            
            # Select features for the model
            feature_columns = [col for col in cat_data.columns if col not in [
                'Date', 'Category', 'Bidding_Round', 'Metric_Type', 'Data_Source', 'Season'
            ] and not col.startswith('Category_') and not col.startswith('Bidding_') and not col.startswith('Season_')]
            
            # Remove target column from features
            if target_column in feature_columns:
                feature_columns.remove(target_column)
            
            # Create sequences
            for i in range(len(cat_data) - sequence_length):
                # Input sequence
                X_seq = cat_data[feature_columns].iloc[i:i+sequence_length].values
                X_sequences.append(X_seq)
                
                # Target (next value)
                y_seq = cat_data[target_column].iloc[i+sequence_length]
                y_sequences.append(y_seq)
        
        if not X_sequences:
            self.logger.warning("No valid sequences created")
            return np.array([]), np.array([])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def prepare_regression_dataset(self, df: pd.DataFrame, target_column: str = 'COE_Price') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare regression dataset for traditional ML models
        
        Args:
            df: Input DataFrame
            target_column: Target variable column
            
        Returns:
            Tuple of (X, y) for regression training
        """
        self.logger.info(f"Preparing regression dataset with target: {target_column}...")
        
        # Filter out rows with missing target values
        df_clean = df.dropna(subset=[target_column])
        
        # Select features for the model
        feature_columns = [col for col in df_clean.columns if col not in [
            'Date', 'Category', 'Bidding_Round', 'Metric_Type', 'Data_Source', 'Season', target_column
        ] and not col.startswith('Category_') and not col.startswith('Bidding_') and not col.startswith('Season_')]
        
        # Remove any remaining non-numeric columns
        numeric_features = []
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                numeric_features.append(col)
        
        X = df_clean[numeric_features]
        y = df_clean[target_column]
        
        # Remove rows with any missing values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                  test_size: float = 0.2, val_size: float = 0.1) -> Tuple:
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion for test set
            val_size: Proportion for validation set
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, shuffle=False
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                      X_test: pd.DataFrame, scaler_type: str = 'standard') -> Tuple:
        """
        Scale features using the specified scaler
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            scaler_type: Type of scaler ('standard' or 'minmax')
            
        Returns:
            Tuple of scaled features and fitted scaler
        """
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        # Fit on training data
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrames with original column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    
    def save_training_data(self, data_dict: Dict, filename_prefix: str = "training_data"):
        """
        Save training data to files
        
        Args:
            data_dict: Dictionary containing training data
            filename_prefix: Prefix for saved files
        """
        processed_data_path = self.config['paths']['processed_data']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for data_type, data in data_dict.items():
            if isinstance(data, tuple):
                # Handle tuple data (e.g., X, y pairs)
                for i, item in enumerate(data):
                    if isinstance(item, pd.DataFrame):
                        filename = f"{filename_prefix}_{data_type}_{i}_{timestamp}.csv"
                        filepath = os.path.join(processed_data_path, filename)
                        item.to_csv(filepath, index=False)
                        self.logger.info(f"Saved {filename}")
                    elif isinstance(item, pd.Series):
                        filename = f"{filename_prefix}_{data_type}_{i}_{timestamp}.csv"
                        filepath = os.path.join(processed_data_path, filename)
                        item.to_frame().to_csv(filepath, index=False)
                        self.logger.info(f"Saved {filename}")
            elif isinstance(data, np.ndarray):
                filename = f"{filename_prefix}_{data_type}_{timestamp}.npy"
                filepath = os.path.join(processed_data_path, filename)
                np.save(filepath, data)
                self.logger.info(f"Saved {filename}")
            elif isinstance(data, pd.DataFrame):
                filename = f"{filename_prefix}_{data_type}_{timestamp}.csv"
                filepath = os.path.join(processed_data_path, filename)
                data.to_csv(filepath, index=False)
                self.logger.info(f"Saved {filename}")
    
    def prepare_all_training_datasets(self) -> Dict:
        """
        Prepare all training datasets for different model types
        
        Returns:
            Dictionary containing all prepared datasets
        """
        self.logger.info("Preparing all training datasets...")
        
        # Load and preprocess data
        df = self.load_latest_data()
        
        if df.empty:
            self.logger.error("No data available for training")
            return {}
        
        # Create features
        df = self.create_time_series_features(df)
        df = self.create_categorical_features(df)
        
        # Prepare different dataset types
        datasets = {}
        
        # 1. Time series dataset for LSTM/RNN
        try:
            X_ts, y_ts = self.prepare_time_series_dataset(df, target_column='COE_Price', sequence_length=12)
            if len(X_ts) > 0:
                datasets['time_series'] = (X_ts, y_ts)
                self.logger.info(f"Time series dataset: {X_ts.shape} sequences")
        except Exception as e:
            self.logger.error(f"Error preparing time series dataset: {e}")
        
        # 2. Regression dataset for traditional ML
        try:
            X_reg, y_reg = self.prepare_regression_dataset(df, target_column='COE_Price')
            if len(X_reg) > 0:
                # Split data
                X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X_reg, y_reg)
                
                # Scale features
                X_train_scaled, X_val_scaled, X_test_scaled, scaler = self.scale_features(
                    X_train, X_val, X_test, scaler_type='standard'
                )
                
                datasets['regression'] = {
                    'X_train': X_train_scaled,
                    'X_val': X_val_scaled,
                    'X_test': X_test_scaled,
                    'y_train': y_train,
                    'y_val': y_val,
                    'y_test': y_test,
                    'scaler': scaler,
                    'feature_names': X_reg.columns.tolist()
                }
                self.logger.info(f"Regression dataset: {X_train_scaled.shape[0]} training samples")
        except Exception as e:
            self.logger.error(f"Error preparing regression dataset: {e}")
        
        # 3. Category-specific datasets
        for category in df['Category'].unique():
            if category == 'Unknown':
                continue
                
            try:
                cat_df = df[df['Category'] == category].copy()
                X_cat, y_cat = self.prepare_regression_dataset(cat_df, target_column='COE_Price')
                
                if len(X_cat) > 0:
                    # Split data
                    X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X_cat, y_cat)
                    
                    # Scale features
                    X_train_scaled, X_val_scaled, X_test_scaled, scaler = self.scale_features(
                        X_train, X_val, X_test, scaler_type='standard'
                    )
                    
                    datasets[f'regression_{category.replace(" ", "_")}'] = {
                        'X_train': X_train_scaled,
                        'X_val': X_val_scaled,
                        'X_test': X_test_scaled,
                        'y_train': y_train,
                        'y_val': y_val,
                        'y_test': y_test,
                        'scaler': scaler,
                        'feature_names': X_cat.columns.tolist()
                    }
                    self.logger.info(f"Category {category} dataset: {X_train_scaled.shape[0]} training samples")
            except Exception as e:
                self.logger.error(f"Error preparing dataset for category {category}: {e}")
        
        # Save all datasets
        self.save_training_data(datasets, "enhanced_training_data")
        
        self.logger.info(f"Prepared {len(datasets)} training datasets")
        return datasets

if __name__ == "__main__":
    # Example usage
    preparator = TrainingDataPreparator()
    datasets = preparator.prepare_all_training_datasets()
    
    print(f"Prepared {len(datasets)} training datasets:")
    for dataset_name, dataset in datasets.items():
        if isinstance(dataset, tuple):
            print(f"  {dataset_name}: {dataset[0].shape} sequences")
        elif isinstance(dataset, dict):
            print(f"  {dataset_name}: {dataset['X_train'].shape[0]} training samples") 