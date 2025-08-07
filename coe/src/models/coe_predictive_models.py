"""
COE Predictive Models

This module implements various machine learning models for predicting COE prices,
including time series models, ensemble methods, and deep learning approaches.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import joblib
import yaml
from pathlib import Path

# Time Series Models
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Optimization
from skopt import gp_minimize
from skopt.space import Real, Integer

class COEPredictiveModels:
    """
    Comprehensive COE price prediction system with multiple model types
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the predictive modeling system
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.raw_data = None  # To store raw data for fallbacks
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create model directories
        self._create_directories()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            raise
    
    def _create_directories(self):
        """Create necessary directories for model storage"""
        for path_key, path_value in self.config['paths'].items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares features for modeling from the new long-format DataFrame.
        The target variable is 'Value' where Metric_Type is 'Quota Premium'.
        """
        self.logger.info("Preparing features from long-format data...")
        
        # Ensure Date is in datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort data to ensure correct order for time-based features
        df = df.sort_values(by=['Category', 'Date']).reset_index(drop=True)

        # Pivot the data to create features from different metric types
        feature_df = df.pivot_table(index=['Date', 'Category'], columns='Metric_Type', values='Value').reset_index()
        
        # Rename columns for clarity and compatibility
        feature_df = feature_df.rename(columns={
            'Quota Premium': 'COE_Price',
            'Bids Received': 'Bids_Received',
            'Successful Bids': 'Successful_Bids'
        })

        # Handle missing values after pivot
        feature_cols = ['COE_Price', 'Quota', 'Bids_Received', 'Successful_Bids']
        for col in feature_cols:
            if col in feature_df.columns:
                feature_df[col] = feature_df.groupby('Category')[col].transform(lambda x: x.ffill().bfill())
            else:
                feature_df[col] = 0 # If a metric type is missing entirely
        
        feature_df = feature_df.dropna(subset=['COE_Price'])

        # --- Feature Engineering ---
        # Time-based features
        feature_df['month'] = feature_df['Date'].dt.month
        feature_df['year'] = feature_df['Date'].dt.year
        feature_df['quarter'] = feature_df['Date'].dt.quarter
        
        # Lag features for the target variable
        for lag in [1, 3, 6]:
            feature_df[f'price_lag_{lag}'] = feature_df.groupby('Category')['COE_Price'].shift(lag)
            
        # Rolling window features for the target variable
        for window in [1, 3, 6, 12, 24]:
            feature_df[f'price_rolling_mean_{window}'] = feature_df.groupby('Category')['COE_Price'].shift(1).rolling(window=window).mean()
            feature_df[f'price_rolling_std_{window}'] = feature_df.groupby('Category')['COE_Price'].shift(1).rolling(window=window).std()

        # Interaction features - check for column existence
        if 'Bids_Received' in feature_df.columns and 'Quota' in feature_df.columns:
            feature_df['bids_to_quota'] = feature_df['Bids_Received'] / feature_df['Quota']
        if 'Successful_Bids' in feature_df.columns and 'Bids_Received' in feature_df.columns:
            feature_df['success_rate'] = feature_df['Successful_Bids'] / feature_df['Bids_Received']
        
        # Replace infinite values that may result from division by zero
        feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Fill any remaining NaNs from feature engineering
        feature_df = feature_df.fillna(0)

        # Define feature columns for ML models
        self.feature_columns = [col for col in [
            'Quota', 'Bids_Received', 'Successful_Bids',
            'month', 'year', 'quarter',
            'price_lag_1', 'price_lag_3', 'price_lag_6',
            'price_rolling_mean_1', 'price_rolling_std_1',
            'price_rolling_mean_3', 'price_rolling_std_3',
            'price_rolling_mean_6', 'price_rolling_std_6',
            'price_rolling_mean_12', 'price_rolling_std_12',
            'price_rolling_mean_24', 'price_rolling_std_24',
            'bids_to_quota', 'success_rate'
        ] if col in feature_df.columns]
        
        self.logger.info(f"Feature preparation complete. {len(self.feature_columns)} features created.")
        return feature_df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets for time series
        
        Args:
            df: DataFrame with features
            
        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        # Sort by date
        df = df.sort_values(['Category', 'Date']).reset_index(drop=True)
        
        # Calculate split indices
        n = len(df)
        train_end = int(n * self.config['models']['training']['train_split'])
        val_end = int(n * (self.config['models']['training']['train_split'] + 
                          self.config['models']['training']['validation_split']))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        self.logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def train_arima_model(self, train_data: pd.DataFrame, category: str):
        """Train ARIMA model and store it in the self.models dictionary."""
        try:
            self.logger.info(f"Training ARIMA model for {category}")
            train_cat = train_data[train_data['Category'] == category]['COE_Price'].dropna()

            if len(train_cat) < self.config['models']['min_records_for_training']:
                self.logger.warning(f"Insufficient data for ARIMA in {category}")
                return

            model = ARIMA(train_cat, order=self.config['models']['arima']['order'])
            fitted_model = model.fit()
            
            self.models[category]['arima'] = {'model': fitted_model, 'category': category}
            self.logger.info(f"ARIMA model for {category} trained successfully.")
            
        except Exception as e:
            self.logger.warning(f"ARIMA model training for {category} failed: {e}")
    
    def train_prophet_model(self, train_data: pd.DataFrame, category: str):
        """Train Prophet model and store it in the self.models dictionary."""
        try:
            self.logger.info(f"Training Prophet model for {category}")
            train_cat = train_data[train_data['Category'] == category][['Date', 'COE_Price']].rename(columns={'Date': 'ds', 'COE_Price': 'y'})

            if len(train_cat) < self.config['models']['min_records_for_training']:
                self.logger.warning(f"Insufficient data for Prophet in {category}")
                return

            model = Prophet()
            model.fit(train_cat)
            
            self.models[category]['prophet'] = {'model': model, 'category': category}
            self.logger.info(f"Prophet model for {category} trained successfully.")
            
        except Exception as e:
            self.logger.warning(f"Prophet model training for {category} failed: {e}")
    
    def train_xgboost_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame, category: str):
        """Train XGBoost model and store it in the self.models dictionary."""
        try:
            self.logger.info(f"Training XGBoost model for {category}")

            train_cat = train_data[train_data['Category'] == category].copy().dropna(subset=self.feature_columns + ['COE_Price'])
            val_cat = val_data[val_data['Category'] == category].copy().dropna(subset=self.feature_columns + ['COE_Price'])

            if len(train_cat) < 20:
                self.logger.warning(f"Insufficient data for XGBoost in {category}")
                return

            X_train = train_cat[self.feature_columns]
            y_train = train_cat['COE_Price']
            X_val = val_cat[self.feature_columns]
            y_val = val_cat['COE_Price']
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val) if len(X_val) > 0 else np.array([])
            
            model = xgb.XGBRegressor(random_state=42)
            
            model.fit(X_train_scaled, y_train)

            self.models[category]['xgboost'] = {
                'model': model,
                'scaler': scaler,
                'feature_columns': self.feature_columns,
                'category': category
            }
            self.logger.info(f"XGBoost model for {category} trained successfully.")
            
        except Exception as e:
            self.logger.warning(f"XGBoost model training for {category} failed: {e}")
    
    def train_lstm_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame, category: str):
        """Train LSTM model and store it in the self.models dictionary."""
        try:
            self.logger.info(f"Training LSTM model for {category}")

            train_cat = train_data[train_data['Category'] == category]['COE_Price'].values
            val_cat = val_data[val_data['Category'] == category]['COE_Price'].values

            if len(train_cat) < 50:
                self.logger.warning(f"Insufficient data for LSTM in {category}")
                return

            scaler = MinMaxScaler()
            train_scaled = scaler.fit_transform(train_cat.reshape(-1, 1)).flatten()
            val_scaled = scaler.transform(val_cat.reshape(-1, 1)).flatten() if len(val_cat) > 0 else np.array([])
            
            sequence_length = self.config['models']['hyperparameters']['lstm']['sequence_length']

            def create_sequences(data, seq_length):
                X, y = [], []
                for i in range(seq_length, len(data)):
                    X.append(data[i-seq_length:i])
                    y.append(data[i])
                return np.array(X), np.array(y)

            X_train, y_train = create_sequences(train_scaled, sequence_length)
            X_val, y_val = create_sequences(val_scaled, sequence_length)

            if len(X_train) == 0:
                self.logger.warning(f"No training sequences created for LSTM in {category}")
                return

            model = Sequential([
                LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            
            model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=0)

            self.models[category]['lstm'] = {
                'model': model,
                'scaler': scaler,
                'sequence_length': sequence_length,
                'category': category
            }
            self.logger.info(f"LSTM model for {category} trained successfully.")
            
        except Exception as e:
            self.logger.warning(f"LSTM model training for {category} failed: {e}")
    
    def train_all_models(self, df: pd.DataFrame) -> Dict:
        """Train all models for each category, using the corrected wide-format data."""
        self.logger.info("Training all models...")
        self.raw_data = df.copy()

        # The dataframe 'df' is already the correctly processed price data.
        # The filter on 'Metric_Type' is no longer needed.
        price_data = df
        
        self.models = {}
        all_train_data, all_val_data, all_test_data = [], [], []

        for category in price_data['Category'].unique():
            if category == 'Unknown':
                continue

            self.logger.info(f"--- Processing Category: {category} ---")
            category_data = price_data[price_data['Category'] == category].sort_values('Date').copy()

            if len(category_data) < self.config['models']['min_records_for_training']:
                self.logger.warning(f"Skipping {category}: Insufficient data ({len(category_data)} records).")
                continue

            # Split data for this specific category
            train_data, val_data, test_data = self.split_data(category_data)
            all_train_data.append(train_data)
            all_val_data.append(val_data)
            all_test_data.append(test_data)

            self.logger.info(f"Data split for {category} - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
            
            if len(train_data) == 0:
                self.logger.warning(f"Skipping {category}: No training data after split.")
                continue

            self.models[category] = {}
            
            # Train models using the category-specific data
            self.train_arima_model(train_data, category)
            self.train_prophet_model(train_data, category)
            self.train_xgboost_model(train_data, val_data, category)
            self.train_lstm_model(train_data, val_data, category)
            
            if not self.models[category]:
                self.logger.error(f"All models failed to train for {category}.")

        # Store combined data splits for evaluation purposes
        if all_train_data:
            self.train_data = pd.concat(all_train_data)
            self.val_data = pd.concat(all_val_data)
            self.test_data = pd.concat(all_test_data)
            
            # Store the full feature dataframe for later use in predictions
            self.full_feature_df = pd.concat([self.train_data, self.val_data, self.test_data]).sort_values('Date').reset_index(drop=True)
        
        return self.models
    
    def _create_future_features(self, category: str, periods: int) -> Optional[pd.DataFrame]:
        """
        Creates a feature DataFrame for future predictions based on historical data.
        """
        if not hasattr(self, 'full_feature_df'):
            self.logger.warning("Full feature DataFrame not available. Cannot create future features.")
            return None

        history_df = self.full_feature_df[self.full_feature_df['Category'] == category].copy()
        if history_df.empty:
            self.logger.warning(f"No historical data for {category}. Cannot create future features.")
            return None
            
        last_known_date = history_df['Date'].max()
        
        future_data = []
        # Use a combined history of actuals and newly generated future points to build features
        combined_history = history_df.copy()

        for i in range(1, periods + 1):
            future_date = last_known_date + pd.DateOffset(months=i)
            new_row = {'Date': future_date, 'Category': category}

            # Time-based features
            new_row['month'] = future_date.month
            new_row['year'] = future_date.year
            new_row['quarter'] = future_date.quarter
            
            # Carry forward last known values for non-price features
            last_known_row = combined_history.iloc[-1]
            for col in ['Quota', 'Bids_Received', 'Successful_Bids']:
                if col in last_known_row:
                    new_row[col] = last_known_row[col]

            # Lag and Rolling features from combined history's 'COE_Price'
            price_history = combined_history['COE_Price']
            for lag in [1, 3, 6]:
                if f'price_lag_{lag}' in self.feature_columns:
                    new_row[f'price_lag_{lag}'] = price_history.iloc[-lag] if len(price_history) >= lag else 0
            
            for window in [1, 3, 6, 12, 24]:
                if f'price_rolling_mean_{window}' in self.feature_columns:
                    new_row[f'price_rolling_mean_{window}'] = price_history.rolling(window=window).mean().iloc[-1]
                if f'price_rolling_std_{window}' in self.feature_columns:
                    new_row[f'price_rolling_std_{window}'] = price_history.rolling(window=window).std().iloc[-1]

            # Interaction features
            if 'bids_to_quota' in self.feature_columns and new_row.get('Quota') and new_row.get('Bids_Received'):
                new_row['bids_to_quota'] = new_row.get('Bids_Received', 0) / new_row.get('Quota', 1)
            if 'success_rate' in self.feature_columns and new_row.get('Successful_Bids') and new_row.get('Bids_Received'):
                new_row['success_rate'] = new_row.get('Successful_Bids', 0) / new_row.get('Bids_Received', 1)

            future_data.append(new_row)
            
            # For multi-step forecasting, we would append the new row (with a predicted price)
            # to combined_history here. For now, this is for single-step prediction features.

        future_df = pd.DataFrame(future_data)
        for col in self.feature_columns:
            if col not in future_df.columns:
                future_df[col] = 0
                
        future_df.replace([np.inf, -np.inf], 0, inplace=True)
        future_df = future_df.fillna(0)
        
        return future_df[self.feature_columns]

    def predict(self, category: str, model_type: str, periods: int = 12) -> np.ndarray:
        """Predict COE prices for a given category and model type."""
        if category not in self.models or model_type not in self.models[category]:
            return np.array([])

        model_info = self.models[category][model_type]
        model = model_info['model']

        # Ensure all return values are numpy arrays
        if model_type == 'arima':
            forecast = model.forecast(steps=periods)
            return forecast.values if hasattr(forecast, 'values') else np.array(forecast)
        
        elif model_type == 'prophet':
            future = model.make_future_dataframe(periods=periods, freq='MS')
            forecast = model.predict(future)
            return forecast['yhat'].tail(periods).values

        elif model_type == 'xgboost':
            future_df = self._create_future_features(category, periods)
            if future_df is None:
                return np.array([])
            
            X_pred = future_df[model_info['feature_columns']]
            X_pred_scaled = model_info['scaler'].transform(X_pred)
            predictions = model.predict(X_pred_scaled)
            return np.array(predictions)

        elif model_type == 'lstm':
            scaler = model_info['scaler']
            sequence_length = model_info['sequence_length']
            
            # Get last known sequence
            last_data = self.raw_data[self.raw_data['Category'] == category]['COE_Price'].dropna().values
            last_sequence = last_data[-sequence_length:]
            
            predictions = []
            current_seq = scaler.transform(last_sequence.reshape(-1, 1)).reshape(1, sequence_length, 1)

            for _ in range(periods):
                pred = model.predict(current_seq)
                predictions.append(scaler.inverse_transform(pred)[0][0])
                # Update sequence for next prediction
                pred_reshaped = pred.reshape(1, 1, 1)
                current_seq = np.append(current_seq[:, 1:, :], pred_reshaped, axis=1)

            return np.array(predictions)
        
        return np.array([])
    
    def get_prediction(self, category: str, periods: int = 12) -> np.ndarray:
        """
        Generates predictions based on the strategy defined in the config.
        Can be 'ensemble' or a specific model like 'lstm'.
        """
        strategy = self.config['models'].get('prediction_strategy', 'ensemble').lower()

        if strategy == 'ensemble':
            self.logger.info(f"Using 'ensemble' prediction strategy for {category}.")
            return self.ensemble_predict(category, periods)
        
        # If a specific model is requested, use it directly
        if strategy in self.models.get(category, {}):
            self.logger.info(f"Using '{strategy}' prediction strategy for {category}.")
            return self.predict(category, strategy, periods)
        else:
            self.logger.warning(
                f"Strategy '{strategy}' not available for {category}. "
                "Defaulting to ensemble prediction."
            )
            return self.ensemble_predict(category, periods)

    def ensemble_predict(self, category: str, periods: int = 12) -> np.ndarray:
        """
        Generate ensemble predictions with a robust fallback mechanism.
        If all models fail, it returns the last known price for the category.
        """
        if category not in self.models:
            self.logger.warning(f"No models for category {category}, using fallback.")
            return self._get_fallback_price(category, periods)

        predictions = {}
        weights = self.config['models']['ensemble']['voting_weights']
        
        # Get predictions from each model
        for model_type in self.models[category].keys():
            try:
                pred = self.predict(category, model_type, periods)
                if pred is not None and len(pred) == periods:
                    predictions[model_type] = pred
            except Exception as e:
                self.logger.debug(f"Prediction from {model_type} for {category} failed: {e}")

        if not predictions:
            self.logger.warning(f"All models failed for {category}, using fallback.")
            return self._get_fallback_price(category, periods)
        
        # Calculate weighted ensemble
        ensemble_pred = np.zeros(periods)
        total_weight = 0
        
        for model_type, pred in predictions.items():
            if model_type in weights:
                weight = weights.get(model_type, 0)
                if weight > 0:
                    ensemble_pred += weight * pred
                    total_weight += weight
        
        if total_weight > 0:
            ensemble_pred /= total_weight
        else:
            # If no weighted models succeeded, average all successful predictions
            self.logger.warning(f"No models with positive weights succeeded for {category}, averaging results.")
            all_preds = np.array(list(predictions.values()))
            if all_preds.size > 0:
                ensemble_pred = np.mean(all_preds, axis=0)
            else:
                return self._get_fallback_price(category, periods)

        return ensemble_pred

    def _get_fallback_price(self, category: str, periods: int) -> np.ndarray:
        """
        Returns the last known 'Quota Premium' price for a category, repeated for all periods.
        """
        if self.raw_data is None or self.raw_data.empty:
            # Absolute fallback if no data is available at all
            return np.array([50000] * periods)
            
        category_data = self.raw_data[
            (self.raw_data['Category'] == category) &
            (self.raw_data['Metric_Type'] == 'Quota Premium')
        ].sort_values('Date', ascending=False)
        
        if not category_data.empty:
            last_price = category_data.iloc[0]['Value']
            return np.array([last_price] * periods)
        else:
            # Fallback if the category has no historical premium data
            return np.array([50000] * periods)
    
    def evaluate_models(self) -> Dict:
        """
        Evaluate all models on test data
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info("Evaluating models...")
        
        results = {}
        
        for category in self.models.keys():
            category_results = {}
            
            # Get actual test values
            test_actual = self.test_data[self.test_data['Category'] == category]['COE_Price'].values
            
            if len(test_actual) == 0:
                continue
            
            for model_type in self.models[category].keys():
                try:
                    # Get predictions for test period
                    test_pred = self.predict(category, model_type, len(test_actual))
                    
                    if len(test_pred) == len(test_actual):
                        # Calculate metrics
                        mae = mean_absolute_error(test_actual, test_pred)
                        mse = mean_squared_error(test_actual, test_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(test_actual, test_pred)
                        
                        category_results[model_type] = {
                            'mae': mae,
                            'mse': mse,
                            'rmse': rmse,
                            'r2': r2
                        }
                
                except Exception as e:
                    self.logger.warning(f"Failed to evaluate {model_type} for {category}: {e}")
            
            results[category] = category_results
        
        return results
    
    def save_models(self, filepath: str = None):
        """
        Save all trained models to disk
        
        Args:
            filepath: Optional filepath, will use default if None
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Path(self.config['paths']['models']) / f"coe_models_{timestamp}.joblib"
        
        # Prepare models for saving (exclude non-serializable parts)
        models_to_save = {}
        
        for category, category_models in self.models.items():
            models_to_save[category] = {}
            
            for model_type, model_info in category_models.items():
                if model_type == 'lstm':
                    # Save LSTM model separately
                    model_path = str(filepath).replace('.joblib', f'_{category}_{model_type}.h5')
                    model_info['model'].save(model_path)
                    
                    # Save other info
                    lstm_info = {k: v for k, v in model_info.items() if k != 'model'}
                    lstm_info['model_path'] = model_path
                    models_to_save[category][model_type] = lstm_info
                else:
                    models_to_save[category][model_type] = model_info
        
        # Save models
        joblib.dump({
            'models': models_to_save,
            'feature_columns': self.feature_columns,
            'config': self.config
        }, filepath)
        
        self.logger.info(f"Models saved to {filepath}")

if __name__ == "__main__":
    # Example usage
    from src.data_collection.coe_data_collector import COEDataCollector
    
    # Collect data
    collector = COEDataCollector()
    raw_data = collector.collect_and_process_data()
    
    # Train models
    models = COEPredictiveModels()
    feature_df = models.prepare_features(raw_data)
    trained_models = models.train_all_models(feature_df)
    
    # Evaluate models
    evaluation_results = models.evaluate_models()
    print("Model evaluation results:", evaluation_results)
    
    # Generate predictions
    for category in trained_models.keys():
        predictions = models.ensemble_predict(category, periods=12)
        print(f"{category} - Next 12 months predictions: {predictions}")
    
    # Save models
    models.save_models() 