"""
Singapore COE Data Collector

This module handles data collection from Singapore's data.gov.sg APIs,
specifically for COE quota and premium data.
"""

import requests
import pandas as pd
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yaml
import os
from pathlib import Path
import numpy as np

class COEDataCollector:
    """
    Collects COE data from Singapore's data.gov.sg APIs
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the data collector with configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.base_url = self.config['data_sources']['coe_quota_premium']['url']
        self.dataset_id = self.config['data_sources']['coe_quota_premium']['dataset_id']
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create data directories
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
        """Create necessary directories for data storage"""
        for path_key, path_value in self.config['paths'].items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
    
    def fetch_coe_data(self, limit: int = 10000, offset: int = 0) -> Dict:
        """
        Fetch COE data from Singapore's API
        
        Args:
            limit: Maximum number of records to fetch
            offset: Starting record number
            
        Returns:
            Dictionary containing the API response
        """
        url = f"{self.base_url}?resource_id={self.dataset_id}&limit={limit}&offset={offset}"
        
        try:
            response = requests.get(
                url, 
                timeout=self.config['api']['timeout']
            )
            response.raise_for_status()
            
            data = response.json()
            self.logger.info(f"Successfully fetched {len(data.get('result', {}).get('records', []))} records")
            return data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching data: {e}")
            raise
    
    def process_coe_data(self, raw_data: Dict) -> pd.DataFrame:
        """
        Process raw COE data into a structured DataFrame
        
        Args:
            raw_data: Raw data from API
            
        Returns:
            Processed DataFrame
        """
        if 'result' not in raw_data or 'records' not in raw_data['result']:
            raise ValueError("Invalid data structure received from API")
        
        records = raw_data['result']['records']
        
        if not records:
            self.logger.warning("No records found in the data")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Process the data structure - the API returns data in a wide format
        # where each column represents a time period
        df_processed = self._transform_wide_to_long(df)
        
        return df_processed
    
    def _transform_wide_to_long(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform wide format data to long format suitable for analysis
        
        Args:
            df: Wide format DataFrame
            
        Returns:
            Long format DataFrame
        """
        # Identify time period columns (those that look like dates)
        time_columns = [col for col in df.columns if any(year in col for year in ['2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025'])]
        
        # Melt the DataFrame to long format
        df_long = df.melt(
            id_vars=['DataSeries'],
            value_vars=time_columns,
            var_name='Period',
            value_name='Value'
        )
        
        # Clean and parse the data
        df_long = self._clean_data(df_long)
        
        return df_long
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize the data
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Convert Period column to datetime
        df['Date'] = pd.to_datetime(df['Period'], format='%Y%b', errors='coerce')
        
        # Clean Value column - handle non-numeric values
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        
        # Filter out rows with invalid dates or values
        df = df.dropna(subset=['Date', 'Value'])
        
        # Sort by date
        df = df.sort_values('Date')
        
        # Extract category information from DataSeries
        df['Category'] = df['DataSeries'].apply(self._extract_category)
        df['Metric_Type'] = df['DataSeries'].apply(self._extract_metric_type)
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def _extract_category(self, data_series: str) -> str:
        """Extract COE category from data series string"""
        data_series_lower = data_series.lower()
        
        # Cat A: Cars up to 1600cc & 130bhp (or 97kW)
        if 'cars up to 1600cc' in data_series_lower or 'up to 1600cc and 97kw' in data_series_lower:
            return 'Cat A'
        # Cat B: Cars above 1600cc or 130bhp (or 97kW)  
        elif 'cars above 1600cc' in data_series_lower or 'above 1600cc or 97kw' in data_series_lower:
            return 'Cat B'
        # Cat C: Goods vehicles & buses
        elif 'goods vehicles' in data_series_lower or 'buses' in data_series_lower:
            return 'Cat C'
        # Cat D: Motorcycles
        elif 'motorcycle' in data_series_lower:
            return 'Cat D'
        # Cat E: Open category
        elif 'open category' in data_series_lower:
            return 'Cat E'
        # Legacy format support
        elif 'cat a' in data_series_lower or 'category a' in data_series_lower:
            return 'Cat A'
        elif 'cat b' in data_series_lower or 'category b' in data_series_lower:
            return 'Cat B'
        elif 'cat c' in data_series_lower or 'category c' in data_series_lower:
            return 'Cat C'
        elif 'cat d' in data_series_lower or 'category d' in data_series_lower:
            return 'Cat D'
        elif 'cat e' in data_series_lower or 'category e' in data_series_lower:
            return 'Cat E'
        else:
            return 'Unknown'
    
    def _extract_metric_type(self, data_series: str) -> str:
        """Extract metric type from data series string"""
        data_series_lower = data_series.lower()
        
        if 'prevailing quota premium' in data_series_lower:
            return 'Prevailing Quota Premium'
        elif 'quota premium' in data_series_lower:
            return 'Quota Premium'
        elif 'quota' in data_series_lower and 'premium' not in data_series_lower:
            return 'Quota'
        elif 'bids received' in data_series_lower:
            return 'Bids Received'
        elif 'successful bids' in data_series_lower:
            return 'Successful Bids'
        else:
            return 'Unknown'
    
    def _extract_bidding_round(self, data_series: str) -> str:
        """Extracts the bidding round (e.g., 1st Bidding) from the data series string."""
        data_series_lower = data_series.lower()
        if '1st bidding' in data_series_lower:
            return '1st Bidding'
        elif '2nd bidding' in data_series_lower:
            return '2nd Bidding'
        else:
            return 'Combined' # For PQP or other non-bidding data

    def process_and_pivot_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes raw data and pivots it into a wide format with metrics as columns.
        """
        self.logger.info("Processing and pivoting raw data...")
        
        # Apply extraction functions
        df['Category'] = df['DataSeries'].apply(self._extract_category)
        df['Metric_Type'] = df['DataSeries'].apply(self._extract_metric_type)
        df['Bidding'] = df['DataSeries'].apply(self._extract_bidding_round)
        
        # Filter out unknown categories and metrics
        df = df[(df['Category'] != 'Unknown') & (df['Metric_Type'] != 'Unknown')]
        
        # Pivot the table
        try:
            pivoted_df = df.pivot_table(
                index=['Date', 'Category', 'Bidding'],
                columns='Metric_Type',
                values='Value'
            ).reset_index()
        except Exception as e:
            self.logger.error(f"Failed to pivot data: {e}")
            # Handle cases with duplicate entries by averaging them
            self.logger.info("Attempting to pivot with aggregation due to duplicate entries...")
            pivoted_df = df.pivot_table(
                index=['Date', 'Category', 'Bidding'],
                columns='Metric_Type',
                values='Value',
                aggfunc='mean'
            ).reset_index()

        # Clean up column names
        pivoted_df.columns.name = None
        
        # Rename columns for clarity
        pivoted_df = pivoted_df.rename(columns={
            'Quota Premium': 'COE_Price',
            'Bids Received': 'Bids_Received',
            'Successful Bids': 'Successful_Bids'
        })
        
        # Ensure all expected columns exist, fill missing with NaN or 0
        expected_cols = ['Date', 'Category', 'Bidding', 'COE_Price', 'Bids_Received', 'Successful_Bids', 'Quota']
        for col in expected_cols:
            if col not in pivoted_df.columns:
                pivoted_df[col] = np.nan
        
        self.logger.info(f"Pivoted data has {pivoted_df.shape[0]} rows and {pivoted_df.shape[1]} columns.")
        return pivoted_df[expected_cols]

    def save_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save processed data to CSV file. Saves both a timestamped file and a
        'coe_data_latest.csv' file for easy access.
        
        Args:
            df: DataFrame to save
            filename: Optional filename, will generate timestamp-based name if None
            
        Returns:
            Path to timestamped saved file
        """
        raw_data_path = self.config['paths']['raw_data']
        
        # Save timestamped file
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"coe_data_{timestamp}.csv"
        
        filepath = os.path.join(raw_data_path, filename)
        df.to_csv(filepath, index=False)
        self.logger.info(f"Data saved to {filepath}")
        
        # Save latest file for easy access
        latest_filepath = os.path.join(raw_data_path, "coe_data_latest.csv")
        df.to_csv(latest_filepath, index=False)
        self.logger.info(f"Latest data also saved to {latest_filepath}")
        
        return filepath
    
    def collect_and_process_data(self) -> pd.DataFrame:
        """Main method to collect, process, and pivot data."""
        self.logger.info("Starting COE data collection and processing...")
        raw_data_dict = self.fetch_coe_data()
        if not raw_data_dict:
            return pd.DataFrame()
            
        raw_df = self.process_coe_data(raw_data_dict)
        
        # Save the raw (long-format) data
        self.save_data(raw_df)
        
        # The main pipeline should use the long-format data
        self.logger.info("Data collection and processing complete.")
        return raw_df

if __name__ == "__main__":
    # Example usage
    collector = COEDataCollector()
    data = collector.collect_and_process_data()
    print(f"Collected {len(data)} records")
    print(data.head()) 