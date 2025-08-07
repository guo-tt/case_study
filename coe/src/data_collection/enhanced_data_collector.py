#!/usr/bin/env python3
"""
Enhanced Singapore COE Data Collector

This module handles data collection from both Singapore's data.gov.sg APIs:
1. COE Quota Premium dataset (d_22094bf608253d36c0c63b52d852dd6e)
2. COE Bidding Results dataset (d_69b3380ad7e51aff3a7dcc84eba52b8a)

It processes and merges both datasets into a unified training dataset.
"""

import requests
import pandas as pd
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import yaml
import os
from pathlib import Path
import numpy as np

class EnhancedCOEDataCollector:
    """
    Enhanced COE data collector that handles multiple data sources
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the enhanced data collector with configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.data_sources = self.config['data_sources']
        
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
    
    def fetch_dataset_data(self, dataset_id: str, limit: int = 10000, offset: int = 0) -> Dict:
        """
        Fetch data from any dataset using its ID
        
        Args:
            dataset_id: The dataset ID to fetch
            limit: Maximum number of records to fetch
            offset: Starting record number
            
        Returns:
            Dictionary containing the API response
        """
        base_url = "https://data.gov.sg/api/action/datastore_search"
        url = f"{base_url}?resource_id={dataset_id}&limit={limit}&offset={offset}"
        
        try:
            response = requests.get(
                url, 
                timeout=self.config['api']['timeout']
            )
            response.raise_for_status()
            
            data = response.json()
            self.logger.info(f"Successfully fetched {len(data.get('result', {}).get('records', []))} records from dataset {dataset_id}")
            return data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching data from dataset {dataset_id}: {e}")
            raise
    
    def process_coe_quota_premium_data(self, raw_data: Dict) -> pd.DataFrame:
        """
        Process the COE Quota Premium dataset (wide format to long format)
        
        Args:
            raw_data: Raw data from API
            
        Returns:
            Processed DataFrame
        """
        if 'result' not in raw_data or 'records' not in raw_data['result']:
            raise ValueError("Invalid data structure received from API")
        
        records = raw_data['result']['records']
        
        if not records:
            self.logger.warning("No records found in the COE Quota Premium data")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Transform wide format to long format
        df_processed = self._transform_wide_to_long(df)
        
        return df_processed
    
    def process_bidding_results_data(self, raw_data: Dict) -> pd.DataFrame:
        """
        Process the COE Bidding Results dataset (already in long format)
        
        Args:
            raw_data: Raw data from API
            
        Returns:
            Processed DataFrame
        """
        if 'result' not in raw_data or 'records' not in raw_data['result']:
            raise ValueError("Invalid data structure received from API")
        
        records = raw_data['result']['records']
        
        if not records:
            self.logger.warning("No records found in the COE Bidding Results data")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Clean and standardize the data
        df = self._clean_bidding_results_data(df)
        
        return df
    
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
        df_long = self._clean_quota_premium_data(df_long)
        
        return df_long
    
    def _clean_quota_premium_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize the quota premium data
        
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
        df['Bidding_Round'] = df['DataSeries'].apply(self._extract_bidding_round)
        
        # Add source identifier
        df['Data_Source'] = 'quota_premium'
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def _clean_bidding_results_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize the bidding results data
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Convert month column to datetime
        df['Date'] = pd.to_datetime(df['month'] + '-01', format='%Y-%m-%d', errors='coerce')
        
        # Clean numeric columns
        numeric_columns = ['quota', 'bids_success', 'bids_received', 'premium']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filter out rows with invalid dates
        df = df.dropna(subset=['Date'])
        
        # Standardize vehicle class names
        df['Category'] = df['vehicle_class'].apply(self._standardize_category)
        
        # Rename columns for consistency
        df = df.rename(columns={
            'quota': 'Quota',
            'bids_success': 'Successful_Bids',
            'bids_received': 'Bids_Received',
            'premium': 'COE_Price',
            'bidding_no': 'Bidding_Round'
        })
        
        # Create Metric_Type column (all records are from bidding results)
        df['Metric_Type'] = 'Bidding_Result'
        
        # Add source identifier
        df['Data_Source'] = 'bidding_results'
        
        # Select and reorder columns for consistency
        columns_to_keep = ['Date', 'Category', 'Bidding_Round', 'Metric_Type', 'Quota', 'Successful_Bids', 'Bids_Received', 'COE_Price', 'Data_Source']
        df = df[columns_to_keep]
        
        # Sort by date
        df = df.sort_values('Date')
        
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
    
    def _standardize_category(self, vehicle_class: str) -> str:
        """Standardize vehicle class names"""
        if vehicle_class == 'Category A':
            return 'Cat A'
        elif vehicle_class == 'Category B':
            return 'Cat B'
        elif vehicle_class == 'Category C':
            return 'Cat C'
        elif vehicle_class == 'Category D':
            return 'Cat D'
        elif vehicle_class == 'Category E':
            return 'Cat E'
        else:
            return vehicle_class
    
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
        """Extracts the bidding round from the data series string"""
        data_series_lower = data_series.lower()
        if '1st bidding' in data_series_lower:
            return '1'
        elif '2nd bidding' in data_series_lower:
            return '2'
        else:
            return 'Combined'
    
    def merge_datasets(self, quota_premium_df: pd.DataFrame, bidding_results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the two datasets into a unified training dataset
        
        Args:
            quota_premium_df: Processed quota premium data
            bidding_results_df: Processed bidding results data
            
        Returns:
            Merged DataFrame
        """
        self.logger.info("Merging datasets...")
        
        # Create a unified structure
        merged_data = []
        
        # Process quota premium data
        if not quota_premium_df.empty:
            # Pivot quota premium data to get metrics as columns
            quota_premium_pivot = self._pivot_quota_premium_data(quota_premium_df)
            merged_data.append(quota_premium_pivot)
        
        # Process bidding results data
        if not bidding_results_df.empty:
            # Bidding results data is already in the right format
            merged_data.append(bidding_results_df)
        
        # Combine all data
        if merged_data:
            final_df = pd.concat(merged_data, ignore_index=True)
            
            # Sort by date and category
            final_df = final_df.sort_values(['Date', 'Category'])
            
            # Remove duplicates
            final_df = final_df.drop_duplicates(subset=['Date', 'Category', 'Bidding_Round', 'Data_Source'])
            
            self.logger.info(f"Merged dataset has {len(final_df)} records")
            return final_df
        else:
            self.logger.warning("No data to merge")
            return pd.DataFrame()
    
    def _pivot_quota_premium_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot quota premium data to get metrics as columns
        
        Args:
            df: Long format quota premium data
            
        Returns:
            Pivoted DataFrame
        """
        # Filter out unknown categories and metrics
        df = df[(df['Category'] != 'Unknown') & (df['Metric_Type'] != 'Unknown')]
        
        # Pivot the table
        try:
            pivoted_df = df.pivot_table(
                index=['Date', 'Category', 'Bidding_Round'],
                columns='Metric_Type',
                values='Value',
                aggfunc='mean'  # Use mean to handle any duplicates
            ).reset_index()
        except Exception as e:
            self.logger.error(f"Failed to pivot quota premium data: {e}")
            return pd.DataFrame()
        
        # Clean up column names
        pivoted_df.columns.name = None
        
        # Rename columns for clarity
        column_mapping = {
            'Quota Premium': 'COE_Price',
            'Bids Received': 'Bids_Received',
            'Successful Bids': 'Successful_Bids',
            'Prevailing Quota Premium': 'Prevailing_Quota_Premium'
        }
        
        pivoted_df = pivoted_df.rename(columns=column_mapping)
        
        # Add missing columns if they don't exist
        expected_cols = ['Date', 'Category', 'Bidding_Round', 'COE_Price', 'Bids_Received', 'Successful_Bids', 'Quota', 'Prevailing_Quota_Premium']
        for col in expected_cols:
            if col not in pivoted_df.columns:
                pivoted_df[col] = np.nan
        
        # Add source identifier
        pivoted_df['Data_Source'] = 'quota_premium'
        
        return pivoted_df[expected_cols + ['Data_Source']]
    
    def save_merged_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save merged data to CSV file
        
        Args:
            df: DataFrame to save
            filename: Optional filename, will generate timestamp-based name if None
            
        Returns:
            Path to saved file
        """
        raw_data_path = self.config['paths']['raw_data']
        
        # Save timestamped file
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_coe_data_{timestamp}.csv"
        
        filepath = os.path.join(raw_data_path, filename)
        df.to_csv(filepath, index=False)
        self.logger.info(f"Enhanced data saved to {filepath}")
        
        # Save latest file for easy access
        latest_filepath = os.path.join(raw_data_path, "enhanced_coe_data_latest.csv")
        df.to_csv(latest_filepath, index=False)
        self.logger.info(f"Latest enhanced data also saved to {latest_filepath}")
        
        return filepath
    
    def collect_and_merge_data(self) -> pd.DataFrame:
        """
        Main method to collect, process, and merge data from both sources
        
        Returns:
            Merged DataFrame
        """
        self.logger.info("Starting enhanced COE data collection and processing...")
        
        # Collect quota premium data
        quota_premium_data = None
        try:
            quota_premium_id = self.data_sources['coe_quota_premium']['dataset_id']
            quota_premium_raw = self.fetch_dataset_data(quota_premium_id)
            quota_premium_data = self.process_coe_quota_premium_data(quota_premium_raw)
            self.logger.info(f"Processed {len(quota_premium_data)} quota premium records")
        except Exception as e:
            self.logger.error(f"Error processing quota premium data: {e}")
            quota_premium_data = pd.DataFrame()
        
        # Collect bidding results data
        bidding_results_data = None
        try:
            bidding_results_id = self.data_sources['economic_indicators']['dataset_id']
            bidding_results_raw = self.fetch_dataset_data(bidding_results_id)
            bidding_results_data = self.process_bidding_results_data(bidding_results_raw)
            self.logger.info(f"Processed {len(bidding_results_data)} bidding results records")
        except Exception as e:
            self.logger.error(f"Error processing bidding results data: {e}")
            bidding_results_data = pd.DataFrame()
        
        # Merge datasets
        merged_data = self.merge_datasets(quota_premium_data, bidding_results_data)
        
        if not merged_data.empty:
            # Save merged data
            self.save_merged_data(merged_data)
            self.logger.info("Enhanced data collection and processing complete.")
        else:
            self.logger.warning("No data was collected or merged")
        
        return merged_data

if __name__ == "__main__":
    # Example usage
    collector = EnhancedCOEDataCollector()
    data = collector.collect_and_merge_data()
    print(f"Collected {len(data)} records")
    if not data.empty:
        print(data.head())
        print(f"Data sources: {data['Data_Source'].value_counts()}")
        print(f"Categories: {data['Category'].value_counts()}")
        print(f"Date range: {data['Date'].min()} to {data['Date'].max()}") 