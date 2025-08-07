"""
Data processing module for preschool demand forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import re

class DataProcessor:
    def __init__(self):
        """Initialize data processor"""
        self.processed_data = {}
        
    def process_population_data(self, population_file: str) -> pd.DataFrame:
        """
        Process population data from SingStat Excel file
        
        Args:
            population_file: Path to population Excel file
            
        Returns:
            Processed population DataFrame
        """
        try:
            # Read Excel file (this would need to be adapted based on actual file structure)
            # For demo purposes, we'll create synthetic data
            return self._create_synthetic_population_data()
        except Exception as e:
            print(f"Error processing population data: {e}")
            return self._create_synthetic_population_data()
    
    def process_bto_data(self, bto_file: str) -> pd.DataFrame:
        """
        Process BTO mapping data
        
        Args:
            bto_file: Path to BTO mapping CSV file
            
        Returns:
            Processed BTO DataFrame
        """
        try:
            bto_data = pd.read_csv(bto_file)
            
            # Clean and validate data
            bto_data = bto_data.dropna(subset=['Subzone', 'Total number of units'])
            
            # Convert completion year to integer
            bto_data['Estimated completion year'] = pd.to_numeric(
                bto_data['Estimated completion year'], errors='coerce'
            )
            
            # Calculate additional demand from BTO projects
            bto_data['Additional_population'] = (
                bto_data['Total number of units'] * 2.5 * 0.15
            )
            bto_data['Additional_demand'] = (
                bto_data['Additional_population'] * 0.8
            )
            
            return bto_data
            
        except Exception as e:
            print(f"Error processing BTO data: {e}")
            return pd.DataFrame()
    
    def process_preschool_data(self, preschool_file: str) -> pd.DataFrame:
        """
        Process preschool centers data
        
        Args:
            preschool_file: Path to preschool centers CSV file
            
        Returns:
            Processed preschool DataFrame
        """
        try:
            preschool_data = pd.read_csv(preschool_file)
            
            # Clean data
            preschool_data = preschool_data.dropna(subset=['centre_address', 'postal_code'])
            
            # Extract subzone information from address
            preschool_data['extracted_subzone'] = preschool_data['centre_address'].apply(
                self._extract_subzone_from_address
            )
            
            # Calculate capacity (assuming 100 children per center)
            preschool_data['estimated_capacity'] = 100
            
            return preschool_data
            
        except Exception as e:
            print(f"Error processing preschool data: {e}")
            return pd.DataFrame()
    
    def _extract_subzone_from_address(self, address: str) -> str:
        """
        Extract subzone information from address
        
        Args:
            address: Center address string
            
        Returns:
            Extracted subzone or empty string
        """
        if pd.isna(address):
            return ""
        
        address = str(address).lower()
        
        # Common subzone patterns
        subzone_patterns = [
            r'woodlands\s+(east|west|central)',
            r'tampines\s+(east|west|north|central)',
            r'jurong\s+(east|west|central)',
            r'ang\s+mo\s+kio',
            r'hougang',
            r'sengkang',
            r'punggol',
            r'bedok\s+(north|south)',
            r'bukit\s+(batok|panjang|timah)',
            r'clementi',
            r'choa\s+chu\s+kang'
        ]
        
        for pattern in subzone_patterns:
            match = re.search(pattern, address)
            if match:
                return match.group(0).title()
        
        return ""
    
    def _create_synthetic_population_data(self) -> pd.DataFrame:
        """
        Create synthetic population data for demonstration
        
        Returns:
            Synthetic population DataFrame
        """
        # Get subzones from BTO data
        bto_data = pd.read_csv('../data/btomapping.csv')
        subzones = bto_data[['Subzone', 'Planning area', 'Region']].drop_duplicates()
        
        # Create population data for 2020-2025
        years = range(2020, 2026)
        data = []
        
        for year in years:
            for _, subzone_info in subzones.iterrows():
                subzone = subzone_info['Subzone']
                planning_area = subzone_info['Planning area']
                region = subzone_info['Region']
                
                # Base population with some realistic variation
                base_pop = np.random.randint(100, 800)
                
                # Add growth trend
                growth_factor = 1 + (year - 2020) * 0.015
                
                # Add regional variation
                regional_factors = {
                    'North-East Region': 1.1,  # Higher growth in newer areas
                    'West Region': 1.05,
                    'East Region': 1.03,
                    'North Region': 1.02,
                    'Central Region': 0.98  # Lower growth in mature areas
                }
                
                regional_factor = regional_factors.get(region, 1.0)
                
                # Calculate final population
                population = int(base_pop * growth_factor * regional_factor)
                
                data.append({
                    'Year': year,
                    'Subzone': subzone,
                    'Planning_area': planning_area,
                    'Region': region,
                    'Population_18m_6y': population
                })
        
        return pd.DataFrame(data)
    
    def map_postal_codes_to_subzones(self, postal_codes: List[str]) -> Dict[str, str]:
        """
        Map postal codes to subzones using external data source
        
        Args:
            postal_codes: List of postal codes
            
        Returns:
            Dictionary mapping postal codes to subzones
        """
        # This would typically use a postal code to subzone mapping service
        # For demo purposes, return empty mapping
        return {}
    
    def validate_data_quality(self, data: pd.DataFrame, data_type: str) -> Dict[str, any]:
        """
        Validate data quality and return quality metrics
        
        Args:
            data: DataFrame to validate
            data_type: Type of data ('population', 'bto', 'preschool')
            
        Returns:
            Dictionary with quality metrics
        """
        quality_metrics = {
            'total_records': len(data),
            'missing_values': data.isnull().sum().to_dict(),
            'duplicate_records': data.duplicated().sum(),
            'data_completeness': (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        }
        
        if data_type == 'population':
            quality_metrics['year_range'] = {
                'min': data['Year'].min() if 'Year' in data.columns else None,
                'max': data['Year'].max() if 'Year' in data.columns else None
            }
        
        elif data_type == 'bto':
            quality_metrics['completion_years'] = {
                'min': data['Estimated completion year'].min() if 'Estimated completion year' in data.columns else None,
                'max': data['Estimated completion year'].max() if 'Estimated completion year' in data.columns else None
            }
        
        elif data_type == 'preschool':
            quality_metrics['service_models'] = data['service_model'].value_counts().to_dict() if 'service_model' in data.columns else {}
        
        return quality_metrics 