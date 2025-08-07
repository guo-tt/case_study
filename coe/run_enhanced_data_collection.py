#!/usr/bin/env python3
"""
Enhanced COE Data Collection and Training Data Preparation

This script runs the complete pipeline to:
1. Collect data from both COE datasets
2. Merge and process the data
3. Prepare training datasets for machine learning models
"""

import sys
import os
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_collection.enhanced_data_collector import EnhancedCOEDataCollector
from src.data_collection.training_data_preparator import TrainingDataPreparator

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/enhanced_data_collection.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Main function to run the enhanced data collection pipeline"""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Starting Enhanced COE Data Collection Pipeline")
    logger.info("=" * 60)
    
    try:
        # Step 1: Collect and merge data from both sources
        logger.info("Step 1: Collecting data from both COE datasets...")
        collector = EnhancedCOEDataCollector()
        merged_data = collector.collect_and_merge_data()
        
        if merged_data.empty:
            logger.error("No data was collected. Exiting.")
            return False
        
        logger.info(f"Successfully collected and merged {len(merged_data)} records")
        logger.info(f"Data sources: {merged_data['Data_Source'].value_counts().to_dict()}")
        logger.info(f"Categories: {merged_data['Category'].value_counts().to_dict()}")
        logger.info(f"Date range: {merged_data['Date'].min()} to {merged_data['Date'].max()}")
        
        # Step 2: Prepare training datasets
        logger.info("Step 2: Preparing training datasets...")
        preparator = TrainingDataPreparator()
        training_datasets = preparator.prepare_all_training_datasets()
        
        if not training_datasets:
            logger.error("No training datasets were prepared. Exiting.")
            return False
        
        # Step 3: Report results
        logger.info("Step 3: Reporting results...")
        logger.info(f"Successfully prepared {len(training_datasets)} training datasets:")
        
        for dataset_name, dataset in training_datasets.items():
            if isinstance(dataset, tuple):
                logger.info(f"  {dataset_name}: {dataset[0].shape} sequences")
            elif isinstance(dataset, dict):
                logger.info(f"  {dataset_name}: {dataset['X_train'].shape[0]} training samples")
        
        logger.info("=" * 60)
        logger.info("Enhanced COE Data Collection Pipeline Completed Successfully!")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in enhanced data collection pipeline: {e}")
        logger.exception("Full traceback:")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 