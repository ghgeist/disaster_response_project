#!/usr/bin/env python3
"""
Data preparation script for disaster response classification.

This script runs the ETL pipeline to process raw disaster response data
and prepare it for machine learning.
"""

# Standard library imports
import argparse
import logging
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Local imports
from disasterproject.data.loader import prepare_data


def setup_logging(level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('data_preparation.log')
        ]
    )


def main():
    """Main function to run data preparation."""
    parser = argparse.ArgumentParser(description='Prepare disaster response data for ML')
    
    parser.add_argument(
        '--messages-file',
        default='data/01_raw/disaster_messages.csv',
        help='Path to disaster_messages.csv file'
    )
    
    parser.add_argument(
        '--categories-file',
        default='data/01_raw/disaster_categories.csv',
        help='Path to disaster_categories.csv file'
    )
    
    parser.add_argument(
        '--output-csv',
        default='data/02_stg/stg_disaster_messages.csv',
        help='Path to save processed CSV file'
    )
    
    parser.add_argument(
        '--output-db',
        default='data/02_stg/stg_disaster_response.db',
        help='Path to save SQLite database'
    )
    
    parser.add_argument(
        '--definitions-file',
        default='data/02_stg/stg_disaster_messages_definitions.csv',
        help='Path to save column definitions CSV file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    try:
        logging.info("Starting data preparation process")
        logging.info(f"Messages file: {args.messages_file}")
        logging.info(f"Categories file: {args.categories_file}")
        logging.info(f"Output CSV: {args.output_csv}")
        logging.info(f"Output DB: {args.output_db}")
        logging.info(f"Definitions file: {args.definitions_file}")
        
        # Run data preparation
        df = prepare_data(
            messages_filepath=args.messages_file,
            categories_filepath=args.categories_file,
            output_csv_path=args.output_csv,
            output_db_path=args.output_db,
            definitions_path=args.definitions_file
        )
        
        logging.info(f"Data preparation completed successfully!")
        logging.info(f"Processed {len(df)} records")
        logging.info(f"Data shape: {df.shape}")
        
        print(f"\n✅ Data preparation completed successfully!")
        print(f"📊 Processed {len(df):,} records")
        print(f"📁 Files created:")
        print(f"   - CSV: {args.output_csv}")
        print(f"   - Database: {args.output_db}")
        print(f"   - Definitions: {args.definitions_file}")
        
    except Exception as e:
        logging.error(f"Data preparation failed: {e}")
        print(f"\n❌ Data preparation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
