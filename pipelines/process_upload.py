"""
Complete pipeline: Preprocess data and upload to Azure Blob Storage
"""
import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.prepare_dataset import MemoryEfficientPreprocessor
from azure_io import check_azure_connection

def main():
    """Run the complete processing and upload pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Check Azure connection first
    if not check_azure_connection():
        logging.error("Azure connection failed. Please check your credentials.")
        return 1
    
    # Set up paths
    dataset_path = Path("data/raw/PaperDataset")
    output_path = Path("data/processed")
    
    try:
        # Initialize preprocessor
        preprocessor = MemoryEfficientPreprocessor(
            dataset_path=str(dataset_path),
            output_path=str(output_path),
            chunk_size=1000
        )
        
        # Process data
        logging.info("Starting data processing...")
        chunk_paths, total_processed = preprocessor.process_in_chunks()
        
        # Combine chunks
        final_df = preprocessor.combine_chunks(chunk_paths)
        
        # Clean, save locally, and upload to Azure
        results = preprocessor.clean_and_save_final_data(final_df)
        
        logging.info("🎉 Pipeline completed successfully!")
        logging.info(f"✅ Processed {results['total_records']} records")
        logging.info(f"✅ Local file: {results['csv_path']}")
        logging.info(f"✅ Azure file: {results['azure_filename']}")
        
        return 0
        
    except Exception as e:
        logging.error(f"❌ Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
