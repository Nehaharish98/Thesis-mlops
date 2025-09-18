import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any
import warnings
import argparse
import gc
warnings.filterwarnings('ignore')

def setup_logging(output_dir: Path = None):
    """Setup logging configuration."""
    log_dir = output_dir if output_dir else Path.cwd()
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'preprocessing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class MemoryEfficientPreprocessor:
    """Memory-efficient preprocessing class."""

    def __init__(self, dataset_path: str, output_path: str = None, chunk_size: int = 1000):
        self.dataset_path = Path(dataset_path).resolve()
        self.output_path = Path(output_path).resolve() if output_path else Path.cwd() / "processed_data"
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size

        # Setup logging
        self.logger = setup_logging(self.output_path)

        # Standard mappings
        self.region_mapping = {
            'us-east-1': 'US_East',
            'eu-west-1': 'EU_West', 
            'ap-southeast-1': 'AP_Southeast',
            'sa-east-1': 'SA_East'
        }

        self.protocol_mapping = {
            'tput_tcp': 'TCP_Throughput',
            'tput_udp': 'UDP_Throughput'
        }

        self.vm_size_mapping = {
            'm3.medium': 'Medium',
            'm3.xlarge': 'XLarge',
            'a2.medium': 'Medium', 
            'a4.xlarge': 'XLarge'
        }

    def find_dataset_path(self):
        """Find the dataset in common locations."""
        if self.dataset_path.exists():
            self.logger.info(f"✅ Using dataset at: {self.dataset_path}")
            return True

        possible_paths = [
            Path.cwd() / "data" / "raw" / "PaperDataset",
            Path.cwd() / "PaperDataset",
            Path.cwd() / "raw" / "PaperDataset"
        ]

        for path in possible_paths:
            if path.exists():
                self.logger.info(f"✅ Found dataset at: {path}")
                self.dataset_path = path.resolve()
                return True

        raise FileNotFoundError(f"Could not find PaperDataset in: {[str(p) for p in possible_paths]}")

    def extract_metadata_from_path(self, filepath: Path) -> Dict[str, Any]:
        """Extract metadata from file path."""
        try:
            parts = filepath.relative_to(self.dataset_path).parts
        except ValueError:
            parts = filepath.parts[-4:]

        metadata = {
            'filepath': str(filepath),
            'provider': parts[0] if len(parts) > 0 else 'Unknown',
            'vm_size': 'Unknown',
            'protocol': 'Unknown',
            'source_region': 'Unknown',
            'dest_region': 'Unknown',
            'timestamp': None
        }

        # Quick parsing for Azure/AWS
        if len(parts) >= 2 and metadata['provider'] in ['Azure', 'AWS']:
            vm_protocol = parts[1]

            # Extract VM size
            vm_match = re.search(r'([am][0-9]\.[a-z]+)', vm_protocol)
            if vm_match:
                metadata['vm_size'] = self.vm_size_mapping.get(vm_match.group(1), vm_match.group(1))

            # Extract protocol
            protocol_match = re.search(r'(tput_[a-z]+)', vm_protocol)
            if protocol_match:
                metadata['protocol'] = self.protocol_mapping.get(protocol_match.group(1), protocol_match.group(1))

        # Extract timestamp from anywhere in path
        timestamp_match = re.search(r'(\d{10})', str(filepath))
        if timestamp_match:
            metadata['timestamp'] = int(timestamp_match.group(1))

        return metadata

    def process_json_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """Process a single JSON file and return records."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            self.logger.debug(f"Error loading {filepath}: {e}")
            return []

        # Get metadata for this file
        metadata = self.extract_metadata_from_path(filepath)

        # Handle different JSON structures
        if isinstance(data, list):
            experiments = data
        elif isinstance(data, dict):
            experiments = [data]
        else:
            return []

        records = []
        for exp in experiments:
            record = metadata.copy()

            # Flatten experiment data (simplified)
            for key, value in exp.items():
                if isinstance(value, (int, float, str)):
                    record[key] = value
                elif isinstance(value, list) and len(value) > 0:
                    if all(isinstance(x, (int, float)) for x in value):
                        record[f"{key}_mean"] = np.mean(value)
                        record[f"{key}_count"] = len(value)
                elif isinstance(value, dict):
                    # Only include first level of nesting
                    for subkey, subval in value.items():
                        if isinstance(subval, (int, float, str)):
                            record[f"{key}_{subkey}"] = subval

            records.append(record)

        return records

    def process_in_chunks(self):
        """Process files in chunks to avoid memory issues."""
        self.find_dataset_path()

        json_files = list(self.dataset_path.rglob("*.json"))
        self.logger.info(f"Found {len(json_files)} JSON files to process")

        if not json_files:
            raise ValueError("No JSON files found")

        # Process in chunks
        all_chunks = []
        chunk_records = []
        processed_count = 0

        for i, filepath in enumerate(json_files):
            # Process file
            file_records = self.process_json_file(filepath)
            chunk_records.extend(file_records)
            processed_count += len(file_records)

            # Save chunk when it reaches chunk_size
            if len(chunk_records) >= self.chunk_size:
                chunk_df = pd.DataFrame(chunk_records)
                chunk_path = self.output_path / f"chunk_{len(all_chunks):03d}.csv"
                chunk_df.to_csv(chunk_path, index=False)
                all_chunks.append(chunk_path)

                self.logger.info(f"Saved chunk {len(all_chunks)} with {len(chunk_records)} records")

                # Clear memory
                chunk_records = []
                del chunk_df
                gc.collect()

            # Progress update
            if (i + 1) % 50 == 0:
                self.logger.info(f"Processed {i + 1}/{len(json_files)} files ({processed_count} experiments)")

        # Save final chunk if any records remain
        if chunk_records:
            chunk_df = pd.DataFrame(chunk_records)
            chunk_path = self.output_path / f"chunk_{len(all_chunks):03d}.csv"
            chunk_df.to_csv(chunk_path, index=False)
            all_chunks.append(chunk_path)
            self.logger.info(f"Saved final chunk with {len(chunk_records)} records")

        self.logger.info(f"Processing complete: {processed_count} experiments in {len(all_chunks)} chunks")
        return all_chunks, processed_count

    def combine_chunks(self, chunk_paths: List[Path]):
        """Combine chunk files into final dataset."""
        self.logger.info("Combining chunks into final dataset...")

        # Read chunks and combine
        dfs = []
        for chunk_path in chunk_paths:
            try:
                df_chunk = pd.read_csv(chunk_path)
                dfs.append(df_chunk)
                self.logger.info(f"Loaded chunk: {chunk_path.name} ({len(df_chunk)} records)")
            except Exception as e:
                self.logger.error(f"Error loading chunk {chunk_path}: {e}")

        if not dfs:
            raise ValueError("No chunks could be loaded")

        # Combine all chunks
        final_df = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"Combined {len(dfs)} chunks into {len(final_df)} total records")

        # Clean up chunk files
        for chunk_path in chunk_paths:
            try:
                chunk_path.unlink()
            except Exception as e:
                self.logger.debug(f"Could not delete chunk {chunk_path}: {e}")

        return final_df

    def clean_and_save_final_data(self, df: pd.DataFrame):
        """Clean and save the final dataset."""
        self.logger.info("Cleaning final dataset...")

        # Basic cleaning
        df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]

        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        self.logger.info(f"Removed {initial_rows - len(df)} duplicate records")

        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')

        # Convert timestamp if present
        if 'timestamp' in df.columns:
            df['experiment_datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')

        # Save final files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV
        csv_path = self.output_path / f"cloud_network_performance_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"✅ Saved CSV: {csv_path} ({len(df)} records)")

        # Parquet (if possible)
        try:
            parquet_path = self.output_path / f"cloud_network_performance_{timestamp}.parquet"
            df.to_parquet(parquet_path, index=False)
            self.logger.info(f"✅ Saved Parquet: {parquet_path}")
        except Exception as e:
            self.logger.info(f"Parquet save skipped (install pyarrow): {e}")

        # Sample files
        if len(df) > 1000:
            sample_df = df.sample(n=1000, random_state=42)
            sample_path = self.output_path / "sample_1000.csv"
            sample_df.to_csv(sample_path, index=False)
            self.logger.info(f"✅ Saved sample: {sample_path}")

        # Summary
        summary_path = self.output_path / f"dataset_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Dataset Summary\n")
            f.write(f"===============\n")
            f.write(f"Total Records: {len(df)}\n")
            f.write(f"Total Features: {len(df.columns)}\n")
            f.write(f"Providers: {df['provider'].value_counts().to_dict()}\n")
            if 'protocol' in df.columns:
                f.write(f"Protocols: {df['protocol'].value_counts().to_dict()}\n")
            f.write(f"Memory Usage: {df.memory_usage().sum() / 1024**2:.2f} MB\n")

        return {
            'csv_path': str(csv_path),
            'summary_path': str(summary_path),
            'total_records': len(df),
            'total_features': len(df.columns)
        }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Memory-efficient preprocessing")
    parser.add_argument('--input', '-i', required=True, help='Input dataset path')
    parser.add_argument('--output', '-o', default='./data/processed', help='Output directory')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Chunk size for processing')
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_args()

    print("🚀 Memory-Efficient Cloud Network Dataset Preprocessing")
    print("=" * 60)

    try:
        preprocessor = MemoryEfficientPreprocessor(
            dataset_path=args.input,
            output_path=args.output,
            chunk_size=args.chunk_size
        )

        # Process in chunks
        chunk_paths, total_processed = preprocessor.process_in_chunks()

        # Combine chunks
        final_df = preprocessor.combine_chunks(chunk_paths)

        # Clean and save
        results = preprocessor.clean_and_save_final_data(final_df)

        print(f"\n🎉 SUCCESS!")
        print(f"✅ Processed {results['total_records']} records")
        print(f"✅ Generated {results['total_features']} features")
        print(f"✅ Saved to: {results['csv_path']}")
        print(f"\n📁 Check your output directory: {args.output}")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())