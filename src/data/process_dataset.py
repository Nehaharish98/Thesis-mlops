import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('parquet_reader.log')
    ]
)
logger = logging.getLogger(__name__)

class ParquetReader:
    """
    A robust parquet file reader with comprehensive error handling and logging.

    This class provides methods to read parquet files with various options
    including column selection, filtering, and batch processing.
    """

    def __init__(self):
        self.logger = logger

    def read_parquet_file(
        self, 
        file_path: Union[str, Path], 
        engine: str = 'pyarrow',
        columns: Optional[List[str]] = None,
        filters: Optional[List[tuple]] = None,
        use_pandas: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Read a parquet file with comprehensive error handling.

        Args:
            file_path: Path to the parquet file
            engine: Engine to use ('pyarrow' or 'fastparquet')
            columns: List of columns to read (None for all)
            filters: List of filter tuples
            use_pandas: Whether to return pandas DataFrame or pyarrow Table

        Returns:
            DataFrame or Table if successful, None if failed
        """
        try:
            file_path = Path(file_path)

            # Validate file exists
            if not file_path.exists():
                self.logger.error(f"File not found: {file_path}")
                return None

            # Validate file extension
            if file_path.suffix.lower() not in ['.parquet', '.parq']:
                self.logger.warning(f"File may not be a parquet file: {file_path}")

            self.logger.info(f"Reading parquet file: {file_path}")
            self.logger.info(f"Engine: {engine}, Columns: {columns}, Filters: {filters}")

            # Read using pandas
            if use_pandas:
                df = pd.read_parquet(
                    file_path, 
                    engine=engine,
                    columns=columns,
                    filters=filters
                )

                self.logger.info(f"Successfully read parquet file. Shape: {df.shape}")
                self.logger.info(f"Columns: {list(df.columns)}")
                self.logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

                return df

            # Read using pyarrow
            else:
                table = pq.read_table(
                    file_path,
                    columns=columns,
                    filters=filters
                )

                self.logger.info(f"Successfully read parquet table. Shape: {table.shape}")
                self.logger.info(f"Schema: {table.schema}")

                return table

        except Exception as e:
            self.logger.error(f"Error reading parquet file {file_path}: {str(e)}")
            return None

    def read_partitioned_dataset(
        self,
        dataset_path: Union[str, Path],
        partition_cols: Optional[List[str]] = None,
        filters: Optional[List[tuple]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Read a partitioned parquet dataset.

        Args:
            dataset_path: Path to the dataset directory
            partition_cols: Columns used for partitioning
            filters: Filters to apply

        Returns:
            Combined DataFrame if successful, None if failed
        """
        try:
            dataset_path = Path(dataset_path)

            if not dataset_path.exists():
                self.logger.error(f"Dataset path not found: {dataset_path}")
                return None

            self.logger.info(f"Reading partitioned dataset: {dataset_path}")

            # Use pyarrow for partitioned datasets
            dataset = pq.ParquetDataset(
                dataset_path,
                filters=filters
            )

            table = dataset.read()
            df = table.to_pandas()

            self.logger.info(f"Successfully read partitioned dataset. Shape: {df.shape}")

            return df

        except Exception as e:
            self.logger.error(f"Error reading partitioned dataset {dataset_path}: {str(e)}")
            return None

    def get_parquet_metadata(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Get metadata from a parquet file.

        Args:
            file_path: Path to the parquet file

        Returns:
            Metadata dictionary if successful, None if failed
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                self.logger.error(f"File not found: {file_path}")
                return None

            # Read metadata using pyarrow
            parquet_file = pq.ParquetFile(file_path)

            metadata = {
                'num_rows': parquet_file.metadata.num_rows,
                'num_columns': parquet_file.metadata.num_columns,
                'num_row_groups': parquet_file.metadata.num_row_groups,
                'created_by': parquet_file.metadata.created_by,
                'schema': parquet_file.schema_arrow,
                'file_size_bytes': file_path.stat().st_size,
                'file_size_mb': file_path.stat().st_size / 1024**2
            }

            self.logger.info(f"Retrieved metadata for {file_path}")
            self.logger.info(f"Rows: {metadata['num_rows']}, Columns: {metadata['num_columns']}")

            return metadata

        except Exception as e:
            self.logger.error(f"Error getting metadata for {file_path}: {str(e)}")
            return None

    def validate_parquet_file(self, file_path: Union[str, Path]) -> bool:
        """
        Validate if a file is a valid parquet file.

        Args:
            file_path: Path to the parquet file

        Returns:
            True if valid parquet file, False otherwise
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                self.logger.error(f"File not found: {file_path}")
                return False

            # Try to read the file metadata
            parquet_file = pq.ParquetFile(file_path)

            # Basic validation checks
            if parquet_file.metadata.num_rows < 0:
                self.logger.error("Invalid number of rows")
                return False

            if parquet_file.metadata.num_columns < 0:
                self.logger.error("Invalid number of columns")
                return False

            self.logger.info(f"Valid parquet file: {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Invalid parquet file {file_path}: {str(e)}")
            return False

# Example usage function
def example_usage():
    """Example of how to use the ParquetReader class."""
    reader = ParquetReader()

    # Example file path - update this to your actual parquet file
    file_path = "data/processed/cloud_network_performance_20250918_174137.parquet"

    # Read the parquet file
    df = reader.read_parquet_file(file_path)

    if df is not None:
        print(f"Successfully loaded data with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst 5 rows:")
        print(df.head())

        # Get basic statistics
        print("\nData types:")
        print(df.dtypes)

        print("\nBasic statistics:")
        print(df.describe())

    # Get metadata
    metadata = reader.get_parquet_metadata(file_path)
    if metadata:
        print(f"\nFile metadata: {metadata}")

    # Validate file
    is_valid = reader.validate_parquet_file(file_path)
    print(f"\nFile is valid: {is_valid}")

if __name__ == "__main__":
    example_usage()