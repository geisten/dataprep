"""Parquet data source."""

from typing import Any, Dict, Optional, List
import pandas as pd
from pathlib import Path

from dataprep.core.base import DataSource


class ParquetSource(DataSource):
    """Data source for reading Parquet files.

    Args:
        path: Path to the Parquet file.
        columns: List of columns to read (default: None = all columns).
        **kwargs: Additional arguments passed to pd.read_parquet().
    """

    def __init__(
        self,
        path: str,
        columns: Optional[List[str]] = None,
        **kwargs: Any
    ):
        """Initialize Parquet source."""
        self.path = Path(path)
        self.columns = columns
        self.kwargs = kwargs

    def read(self) -> pd.DataFrame:
        """Read data from Parquet file.

        Returns:
            pd.DataFrame: The data read from the Parquet file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.path}")

        return pd.read_parquet(
            self.path,
            columns=self.columns,
            **self.kwargs
        )

    def validate(self) -> bool:
        """Validate that the Parquet file exists and is readable.

        Returns:
            bool: True if valid, False otherwise.
        """
        return self.path.exists() and self.path.is_file()

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the Parquet file.

        Returns:
            Dict[str, Any]: Metadata dictionary.
        """
        metadata = {
            'path': str(self.path),
            'columns': self.columns,
            'exists': self.path.exists(),
        }

        if self.path.exists():
            metadata['size_bytes'] = self.path.stat().st_size

        return metadata
