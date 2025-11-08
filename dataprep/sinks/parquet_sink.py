"""Parquet data sink."""

from typing import Any, Dict, Optional
import pandas as pd
from pathlib import Path

from dataprep.core.base import DataSink


class ParquetSink(DataSink):
    """Data sink for writing to Parquet files.

    Args:
        path: Path to the output Parquet file.
        compression: Compression to use ('snappy', 'gzip', 'brotli', None).
        index: Whether to write row index (default: False).
        **kwargs: Additional arguments passed to DataFrame.to_parquet().
    """

    def __init__(
        self,
        path: str,
        compression: Optional[str] = 'snappy',
        index: bool = False,
        **kwargs: Any
    ):
        """Initialize Parquet sink."""
        self.path = Path(path)
        self.compression = compression
        self.index = index
        self.kwargs = kwargs

        # Create parent directory if it doesn't exist
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, data: pd.DataFrame) -> None:
        """Write data to Parquet file.

        Args:
            data: The data to write.
        """
        data.to_parquet(
            self.path,
            compression=self.compression,
            index=self.index,
            **self.kwargs
        )

    def validate(self) -> bool:
        """Validate that the output directory exists and is writable.

        Returns:
            bool: True if valid, False otherwise.
        """
        return self.path.parent.exists() and self.path.parent.is_dir()

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the Parquet sink.

        Returns:
            Dict[str, Any]: Metadata dictionary.
        """
        metadata = {
            'path': str(self.path),
            'compression': self.compression,
            'index': self.index,
        }

        if self.path.exists():
            metadata['size_bytes'] = self.path.stat().st_size

        return metadata
