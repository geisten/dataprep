"""CSV data sink."""

from typing import Any, Dict, Optional
import pandas as pd
from pathlib import Path

from dataprep.core.base import DataSink


class CSVSink(DataSink):
    """Data sink for writing to CSV files.

    Args:
        path: Path to the output CSV file.
        delimiter: Delimiter character (default: ',').
        encoding: File encoding (default: 'utf-8').
        index: Whether to write row index (default: False).
        **kwargs: Additional arguments passed to DataFrame.to_csv().
    """

    def __init__(
        self,
        path: str,
        delimiter: str = ',',
        encoding: str = 'utf-8',
        index: bool = False,
        **kwargs: Any
    ):
        """Initialize CSV sink."""
        self.path = Path(path)
        self.delimiter = delimiter
        self.encoding = encoding
        self.index = index
        self.kwargs = kwargs

        # Create parent directory if it doesn't exist
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, data: pd.DataFrame) -> None:
        """Write data to CSV file.

        Args:
            data: The data to write.
        """
        data.to_csv(
            self.path,
            sep=self.delimiter,
            encoding=self.encoding,
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
        """Get metadata about the CSV sink.

        Returns:
            Dict[str, Any]: Metadata dictionary.
        """
        metadata = {
            'path': str(self.path),
            'delimiter': self.delimiter,
            'encoding': self.encoding,
            'index': self.index,
        }

        if self.path.exists():
            metadata['size_bytes'] = self.path.stat().st_size

        return metadata
