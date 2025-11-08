"""CSV data source."""

from typing import Any, Dict, Optional
import pandas as pd
from pathlib import Path

from dataprep.core.base import DataSource


class CSVSource(DataSource):
    """Data source for reading CSV files.

    Args:
        path: Path to the CSV file.
        delimiter: Delimiter character (default: ',').
        encoding: File encoding (default: 'utf-8').
        header: Row number to use as column names (default: 0).
        **kwargs: Additional arguments passed to pd.read_csv().
    """

    def __init__(
        self,
        path: str,
        delimiter: str = ',',
        encoding: str = 'utf-8',
        header: Optional[int] = 0,
        **kwargs: Any
    ):
        """Initialize CSV source."""
        self.path = Path(path)
        self.delimiter = delimiter
        self.encoding = encoding
        self.header = header
        self.kwargs = kwargs

    def read(self) -> pd.DataFrame:
        """Read data from CSV file.

        Returns:
            pd.DataFrame: The data read from the CSV file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.path}")

        return pd.read_csv(
            self.path,
            delimiter=self.delimiter,
            encoding=self.encoding,
            header=self.header,
            **self.kwargs
        )

    def validate(self) -> bool:
        """Validate that the CSV file exists and is readable.

        Returns:
            bool: True if valid, False otherwise.
        """
        return self.path.exists() and self.path.is_file()

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the CSV file.

        Returns:
            Dict[str, Any]: Metadata dictionary.
        """
        metadata = {
            'path': str(self.path),
            'delimiter': self.delimiter,
            'encoding': self.encoding,
            'exists': self.path.exists(),
        }

        if self.path.exists():
            metadata['size_bytes'] = self.path.stat().st_size

        return metadata
