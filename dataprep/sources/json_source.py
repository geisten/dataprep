"""JSON data source."""

from typing import Any, Dict, Optional
import pandas as pd
from pathlib import Path

from dataprep.core.base import DataSource


class JSONSource(DataSource):
    """Data source for reading JSON files.

    Args:
        path: Path to the JSON file.
        orient: Format of the JSON string ('records', 'index', 'columns', etc.).
        encoding: File encoding (default: 'utf-8').
        **kwargs: Additional arguments passed to pd.read_json().
    """

    def __init__(
        self,
        path: str,
        orient: Optional[str] = None,
        encoding: str = 'utf-8',
        **kwargs: Any
    ):
        """Initialize JSON source."""
        self.path = Path(path)
        self.orient = orient
        self.encoding = encoding
        self.kwargs = kwargs

    def read(self) -> pd.DataFrame:
        """Read data from JSON file.

        Returns:
            pd.DataFrame: The data read from the JSON file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.path}")

        return pd.read_json(
            self.path,
            orient=self.orient,
            encoding=self.encoding,
            **self.kwargs
        )

    def validate(self) -> bool:
        """Validate that the JSON file exists and is readable.

        Returns:
            bool: True if valid, False otherwise.
        """
        return self.path.exists() and self.path.is_file()

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the JSON file.

        Returns:
            Dict[str, Any]: Metadata dictionary.
        """
        metadata = {
            'path': str(self.path),
            'orient': self.orient,
            'encoding': self.encoding,
            'exists': self.path.exists(),
        }

        if self.path.exists():
            metadata['size_bytes'] = self.path.stat().st_size

        return metadata
