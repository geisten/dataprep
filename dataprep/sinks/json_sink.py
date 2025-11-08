"""JSON data sink."""

from typing import Any, Dict, Optional
import pandas as pd
from pathlib import Path

from dataprep.core.base import DataSink


class JSONSink(DataSink):
    """Data sink for writing to JSON files.

    Args:
        path: Path to the output JSON file.
        orient: Format of the JSON string ('records', 'index', 'columns', etc.).
        encoding: File encoding (default: 'utf-8').
        indent: Indentation level for pretty printing (default: 2).
        **kwargs: Additional arguments passed to DataFrame.to_json().
    """

    def __init__(
        self,
        path: str,
        orient: str = 'records',
        encoding: str = 'utf-8',
        indent: Optional[int] = 2,
        **kwargs: Any
    ):
        """Initialize JSON sink."""
        self.path = Path(path)
        self.orient = orient
        self.encoding = encoding
        self.indent = indent
        self.kwargs = kwargs

        # Create parent directory if it doesn't exist
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, data: pd.DataFrame) -> None:
        """Write data to JSON file.

        Args:
            data: The data to write.
        """
        data.to_json(
            self.path,
            orient=self.orient,
            indent=self.indent,
            **self.kwargs
        )

    def validate(self) -> bool:
        """Validate that the output directory exists and is writable.

        Returns:
            bool: True if valid, False otherwise.
        """
        return self.path.parent.exists() and self.path.parent.is_dir()

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the JSON sink.

        Returns:
            Dict[str, Any]: Metadata dictionary.
        """
        metadata = {
            'path': str(self.path),
            'orient': self.orient,
            'encoding': self.encoding,
            'indent': self.indent,
        }

        if self.path.exists():
            metadata['size_bytes'] = self.path.stat().st_size

        return metadata
