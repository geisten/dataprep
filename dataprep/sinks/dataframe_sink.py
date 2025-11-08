"""DataFrame data sink for in-memory storage."""

from typing import Any, Dict
import pandas as pd

from dataprep.core.base import DataSink


class DataFrameSink(DataSink):
    """Data sink for storing data in memory as a pandas DataFrame.

    This is useful for testing or when you want to keep the processed
    data in memory for further processing.

    Args:
        None
    """

    def __init__(self):
        """Initialize DataFrame sink."""
        self.data: pd.DataFrame = None

    def write(self, data: pd.DataFrame) -> None:
        """Write data to internal storage.

        Args:
            data: The data to write.
        """
        self.data = data.copy()

    def get_data(self) -> pd.DataFrame:
        """Get the stored data.

        Returns:
            pd.DataFrame: The stored data.

        Raises:
            ValueError: If no data has been written yet.
        """
        if self.data is None:
            raise ValueError("No data has been written to this sink yet")
        return self.data.copy()

    def validate(self) -> bool:
        """Validate that the sink is ready.

        Returns:
            bool: Always True for DataFrame sink.
        """
        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the stored data.

        Returns:
            Dict[str, Any]: Metadata dictionary.
        """
        if self.data is None:
            return {'has_data': False}

        return {
            'has_data': True,
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': {col: str(dtype) for col, dtype in self.data.dtypes.items()},
            'memory_usage_bytes': self.data.memory_usage(deep=True).sum(),
        }
