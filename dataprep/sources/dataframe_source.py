"""DataFrame data source for in-memory data."""

from typing import Any, Dict
import pandas as pd

from dataprep.core.base import DataSource


class DataFrameSource(DataSource):
    """Data source for in-memory pandas DataFrames.

    Args:
        data: The pandas DataFrame to use as a source.
    """

    def __init__(self, data: pd.DataFrame):
        """Initialize DataFrame source."""
        self.data = data.copy()

    def read(self) -> pd.DataFrame:
        """Read data from the DataFrame.

        Returns:
            pd.DataFrame: A copy of the stored DataFrame.
        """
        return self.data.copy()

    def validate(self) -> bool:
        """Validate that the DataFrame is valid.

        Returns:
            bool: True if valid, False otherwise.
        """
        return isinstance(self.data, pd.DataFrame) and not self.data.empty

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the DataFrame.

        Returns:
            Dict[str, Any]: Metadata dictionary.
        """
        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': {col: str(dtype) for col, dtype in self.data.dtypes.items()},
            'memory_usage_bytes': self.data.memory_usage(deep=True).sum(),
        }
