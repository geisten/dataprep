"""Fill missing values transformation."""

from typing import Any, Dict, Optional, Union
import pandas as pd

from dataprep.core.base import Transform


class FillMissing(Transform):
    """Fill missing values with a specified strategy.

    Args:
        strategy: Strategy for filling missing values:
                 'mean', 'median', 'mode', 'constant', 'forward', 'backward'.
        value: Value to use when strategy='constant' (default: 0).
        columns: List of columns to apply the transformation to (default: None = all).
    """

    def __init__(
        self,
        strategy: str = 'mean',
        value: Any = 0,
        columns: Optional[list] = None
    ):
        """Initialize FillMissing transformation."""
        super().__init__()
        self.strategy = strategy
        self.value = value
        self.columns = columns
        self._fill_values: Dict[str, Any] = {}

    def fit(self, data: pd.DataFrame) -> 'FillMissing':
        """Fit the transformation by computing fill values.

        Args:
            data: The data to fit.

        Returns:
            FillMissing: The fitted transformation (self).
        """
        cols = self.columns if self.columns is not None else data.columns.tolist()

        for col in cols:
            if col not in data.columns:
                continue

            if self.strategy == 'mean':
                self._fill_values[col] = data[col].mean()
            elif self.strategy == 'median':
                self._fill_values[col] = data[col].median()
            elif self.strategy == 'mode':
                mode_values = data[col].mode()
                self._fill_values[col] = mode_values[0] if len(mode_values) > 0 else None
            elif self.strategy == 'constant':
                self._fill_values[col] = self.value

        self._fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in the data.

        Args:
            data: The data to transform.

        Returns:
            pd.DataFrame: The transformed data.
        """
        result = data.copy()

        if self.strategy in ['forward', 'backward']:
            # Forward/backward fill doesn't need fitting
            method = 'ffill' if self.strategy == 'forward' else 'bfill'
            cols = self.columns if self.columns is not None else result.columns.tolist()
            result[cols] = result[cols].fillna(method=method)
        else:
            # Use learned fill values
            for col, fill_value in self._fill_values.items():
                if col in result.columns:
                    result[col] = result[col].fillna(fill_value)

        return result

    def get_params(self) -> Dict[str, Any]:
        """Get parameters of the transformation.

        Returns:
            Dict[str, Any]: Parameter dictionary.
        """
        return {
            'strategy': self.strategy,
            'value': self.value,
            'columns': self.columns,
        }
