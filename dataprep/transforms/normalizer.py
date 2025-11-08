"""Normalization transformation."""

from typing import Any, Dict, Optional, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from dataprep.core.base import Transform


class Normalizer(Transform):
    """Normalize numerical features.

    Args:
        method: Normalization method ('standard', 'minmax').
               'standard': zero mean and unit variance.
               'minmax': scale to [0, 1] range.
        columns: List of columns to normalize (default: None = all numeric columns).
    """

    def __init__(
        self,
        method: str = 'standard',
        columns: Optional[List[str]] = None
    ):
        """Initialize Normalizer transformation."""
        super().__init__()
        self.method = method
        self.columns = columns
        self._scaler = None
        self._numeric_columns: List[str] = []

    def fit(self, data: pd.DataFrame) -> 'Normalizer':
        """Fit the normalization parameters.

        Args:
            data: The data to fit.

        Returns:
            Normalizer: The fitted transformation (self).
        """
        # Determine which columns to normalize
        if self.columns is not None:
            self._numeric_columns = self.columns
        else:
            self._numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        # Create and fit the scaler
        if self.method == 'standard':
            self._scaler = StandardScaler()
        elif self.method == 'minmax':
            self._scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        if self._numeric_columns:
            self._scaler.fit(data[self._numeric_columns])

        self._fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize the data.

        Args:
            data: The data to transform.

        Returns:
            pd.DataFrame: The transformed data.
        """
        result = data.copy()

        if self._numeric_columns and self._scaler is not None:
            result[self._numeric_columns] = self._scaler.transform(data[self._numeric_columns])

        return result

    def get_params(self) -> Dict[str, Any]:
        """Get parameters of the transformation.

        Returns:
            Dict[str, Any]: Parameter dictionary.
        """
        return {
            'method': self.method,
            'columns': self.columns,
        }
