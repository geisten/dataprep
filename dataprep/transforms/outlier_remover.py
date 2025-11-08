"""Outlier removal transformation."""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from dataprep.core.base import Transform


class OutlierRemover(Transform):
    """Remove outliers using IQR or Z-score method.

    Args:
        method: Method for outlier detection ('iqr' or 'zscore').
        threshold: Threshold for outlier detection:
                  - For IQR: multiplier for IQR (default: 1.5)
                  - For Z-score: number of standard deviations (default: 3)
        columns: List of columns to check for outliers (default: None = all numeric).
    """

    def __init__(
        self,
        method: str = 'iqr',
        threshold: float = None,
        columns: Optional[List[str]] = None
    ):
        """Initialize OutlierRemover transformation."""
        super().__init__()
        self.method = method
        self.threshold = threshold if threshold is not None else (1.5 if method == 'iqr' else 3.0)
        self.columns = columns
        self._bounds: Dict[str, tuple] = {}
        self._numeric_columns: List[str] = []

    def fit(self, data: pd.DataFrame) -> 'OutlierRemover':
        """Fit the outlier detector by computing bounds.

        Args:
            data: The data to fit.

        Returns:
            OutlierRemover: The fitted transformation (self).
        """
        # Determine which columns to check
        if self.columns is not None:
            self._numeric_columns = self.columns
        else:
            self._numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        # Compute bounds for each column
        for col in self._numeric_columns:
            if col not in data.columns:
                continue

            if self.method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.threshold * IQR
                upper_bound = Q3 + self.threshold * IQR
            elif self.method == 'zscore':
                mean = data[col].mean()
                std = data[col].std()
                lower_bound = mean - self.threshold * std
                upper_bound = mean + self.threshold * std
            else:
                raise ValueError(f"Unknown outlier detection method: {self.method}")

            self._bounds[col] = (lower_bound, upper_bound)

        self._fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from the data.

        Args:
            data: The data to transform.

        Returns:
            pd.DataFrame: The transformed data with outliers removed.
        """
        result = data.copy()

        # Create a mask for rows to keep
        mask = pd.Series([True] * len(result))

        for col, (lower, upper) in self._bounds.items():
            if col in result.columns:
                mask &= (result[col] >= lower) & (result[col] <= upper)

        return result[mask]

    def get_params(self) -> Dict[str, Any]:
        """Get parameters of the transformation.

        Returns:
            Dict[str, Any]: Parameter dictionary.
        """
        return {
            'method': self.method,
            'threshold': self.threshold,
            'columns': self.columns,
        }
