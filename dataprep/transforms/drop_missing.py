"""Drop missing values transformation."""

from typing import Any, Dict, Optional
import pandas as pd

from dataprep.core.base import Transform


class DropMissing(Transform):
    """Drop rows or columns with missing values.

    Args:
        axis: 0 to drop rows, 1 to drop columns (default: 0).
        threshold: Fraction of non-missing values required to keep (default: None).
                  If threshold=0.8, rows/columns with less than 80% non-missing
                  values will be dropped.
        subset: List of columns to consider for row-wise dropping (default: None).
    """

    def __init__(
        self,
        axis: int = 0,
        threshold: Optional[float] = None,
        subset: Optional[list] = None
    ):
        """Initialize DropMissing transformation."""
        super().__init__()
        self.axis = axis
        self.threshold = threshold
        self.subset = subset

    def fit(self, data: pd.DataFrame) -> 'DropMissing':
        """Fit the transformation (no-op for DropMissing).

        Args:
            data: The data to fit.

        Returns:
            DropMissing: The fitted transformation (self).
        """
        self._fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Drop rows or columns with missing values.

        Args:
            data: The data to transform.

        Returns:
            pd.DataFrame: The transformed data.
        """
        if self.threshold is not None:
            thresh = int(self.threshold * len(data.columns if self.axis == 0 else data.index))
            return data.dropna(axis=self.axis, thresh=thresh, subset=self.subset)
        else:
            return data.dropna(axis=self.axis, subset=self.subset)

    def get_params(self) -> Dict[str, Any]:
        """Get parameters of the transformation.

        Returns:
            Dict[str, Any]: Parameter dictionary.
        """
        return {
            'axis': self.axis,
            'threshold': self.threshold,
            'subset': self.subset,
        }
