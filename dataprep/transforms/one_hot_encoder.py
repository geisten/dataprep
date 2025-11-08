"""One-hot encoding transformation."""

from typing import Any, Dict, List, Optional
import pandas as pd

from dataprep.core.base import Transform


class OneHotEncoder(Transform):
    """One-hot encode categorical features.

    Args:
        columns: List of columns to encode (default: None = all object columns).
        drop_first: Whether to drop the first category to avoid multicollinearity (default: False).
        prefix: Prefix for the new column names (default: None = use column name).
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        drop_first: bool = False,
        prefix: Optional[str] = None
    ):
        """Initialize OneHotEncoder transformation."""
        super().__init__()
        self.columns = columns
        self.drop_first = drop_first
        self.prefix = prefix
        self._categories: Dict[str, List[str]] = {}

    def fit(self, data: pd.DataFrame) -> 'OneHotEncoder':
        """Fit the encoder by learning categories.

        Args:
            data: The data to fit.

        Returns:
            OneHotEncoder: The fitted transformation (self).
        """
        # Determine which columns to encode
        if self.columns is not None:
            cols_to_encode = self.columns
        else:
            cols_to_encode = data.select_dtypes(include=['object', 'category']).columns.tolist()

        # Learn categories for each column
        for col in cols_to_encode:
            if col in data.columns:
                self._categories[col] = data[col].unique().tolist()

        self._fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode the data.

        Args:
            data: The data to transform.

        Returns:
            pd.DataFrame: The transformed data.
        """
        result = data.copy()

        for col in self._categories.keys():
            if col in result.columns:
                # Create one-hot encoded columns
                prefix = self.prefix if self.prefix is not None else col
                dummies = pd.get_dummies(
                    result[col],
                    prefix=prefix,
                    drop_first=self.drop_first
                )

                # Drop original column and add dummies
                result = result.drop(columns=[col])
                result = pd.concat([result, dummies], axis=1)

        return result

    def get_params(self) -> Dict[str, Any]:
        """Get parameters of the transformation.

        Returns:
            Dict[str, Any]: Parameter dictionary.
        """
        return {
            'columns': self.columns,
            'drop_first': self.drop_first,
            'prefix': self.prefix,
        }
